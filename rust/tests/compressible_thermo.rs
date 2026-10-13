// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! CMP-08's both-builds gate: the device kernels of SPEC-LIT §122.5
//! (`cuda/compressible.cu`) against the host twins of
//! `src/compressible/thermo.rs`, through the public API
//! `ofgpu::compressible::thermo`. The four names are the lib tests'
//! names, and they run in BOTH builds: in the default build the twins
//! agree to four ulp, under `single` the same runs are the f32
//! measurement, with §122.2's control asserting the 1 Pa perturbation's
//! error at 300 K stays above 1e-3.
//! No GPL-licensed source was consulted.

#![allow(clippy::excessive_precision)]

use ofgpu::compressible::thermo::{
    gas_state, sound_mach, transport, PerfectGas, ThermoKernels, Viscosity,
};
use ofgpu::device::Gpu;
use ofgpu::{Scalar, Vec3};

/// The process's device, if the box has one: every GPU test returns early
/// rather than failing.
fn gpu() -> Option<Gpu> {
    Gpu::new(0).ok()
}

/// SplitMix64, seed 20261013; `u()` is uniform on [0, 1) as
/// `(x >> 11) 2^-53`.
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

/// The states of §122.5's gate table: 4096 random states, each drawn in
/// the order `T = 200 + 1800 u`, `p = -5e4 + 5.5e5 u`, then one draw per
/// velocity component on [-800, 800], all formed in f64 and cast once to
/// `Scalar`; then state 4096, the rest state `T = 300`, `p = 0`,
/// `U = 0`. So `n = 4097`.
fn fixture() -> (Vec<Scalar>, Vec<Scalar>, Vec<Vec3>) {
    let mut r = SplitMix64(20261013);
    let mut t = Vec::with_capacity(4097);
    let mut p = Vec::with_capacity(4097);
    let mut u = Vec::with_capacity(4097);
    for _ in 0..4096 {
        let tf = 200.0 + 1800.0 * r.u();
        let pf = -5.0e4 + 5.5e5 * r.u();
        let ux = -800.0 + 1600.0 * r.u();
        let uy = -800.0 + 1600.0 * r.u();
        let uz = -800.0 + 1600.0 * r.u();
        t.push(tf as Scalar);
        p.push(pf as Scalar);
        u.push(Vec3::new(ux as Scalar, uy as Scalar, uz as Scalar));
    }
    t.push(300.0 as Scalar);
    p.push(0.0 as Scalar);
    u.push(Vec3::ZERO);
    (t, p, u)
}

/// The two gases of the gate table: Sutherland air and the same gas at a
/// constant viscosity, both about `p_op = 101325` Pa.
fn gases() -> Vec<(&'static str, PerfectGas)> {
    let suth = PerfectGas::air(101325.0);
    let konst = PerfectGas { viscosity: Viscosity::Constant { mu: 1.8e-5 }, ..suth };
    vec![("sutherland air", suth), ("constant 1.8e-5", konst)]
}

/// §122.5's ulp distance: the count of representable values between two
/// finite numbers of one sign, which is the difference of their bit
/// patterns read as integers. A different sign or a non-finite value
/// fails here, naming the field and the state.
fn ulp_distance(a: Scalar, b: Scalar, field: &str, i: usize) -> u64 {
    assert!(
        a.is_finite() && b.is_finite(),
        "{field} at state {i}: non-finite, device {a:e} host {b:e}"
    );
    assert!(
        (a == 0.0 && b == 0.0) || a.is_sign_negative() == b.is_sign_negative(),
        "{field} at state {i}: the signs differ, device {a:e} host {b:e}"
    );
    a.to_bits().abs_diff(b.to_bits()) as u64
}

/// The distance from `x` to the next representable number above it - the
/// one-ulp bar for the round trip through (122.10) and (122.13).
fn ulp(x: Scalar) -> Scalar {
    Scalar::from_bits(x.to_bits() + 1) - x
}

/// The R-class relative tolerance (D-F32-1): `1e-14` in the default
/// build; under `single`, `max(1e-14, min(1e-14 * 2^29, 1e-5))`.
fn r_tol() -> f64 {
    if cfg!(feature = "single") {
        (1.0e-14 * (1u64 << 29) as f64).min(1.0e-5).max(1.0e-14)
    } else {
        1.0e-14
    }
}

/// `|got - want| / |want|`, in f64 in both builds - the R rows compare in
/// f64 (§122.5).
fn rel(got: Scalar, want: f64) -> f64 {
    (f64::from(got) - want).abs() / want.abs()
}

/// The six fields the device twin returns, in launcher order.
const FIELDS: [&str; 6] = ["psi", "rho", "mu", "kappa", "c", "mach"];

/// §122.5's first gate: the three kernels against the host twins, both
/// gases, all 4097 states, each field within four ulp - and `M` exactly 0
/// at the rest state.
#[test]
fn gpu_matches_host_to_4ulp() {
    let Some(g) = gpu() else { return };
    let k = ThermoKernels::new(&g).expect("kernels");
    let (t, p, u) = fixture();
    let n = t.len();
    let dt = g.upload(&t).expect("upload t");
    let dp = g.upload(&p).expect("upload p");
    let du = g.upload(&u).expect("upload u");
    let mut dpsi = g.zeros::<Scalar>(n).expect("alloc psi");
    let mut drho = g.zeros::<Scalar>(n).expect("alloc rho");
    let mut dmu = g.zeros::<Scalar>(n).expect("alloc mu");
    let mut dkappa = g.zeros::<Scalar>(n).expect("alloc kappa");
    let mut dc = g.zeros::<Scalar>(n).expect("alloc c");
    let mut dmach = g.zeros::<Scalar>(n).expect("alloc mach");
    for (name, gas) in gases() {
        let c = gas.constants();
        gas_state(&g, &k, &c, &mut dpsi, &mut drho, &dp, &dt, n).expect("gas_state");
        transport(&g, &k, &c, &mut dmu, &mut dkappa, &dt, n).expect("transport");
        sound_mach(&g, &k, &c, &mut dc, &mut dmach, &du, &dt, n).expect("sound_mach");
        g.sync().expect("sync");
        let psi = g.download(&dpsi).expect("download psi");
        let rho = g.download(&drho).expect("download rho");
        let mu = g.download(&dmu).expect("download mu");
        let kappa = g.download(&dkappa).expect("download kappa");
        let cs = g.download(&dc).expect("download c");
        let mach = g.download(&dmach).expect("download mach");
        let mut worst = [0u64; 6];
        for i in 0..n {
            let want = [
                c.psi(t[i]),
                c.rho(p[i], t[i]),
                c.mu(t[i]),
                c.kappa(t[i]),
                c.sound_speed(t[i]),
                c.mach(u[i], t[i]),
            ];
            let got = [psi[i], rho[i], mu[i], kappa[i], cs[i], mach[i]];
            for f in 0..6 {
                let d = ulp_distance(got[f], want[f], FIELDS[f], i);
                worst[f] = worst[f].max(d);
                assert!(
                    d <= 4,
                    "{}: at state {i} the device gave {:e} and the host {:e} \
                     - {d} ulp",
                    FIELDS[f],
                    got[f],
                    want[f]
                );
            }
        }
        assert!(
            mach[n - 1] == 0.0,
            "at rest M is exactly 0, got {:e}",
            mach[n - 1]
        );
        println!(
            "  {name}: worst ulp  psi {}  rho {}  mu {}  kappa {}  c {}  \
             mach {}",
            worst[0], worst[1], worst[2], worst[3], worst[4], worst[5]
        );
    }
}

/// §122.5's second gate: Sutherland air lands on the four digits White and
/// the Standard Atmosphere print at 273.15 K, on the oracle's three values
/// to the R-class tolerance, and the constant gas returns its number at
/// every temperature.
#[test]
fn sutherland_air_at_273_15_is_1_716e_5() {
    let c = PerfectGas::air(101325.0).constants();
    let m273 = c.mu(273.15);
    assert!(
        (f64::from(m273) - 1.716e-5).abs() < 5e-9,
        "mu(273.15) = {m273:e}, the four digits White and the Standard \
         Atmosphere print"
    );
    assert!(
        rel(m273, 1.716079266245527e-05) <= r_tol(),
        "mu(273.15) = {m273:e}, wanted 1.716079266245527e-05"
    );
    let m300 = c.mu(300.0);
    assert!(
        rel(m300, 1.8460015185931457e-05) <= r_tol(),
        "mu(300) = {m300:e}, wanted 1.8460015185931457e-05"
    );
    let k300 = c.kappa(300.0);
    assert!(
        rel(k300, 0.026122028560991366) <= r_tol(),
        "kappa(300) = {k300:e}, wanted 0.026122028560991366"
    );
    let base = PerfectGas::air(101325.0);
    let konst = PerfectGas { viscosity: Viscosity::Constant { mu: 1.8e-5 }, ..base };
    let kc = konst.constants();
    let want: Scalar = 1.8e-5;
    for t in [200.0, 300.0, 2000.0] {
        assert!(
            kc.mu(t).to_bits() == want.to_bits(),
            "the constant gas at {t} K gave {:e}, wanted 1.8e-5",
            kc.mu(t)
        );
        assert!(
            kc.kappa(t).to_bits() == (kc.mu0 * kc.cpr).to_bits(),
            "kappa({t}) is mu0 cpr bit for bit, got {:e}",
            kc.kappa(t)
        );
    }
}

/// §122.5's third gate: `R_s` and `c_p` on the oracle to `1e-15` relative,
/// `rho` and the sound speed on theirs to the R-class tolerance, the round
/// trip `p -> rho -> p` within four ulp over all 4097 states, the
/// stated-`c_p` refusals of §122.1, and `validate`'s refusals by name.
#[test]
fn ideal_gas_round_trip() {
    let gas = PerfectGas::air(101325.0);
    assert!(
        (gas.r_s() - 287.0550227690948).abs() / 287.0550227690948 <= 1e-15,
        "r_s = {}, wanted 287.0550227690948",
        gas.r_s()
    );
    assert!(
        (gas.cp() - 1004.692579691832).abs() / 1004.692579691832 <= 1e-15,
        "c_p = {}, wanted 1004.692579691832",
        gas.cp()
    );
    assert_eq!(gas.cv(), gas.cp() - gas.r_s(), "c_v is c_p - R_s exactly");
    let c = gas.constants();
    assert!(
        rel(c.rho(0.0, 288.15), 1.2249908312368587) <= r_tol(),
        "rho(0, 288.15) = {:e}, wanted 1.2249908312368587",
        c.rho(0.0, 288.15)
    );
    assert!(
        rel(c.sound_speed(288.15), 340.29526405061904) <= r_tol(),
        "c(288.15) = {:e}, wanted 340.29526405061904",
        c.sound_speed(288.15)
    );
    assert!(
        rel(c.sound_speed(300.0), 347.2219888817812) <= r_tol(),
        "c(300) = {:e}, wanted 347.2219888817812",
        c.sound_speed(300.0)
    );
    // The round trip over the 4097 states: (122.13) inverts (122.10) to
    // within four ulp of p_abs.
    let (t, p, _) = fixture();
    for (i, (&ti, &pi)) in t.iter().zip(p.iter()).enumerate() {
        let a = pi + c.pop;
        let back = c.p_from_rho(c.rho(pi, ti), ti);
        assert!(
            (back - pi).abs() <= 4.0 * ulp(a),
            "state {i}: p -> rho -> p gave {back:e}, wanted {pi:e}"
        );
    }
    // The stated-c_p rule of §122.1.
    assert!(gas.check_stated_cp(gas.cp()).is_ok());
    assert!(gas.check_stated_cp(gas.cp() * (1.0 + 5e-7)).is_ok());
    let Err(e) = gas.check_stated_cp(1006.0) else {
        panic!("c_p = 1006 is 0.13 percent off (122.5); it must be refused")
    };
    let m = format!("{e}");
    for what in ["physics.compressible.cp", "1006", "1004.69"] {
        assert!(m.contains(what), "'{what}' is not in the refusal: {m}");
    }
    // validate: air is fine, and each bad field is refused by name.
    assert!(gas.validate().is_ok());
    let mut g1 = gas;
    g1.gamma = 1.0;
    assert!(format!("{}", g1.validate().unwrap_err()).contains("gamma"));
    let mut g2 = gas;
    g2.w = 0.0;
    assert!(format!("{}", g2.validate().unwrap_err()).contains("w "));
    let mut g3 = gas;
    g3.p_op = -1.0;
    assert!(format!("{}", g3.validate().unwrap_err()).contains("p_op"));
    let mut g4 = gas;
    g4.viscosity = Viscosity::Constant { mu: -1.0 };
    assert!(format!("{}", g4.validate().unwrap_err()).contains("mu"));
    let g5 = PerfectGas { viscosity: Viscosity::Constant { mu: 0.0 }, ..gas };
    assert!(g5.validate().is_ok(), "mu = 0 is the inviscid gas");
}

/// §122.5's fourth gate, §122.2's point: a 1 Pa gauge perturbation on
/// 1e5 Pa survives in the working precision, on the device and on the
/// host twin alike - `rho(1) > rho(0)` and the perturbation back through
/// `psi` within 1 percent. Under `single` the error at 300 K is asserted
/// ABOVE 1e-3 (§122.2's oracle: 4.209e-3), so the row is known to measure
/// f32 arithmetic and not a vacuous pass.
#[test]
fn f32_gauge_pressure_resolves_1pa() {
    let Some(g) = gpu() else { return };
    let k = ThermoKernels::new(&g).expect("kernels");
    let gas = PerfectGas::air(1.0e5);
    let c = gas.constants();
    let p = [0.0 as Scalar, 1.0 as Scalar];
    let dp = g.upload(&p).expect("upload p");
    let mut dpsi = g.zeros::<Scalar>(2).expect("alloc psi");
    let mut drho = g.zeros::<Scalar>(2).expect("alloc rho");
    for tv in [300.0, 288.15, 273.15] {
        let t = [tv as Scalar, tv as Scalar];
        let dt = g.upload(&t).expect("upload t");
        gas_state(&g, &k, &c, &mut dpsi, &mut drho, &dp, &dt, 2).expect("gas_state");
        g.sync().expect("sync");
        let r = g.download(&drho).expect("download rho");
        let h0 = c.rho(0.0, tv as Scalar);
        let h1 = c.rho(1.0, tv as Scalar);
        assert!(r[1] > r[0], "device: rho(1) = {:e}, rho(0) = {:e}", r[1], r[0]);
        assert!(h1 > h0, "host: rho(1) = {h1:e}, rho(0) = {h0:e}");
        let psi1 = f64::from(c.psi(tv as Scalar));
        let err_dev = ((f64::from(r[1]) - f64::from(r[0])) / psi1 - 1.0).abs();
        let err_host = ((f64::from(h1) - f64::from(h0)) / psi1 - 1.0).abs();
        assert!(
            err_dev <= 0.01,
            "device at {tv} K: the 1 Pa perturbation came back as \
             {err_dev:.3e} of psi"
        );
        assert!(
            err_host <= 0.01,
            "host at {tv} K: the 1 Pa perturbation came back as \
             {err_host:.3e} of psi"
        );
        println!("  {tv} K: device {err_dev:.4e}  host {err_host:.4e}");
        #[cfg(feature = "single")]
        {
            if tv == 300.0 {
                assert!(
                    err_dev > 1e-3,
                    "the f32 control: the error at 300 K measured \
                     {err_dev:.3e}, and §122.2's oracle gives 4.209e-3 - \
                     above 1e-3"
                );
            }
        }
    }
}
