// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! The reacting dust parcel of Phase B - SPEC-LIT §159 (docs/17 DST-12).
//! A parcel carries moisture, volatile, char and ash masses, staged in that
//! order. It heats by (159.1) with the Ranz-Marshall Nusselt number (159.2)
//! and Bird's blowing correction (159.3), dries by §76's evaporation (the
//! Ranz-Marshall rung of (76.1) and the heat-limited boiling branch of
//! (76.9)), devolatilises by SFOM (159.4), and burns its char at the
//! kinetic-diffusion rate (159.5) of [`crate::dust::char`] with the heat
//! retention h_s (DST-13); ash is inert. Each step reports the mass and the
//! heat it exchanged with the gas, the accumulators of §159.7 and §159.8.
//!
//! Everything here is the host twin of `cuda/dustparticle.cu`: `f64` in
//! both builds, never [`crate::Scalar`], and the same operations in the same
//! order as the kernel, which is compiled with `-fmad=false`, so the gate
//! between them is four units in the last place. Positions and the walk are
//! not touched: they stay f64 as F32-12 made them (SPEC-LIT 118.9).
//!
//! Written from:
//!   SPEC-LIT §159.1 - (159.1) the particle energy, (159.2) Ranz-Marshall,
//!     (159.3) Bird's correction and its series below |beta| = 1e-3
//!   SPEC-LIT §159.2 - (159.4) SFOM devolatilisation
//!   SPEC-LIT §159.3 - (159.6) the closed forms the gates hold
//!   SPEC-LIT §159.4 - (159.7) the radiation temperature
//!   SPEC-LIT §159.4 - (159.7) the cloud's a_p and e_p per cell (DST-18)
//!   SPEC-LIT §159.7 - the stages, the segment integrator and the
//!     accumulators (derived here)
//!   SPEC-LIT §159.8 - the char stage, its accumulators and the stop on a
//!     particle with no mass left (DST-13)
//!   SPEC-LIT §76.1, §76.4, §76.5, §76.9 - the moisture closures: (76.1),
//!     the film state, Watson's h_v, Marrero-Mason's D(T), the surface
//!     mass fraction and the boiling branch; (S54.3) of §54 - Hyland-Wexler p_ws
//!   SPEC-LIT §77.1 - (77.5) the end-of-step temperature the vapour leaves at
//!   A. Islas, A. Rodriguez Fernandez, C. Betegon, E. Martinez-Paneda,
//!     A. Pandal, Computational assessment of biomass dust explosions in the
//!     20L sphere, arXiv:2207.13149 (2022) - eqs. (13)-(17), (20)-(23)
//! The specification was informed by a review of FSCDustFoam (Huang &
//! Lipatnikov, GPL) and no code was copied.
//! No GPL-licensed source was consulted.

use crate::dust::char::{self, CharInputs, CharProps};
use crate::error::{Error, Result};
use std::f64::consts::PI;

/// Stefan-Boltzmann constant, W/(m2 K4) (CODATA 2018, exact in SI).
pub const SIGMA_SB: f64 = 5.670_374_419e-8;
/// Molar gas constant, J/(mol K) (CODATA 2018, exact in SI).
pub const R0: f64 = 8.314_462_618_153_24;
/// |beta| below which (159.3)'s factor is its series (SPEC-LIT §159.1).
pub const BIRD_SERIES: f64 = 1.0e-3;
/// The volatile stage ends when m_v falls to this fraction of the parcel's
/// remaining mass; the remnant leaves in that step (SPEC-LIT §159.7). An
/// all-volatile parcel never meets the fraction - its remaining mass IS
/// m_v - so the stage also ends when the remnant drops below the smallest
/// normal double.
pub const STAGE_END: f64 = 1.0e-6;
/// Segments one step may take: drying sub-cooled, boiling, volatiles and
/// the inert rest need four; the rest is headroom (SPEC-LIT §159.7).
pub const MAX_SEGMENTS: usize = 8;
/// x below which phi1 and g2 of §159.7 are their series.
pub const PHI_SERIES: f64 = 1.0;
/// Dry air, kg/mol (§76.5's carrier).
pub const W_DRY_AIR: f64 = 28.966e-3;

fn refuse(who: &str, msg: String) -> Error {
    Error::Refused(format!("dust::particle::{who}: {msg}"))
}

/// The stage a parcel is in (SPEC-LIT §159.1): moisture leaves first, then
/// volatiles, then char; ash is inert.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stage {
    Moisture,
    Volatiles,
    Char,
}

/// (159.3)'s factor `beta/(e^beta - 1)`: the series
/// `1 - beta/2 + beta^2/12 - beta^4/720` below |beta| = 1e-3, where the
/// definition cancels, and `beta/expm1(beta)` above.
#[must_use]
pub fn bird_factor(beta: f64) -> f64 {
    if beta.abs() < BIRD_SERIES {
        let b2 = beta * beta;
        1.0 - 0.5 * beta + b2 / 12.0 - b2 * b2 / 720.0
    } else {
        beta / beta.exp_m1()
    }
}

/// phi1(x) = (1 - e^-x)/x and g2(x) = (x - 1 + e^-x)/x^2 of §159.7, the
/// series sum_j (-x)^j/(j+1)! and sum_j (-x)^j/(j+2)! below x = 1 (each
/// stopped when its term is below 1e-17 of its sum, at most 30 terms).
#[must_use]
pub fn phi_g(x: f64) -> (f64, f64) {
    if x < PHI_SERIES {
        let mut t1 = 1.0;
        let mut s1 = 1.0;
        let mut t2 = 0.5;
        let mut s2 = 0.5;
        for j in 1..30 {
            t1 = t1 * (-x) / (j as f64 + 1.0);
            t2 = t2 * (-x) / (j as f64 + 2.0);
            s1 += t1;
            s2 += t2;
            if t1.abs() < 1.0e-17 * s1.abs() && t2.abs() < 1.0e-17 * s2.abs() {
                break;
            }
        }
        (s1, s2)
    } else {
        let em = (-x).exp_m1();
        (-em / x, (x + em) / (x * x))
    }
}

/// Hyland-Wexler saturation pressure over water (T >= 273.15 K) and over
/// ice below, Pa - SPEC-LIT (S54.3), the coefficients `psychro::p_ws` and
/// `cuda/parcels.cu` carry, here in f64 in both builds.
#[must_use]
pub fn p_ws(t: f64) -> f64 {
    let l = t.ln();
    if t < 273.15 {
        (-5.674_535_9e3 / t + 6.392_524_7 + -9.677_843e-3 * t + 6.221_570_1e-7 * t * t
            + 2.074_782_5e-9 * t * t * t
            + -9.484_024e-13 * t * t * t * t
            + 4.163_501_9 * l)
            .exp()
    } else {
        (-5.800_220_6e3 / t + 1.391_499_3 + -4.864_023_9e-2 * t + 4.176_476_8e-5 * t * t
            + -1.445_209_3e-8 * t * t * t
            + 6.545_967_3 * l)
            .exp()
    }
}

/// The moisture's properties: water (§76.3's `LiquidProperties::water`,
/// in f64), the carrier and the ambient pressure, with the boiling point
/// at that pressure solved once.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Drying {
    pub w_vap: f64,
    pub t_boil: f64,
    pub p_boil: f64,
    pub h_v_boil: f64,
    pub t_crit: f64,
    pub d_vap: f64,
    pub t_ref_d: f64,
    pub d_exp: f64,
    pub cp_vap: f64,
    pub w_carrier: f64,
    pub p_amb: f64,
    /// T_b(p_amb), K: the root of p_ws(T) = p_amb.
    pub t_boil_p: f64,
}

impl Drying {
    /// Water in dry air at `p_amb`. Refused when p_amb is not inside the
    /// range p_ws spans between 150 K and the critical temperature.
    pub fn water(p_amb: f64) -> Result<Self> {
        let t_crit = 647.096;
        let (mut lo, mut hi) = (150.0_f64, t_crit - 1.0e-6);
        if !(p_amb > p_ws(lo) && p_amb < p_ws(hi)) {
            return Err(refuse(
                "Drying::water",
                format!("p_amb = {p_amb} Pa is outside the water saturation range"),
            ));
        }
        for _ in 0..200 {
            let mid = 0.5 * (lo + hi);
            if p_ws(mid) < p_amb {
                lo = mid;
            } else {
                hi = mid;
            }
            if hi - lo <= 1.0e-12 * hi {
                break;
            }
        }
        Ok(Self {
            w_vap: 18.015_268e-3,
            t_boil: 373.15,
            p_boil: 101_325.0,
            h_v_boil: 2.257e6,
            t_crit,
            d_vap: 2.505_362_314_4e-5,
            t_ref_d: 298.15,
            d_exp: 2.072,
            cp_vap: 1880.0,
            w_carrier: W_DRY_AIR,
            p_amb,
            t_boil_p: 0.5 * (lo + hi),
        })
    }
}

/// Watson's latent heat (§76.4), J/kg; zero at and above T_crit.
#[must_use]
pub fn latent_heat(dr: &Drying, t: f64) -> f64 {
    if t >= dr.t_crit {
        return 0.0;
    }
    dr.h_v_boil * ((dr.t_crit - t) / (dr.t_crit - dr.t_boil)).powf(0.38)
}

/// What a case states for its reacting parcels.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ParticleInputs {
    /// C_p of (159.1), J/(kg K), one value for the whole particle.
    pub cp: f64,
    /// epsilon_0 of (159.1), in [0, 1].
    pub emissivity: f64,
    /// SFOM (159.4): A, 1/s.
    pub devol_a: f64,
    /// SFOM (159.4): E_a, J/mol.
    pub devol_e_a: f64,
    /// Delta H of devolatilisation, J/kg; positive is endothermic.
    pub devol_dh: f64,
    /// C_p of the volatile vapour in (159.3), J/(kg K).
    pub cp_volatile: f64,
    pub drying: Drying,
    /// The gas's constant transport properties (as §76's `GasState`).
    pub mu: f64,
    pub k_g: f64,
    pub cp_g: f64,
    /// The char reaction (159.5); [`CharInputs::inert`] leaves the char
    /// inert, as DST-12 did.
    pub char_rate: CharInputs,
}

/// [`ParticleInputs`] checked, with the two derived constants the kernel
/// takes: theta = E_a/R0 and Pr^(1/3), and the char's [`CharProps`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ParticleProps {
    pub cp: f64,
    pub emissivity: f64,
    pub devol_a: f64,
    pub theta: f64,
    pub devol_dh: f64,
    pub cp_volatile: f64,
    pub drying: Drying,
    pub mu: f64,
    pub k_g: f64,
    pub cp_g: f64,
    pub cbrt_pr: f64,
    pub char_rate: CharProps,
}

impl ParticleProps {
    pub fn new(i: &ParticleInputs) -> Result<Self> {
        let pos = [
            ("cp", i.cp),
            ("cp_volatile", i.cp_volatile),
            ("mu", i.mu),
            ("k_g", i.k_g),
            ("cp_g", i.cp_g),
        ];
        for (name, v) in pos {
            if !(v.is_finite() && v > 0.0) {
                return Err(refuse("ParticleProps::new", format!("{name} = {v} must be finite and > 0")));
            }
        }
        for (name, v) in [("devol_a", i.devol_a), ("devol_e_a", i.devol_e_a)] {
            if !(v.is_finite() && v >= 0.0) {
                return Err(refuse("ParticleProps::new", format!("{name} = {v} must be finite and >= 0")));
            }
        }
        if !(i.emissivity >= 0.0 && i.emissivity <= 1.0) {
            return Err(refuse(
                "ParticleProps::new",
                format!("emissivity = {} must lie in [0, 1]", i.emissivity),
            ));
        }
        if !i.devol_dh.is_finite() {
            return Err(refuse("ParticleProps::new", format!("devol_dh = {} must be finite", i.devol_dh)));
        }
        Ok(Self {
            cp: i.cp,
            emissivity: i.emissivity,
            devol_a: i.devol_a,
            theta: i.devol_e_a / R0,
            devol_dh: i.devol_dh,
            cp_volatile: i.cp_volatile,
            drying: i.drying,
            mu: i.mu,
            k_g: i.k_g,
            cp_g: i.cp_g,
            cbrt_pr: (i.mu * i.cp_g / i.k_g).cbrt(),
            char_rate: CharProps::new(&i.char_rate)?,
        })
    }
}

/// One particle of a parcel: kg each, K, m.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ParticleState {
    pub m_w: f64,
    pub m_v: f64,
    pub m_c: f64,
    pub m_a: f64,
    pub t: f64,
    pub d: f64,
}

impl ParticleState {
    #[must_use]
    pub fn mass(&self) -> f64 {
        self.m_w + self.m_v + self.m_c + self.m_a
    }

    #[must_use]
    pub fn stage(&self) -> Stage {
        if self.m_w > 0.0 {
            Stage::Moisture
        } else if self.m_v > 0.0 {
            Stage::Volatiles
        } else {
            Stage::Char
        }
    }
}

/// The gas a parcel sees over a step, frozen: the cell's T, water vapour
/// mass fraction, density, incident radiation G and O2 partial pressure
/// p_O2 (Pa; no char burns at p_O2 <= 0), and |u_p - u|.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GasAtParcel {
    pub t: f64,
    pub y_w: f64,
    pub rho: f64,
    pub g: f64,
    pub u_rel: f64,
    pub p_o2: f64,
}

/// What one particle exchanged with the gas over a step (§159.7): the
/// water and volatile mass given, each times the end-of-step T_p (77.5),
/// the heat the gas gave by convection and the net radiation absorbed, J.
/// DST-13's char stage (§159.8) adds the carbon burnt, the O2 taken, the
/// CO2 given, the carbon times the end-of-step T_p, and the gas's share
/// (1 - h_s) of the reaction heat, J. The end-of-step T_p of the products
/// and the stored char mass are rounded to [`crate::Scalar`] as the device
/// stores them - the two places the host twin meets the build's precision.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Exchange {
    pub m_w: f64,
    pub m_v: f64,
    pub mt_w: f64,
    pub mt_v: f64,
    pub q_conv: f64,
    pub q_rad: f64,
    pub m_c: f64,
    pub m_o2: f64,
    pub m_co2: f64,
    pub mt_c: f64,
    pub q_char: f64,
}

/// One frozen-rate segment of (159.1): the end temperature and the
/// convective and radiative heat, by §159.7's phi1/g2 form.
#[allow(clippy::too_many_arguments)]
fn segment(a: f64, b: f64, rad0: f64, s: f64, t_g: f64, t0: f64, tseg: f64, mc: f64) -> (f64, f64, f64) {
    let num = a * (t_g - t0) + rad0 + s;
    let lam = a + b;
    let x = lam * tseg / mc;
    let (p1, g2) = phi_g(x);
    let tt = tseg / mc;
    let w = num * tseg * tt * g2;
    let t1 = t0 + num * tt * p1;
    let qc = a * ((t_g - t0) * tseg - w);
    let qr = rad0 * tseg - b * w;
    (t1, qc, qr)
}

/// One step of `dt` seconds for one particle in a frozen gas (§159.7). A
/// particle with no mass or no diameter is returned unchanged. Refused:
/// a `dt` that is not finite and > 0.
pub fn step(p: &ParticleProps, s: &ParticleState, gas: &GasAtParcel, dt: f64) -> Result<(ParticleState, Exchange)> {
    if !(dt.is_finite() && dt > 0.0) {
        return Err(refuse("step", format!("dt = {dt} must be finite and > 0")));
    }
    let mut st = *s;
    let mut ex = Exchange::default();
    if !(st.mass() > 0.0 && st.d > 0.0) {
        return Ok((st, ex));
    }
    let dr = &p.drying;
    let tb = dr.t_boil_p;
    let g4 = 0.25 * gas.g;
    let mut rem = dt;
    for _ in 0..MAX_SEGMENTS {
        // A particle with no mass left has nothing to heat (§159.8).
        if rem <= 0.0 || !(st.mass() > 0.0) {
            break;
        }
        // The diameter moves only in the char stage, so these are the
        // step's values until the char burns.
        let re = gas.rho * gas.u_rel * st.d / p.mu;
        let sqre = re.sqrt();
        let nu0 = 2.0 + 0.6 * sqre * p.cbrt_pr;
        let a0 = PI * st.d * p.k_g * nu0;
        let ar = PI * st.d * st.d * p.emissivity;
        let mc = st.mass() * p.cp;
        let t0 = st.t;
        let t03 = t0 * t0 * t0;
        let rad0 = ar * (g4 - SIGMA_SB * t03 * t0);
        let b = 4.0 * ar * SIGMA_SB * t03;
        if st.m_w > 0.0 {
            let hv = latent_heat(dr, t0.min(tb));
            let bnum = dr.cp_vap * (gas.t - tb) / hv;
            let fb = if bnum > 0.0 { bnum.ln_1p() / bnum } else { 1.0 };
            let ab = a0 * fb;
            let tb3 = tb * tb * tb;
            let rad_b = ar * (g4 - SIGMA_SB * tb3 * tb);
            let q_b = ab * (gas.t - tb) + rad_b;
            if t0 >= tb && q_b > 0.0 {
                // (76.9): heat-limited, T_p held at T_b.
                let mdot = q_b / hv;
                let t_out = st.m_w / mdot;
                let (tseg, dm, out) = if t_out <= rem {
                    (t_out, st.m_w, true)
                } else {
                    (rem, mdot * rem, false)
                };
                let out = out || dm >= st.m_w;
                let dm = if out { st.m_w } else { dm };
                st.m_w = if out { 0.0 } else { st.m_w - dm };
                ex.m_w += dm;
                ex.q_conv += ab * (gas.t - tb) * tseg;
                ex.q_rad += rad_b * tseg;
                st.t = tb;
                rem = if out { rem - tseg } else { 0.0 };
                continue;
            }
            // (76.1), the Ranz-Marshall rung, with the (76.4) film.
            let tp = t0.min(tb);
            let tf = tp + (gas.t - tp) / 3.0;
            let rhof = gas.rho * gas.t / tf;
            let df = dr.d_vap * (tf / dr.t_ref_d).powf(dr.d_exp) * (dr.p_boil / dr.p_amb);
            let sc = p.mu / (rhof * df);
            let sh = 2.0 + 0.6 * sqre * sc.cbrt();
            let x = (p_ws(tp) / dr.p_amb).clamp(0.0, 1.0);
            let ys = x * dr.w_vap / (dr.w_carrier + x * (dr.w_vap - dr.w_carrier));
            let mdot = (PI * st.d * rhof * df * sh * (ys - gas.y_w)).max(0.0);
            let a = a0 * bird_factor(dr.cp_vap * mdot / a0);
            let sv = -mdot * hv;
            let mut tseg = rem;
            let mut out = false;
            if mdot > 0.0 {
                let t_out = st.m_w / mdot;
                if t_out <= rem {
                    tseg = t_out;
                    out = true;
                }
            }
            let mut hit = false;
            let num = a * (gas.t - t0) + rad0 + sv;
            let lam = a + b;
            if num > 0.0 && t0 < tb {
                let r = lam * (tb - t0) / num;
                if r < 1.0 {
                    let t_hit = -(-r).ln_1p() * mc / lam;
                    if t_hit < tseg {
                        tseg = t_hit;
                        out = false;
                        hit = true;
                    }
                }
            }
            let (t1, qc, qr) = segment(a, b, rad0, sv, gas.t, t0, tseg, mc);
            let dm = if out { st.m_w } else { (mdot * tseg).min(st.m_w) };
            let out = out || dm >= st.m_w;
            st.m_w = if out { 0.0 } else { st.m_w - dm };
            ex.m_w += dm;
            ex.q_conv += qc;
            ex.q_rad += qr;
            st.t = if hit { tb } else { t1 };
            rem = if out || hit { rem - tseg } else { 0.0 };
        } else if st.m_v > 0.0 {
            // (159.4) at the frozen T_p: m_v decays by e^(-k t) exactly.
            let k = p.devol_a * (-p.theta / t0).exp();
            let e = -(-k * rem).exp_m1();
            let mut dm = st.m_v * e;
            let mut mv = st.m_v - dm;
            if mv <= STAGE_END * (st.mass() - dm) || mv < f64::MIN_POSITIVE {
                dm = st.m_v;
                mv = 0.0;
            }
            let mdot = dm / rem;
            let a = a0 * bird_factor(p.cp_volatile * mdot / a0);
            let sv = -mdot * p.devol_dh;
            let (t1, qc, qr) = segment(a, b, rad0, sv, gas.t, t0, rem, mc);
            st.m_v = mv;
            ex.m_v += dm;
            ex.q_conv += qc;
            ex.q_rad += qr;
            st.t = t1;
            rem = 0.0;
        } else if let Some(bn) = char::burn(&p.char_rate, st.m_c, st.m_a, st.d, t0, gas.t, gas.p_o2, rem) {
            // (159.5): the char burns; h_s of its heat stays in the particle.
            let ch = &p.char_rate;
            let q = bn.dm * ch.h_c;
            if bn.tseg > 0.0 {
                let sv = ch.h_s * q / bn.tseg;
                let (t1, qc, qr) = segment(a0, b, rad0, sv, gas.t, t0, bn.tseg, mc);
                ex.q_conv += qc;
                ex.q_rad += qr;
                st.t = t1;
            } else {
                st.t = t0 + ch.h_s * q / mc;
            }
            ex.m_c += bn.dm;
            ex.m_o2 += bn.dm * ch.o2_per_c;
            ex.m_co2 += bn.dm * ch.co2_per_c;
            ex.q_char += ch.gas_share * q;
            st.m_c = bn.m_c;
            st.d = bn.d;
            rem = if bn.out { rem - bn.tseg } else { 0.0 };
        } else {
            // Ash, char with nothing to burn it, or a particle that is out.
            let (t1, qc, qr) = segment(a0, b, rad0, 0.0, gas.t, t0, rem, mc);
            ex.q_conv += qc;
            ex.q_rad += qr;
            st.t = t1;
            rem = 0.0;
        }
    }
    // (77.5): the end-of-step temperature as the state stores it, so the
    // ledger is a function of the endpoint a snapshot sees (SPEC-LIT 77.1).
    #[allow(clippy::unnecessary_cast)]
    let t_end = st.t as crate::Scalar as f64;
    ex.mt_w = ex.m_w * t_end;
    ex.mt_v = ex.m_v * t_end;
    ex.mt_c = ex.m_c * t_end;
    Ok((st, ex))
}

/// (159.7) of SPEC-LIT 159.4 per cell, from a cloud's parcels: `a_p` (1/m)
/// and `e_p` (W/m3), the absorption the cloud adds to P1's and the grey
/// emission the parcels' own temperatures send into `G`'s equation - what
/// [`crate::radiation::p1::P1::set_particles`] hands the solve. A cell
/// with no parcel carries 0 in both.
#[derive(Debug, Clone, PartialEq)]
pub struct CloudRadiation {
    pub a_p: Vec<f64>,
    pub e_p: Vec<f64>,
}

/// (159.7) of SPEC-LIT 159.4 per cell from one cloud's parcels, in f64 in
/// both builds: `a_p[c] = sum_parcels n_p eps_0 pi d_p^2 / 4 / V_c` (1/m),
/// the absorption P1 adds to the gas's, and `e_p[c] = sum n_p pi d_p^2
/// eps_0 sigma_SB T_p^4 / V_c` (W/m3), the emission of the parcels' own
/// temperatures. `volume.len()` is the cell count; THE CALLER passes only
/// the parcels that radiate (mass > 0, d > 0), one entry per parcel, in
/// slot order. `ar` is the SAME expression and order `step` uses.
pub fn cloud_radiation(
    emissivity: f64,
    n_p: &[f64],
    d: &[f64],
    t_p: &[f64],
    cell: &[usize],
    volume: &[f64],
) -> Result<CloudRadiation> {
    if !(n_p.len() == d.len() && d.len() == t_p.len() && t_p.len() == cell.len()) {
        return Err(refuse(
            "cloud_radiation",
            format!(
                "the lengths of n_p ({}), d ({}), t_p ({}) and cell ({}) are \
                 not all equal",
                n_p.len(),
                d.len(),
                t_p.len(),
                cell.len()
            ),
        ));
    }
    for i in 0..n_p.len() {
        if cell[i] >= volume.len() {
            return Err(refuse(
                "cloud_radiation",
                format!(
                    "cell[{i}] = {} is not a cell of a {}-cell volume",
                    cell[i],
                    volume.len()
                ),
            ));
        }
    }
    for (c, &v) in volume.iter().enumerate() {
        if !(v.is_finite() && v > 0.0) {
            return Err(refuse(
                "cloud_radiation",
                format!("volume[{c}] = {v} must be finite and > 0"),
            ));
        }
    }
    for i in 0..n_p.len() {
        for (name, v) in [("n_p", n_p[i]), ("d", d[i]), ("t_p", t_p[i])] {
            if !(v.is_finite() && v >= 0.0) {
                return Err(refuse(
                    "cloud_radiation",
                    format!("{name}[{i}] = {v} must be finite and >= 0"),
                ));
            }
        }
    }
    if !(emissivity >= 0.0 && emissivity <= 1.0) {
        return Err(refuse(
            "cloud_radiation",
            format!("emissivity = {emissivity} must lie in [0, 1]"),
        ));
    }
    let mut sa = vec![0.0; volume.len()];
    let mut se = vec![0.0; volume.len()];
    for i in 0..n_p.len() {
        let ar_i = PI * d[i] * d[i] * emissivity;
        let t3 = t_p[i] * t_p[i] * t_p[i];
        let c = cell[i];
        sa[c] += n_p[i] * ar_i;
        se[c] += n_p[i] * ar_i * SIGMA_SB * t3 * t_p[i];
    }
    let a_p = sa.iter().zip(volume).map(|(&s, &v)| 0.25 * s / v).collect();
    let e_p = se.iter().zip(volume).map(|(&s, &v)| s / v).collect();
    Ok(CloudRadiation { a_p, e_p })
}

#[cfg(test)]
#[path = "tests_particle.rs"]
mod tests;
