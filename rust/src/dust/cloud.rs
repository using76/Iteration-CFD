// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! The gas side of a Phase B dust cloud, host `f64` in both builds -
//! SPEC-LIT §159.10 (docs/17 DST-15): (159.16) the volatile pseudo-species
//! `VOL`, (159.18) the particle's enthalpy from its stage datums, (159.19)
//! the gather of what a step's parcels gave the gas, (159.20) the cell
//! sources that carry it, and (159.21) the cloud's own AICC end state, the
//! bar the closed box's driver is held against. Every mass and energy term
//! is a difference of stored particle words, so the gas's gains over a run
//! telescope to the cloud's losses, as (77.4)-(77.5) do for a vapour.
//!
//! Written from:
//!   SPEC-LIT §152.1-§152.2 - the corn-starch formula `C6H7.88O4.98` and
//!     the derived a5 from its heat of formation
//!   SPEC-LIT §152.4 - the NASA-7 form of the starch's pseudo-species
//!     record (b1 = a0, b2 = a1, b3 = b4 = b5 = 0, b6 = a5, b7 = 0)
//!   SPEC-LIT §152.6 - the complete oxidation the volatile reaction is the
//!     50-fold image of
//!   SPEC-LIT §159.7 - the Watson latent heat and the devolatilisation
//!     heat dH the volatile's datum carries
//!   SPEC-LIT §159.8 - the element ratios the char's CO2 and O2 terms
//!     mirror
//!   SPEC-LIT §159.9 - the cloud a parcel pool makes
//!   SPEC-LIT §159.10 - (159.16) the VOL record, (159.18) the particle
//!     enthalpy, (159.19) the gather, (159.20) the sources and (159.21)
//!     the cloud's AICC, as built
//!   C. Huang, M. Bloching & A. N. Lipatnikov, J. Loss Prev. Process Ind.
//!     75 (2022) 104707, DOI 10.1016/j.jlp.2021.104707 - eqs. (1)-(2) and
//!     Table 1 (CC BY)
//! The specification was informed by a review of FSCDustFoam (Huang &
//! Lipatnikov, GPL) and no code was copied.
//! No GPL-licensed source was consulted.

use crate::chem::thermo::SpeciesThermo;
use crate::dust::particle::{latent_heat, Drying, ParticleState};
use crate::error::{Error, Result};

/// K, T° of (159.18): the datum of the particle's sensible enthalpy and of
/// every stage datum.
pub const T_DATUM: f64 = crate::dust::thermo::T_STD;

/// The cloud's four gas members, in the order of every `[_; 4]` below.
pub const MEMBERS: [&str; 4] = ["VOL", "H2O", "CO2", "O2"];

/// gcd of two non-negative counts.
fn gcd(a: i64, b: i64) -> i64 {
    let (mut a, mut b) = (a.unsigned_abs(), b.unsigned_abs());
    while b != 0 {
        let t = a % b;
        a = b;
        b = t;
    }
    a as i64
}

/// (159.16): the volatile pseudo-species `VOL` - the dust's own fuel
/// formula of §152.1 as one gas species, scaled so its CHEMKIN record
/// carries whole element counts. The record is the starch's per-mole
/// NASA-7 polynomial times the same scale, so per kilogram it is the
/// starch exactly. Refused (Error::Config naming `dust.cloud`): a scaled
/// atom count that is not whole or lands above 999 (a record's count field
/// holds three digits).
pub fn volatile_species(
    fuel: &crate::dust::thermo::StarchFuel,
    elements: &crate::chem::thermo::ElementTable,
) -> Result<SpeciesThermo> {
    const W: &str = "dust.cloud: the volatile species VOL (SPEC-LIT §159.10)";
    let g = gcd(gcd(fuel.c_centi, fuel.h_centi), fuel.o_centi);
    let scale = 100 / gcd(100, g);
    let count = |centi: i64| -> Result<i64> {
        let n = centi * scale;
        if n % 100 != 0 {
            return Err(Error::Config(format!(
                "{W}: the formula count {centi}/100 times the scale {scale} is not a \
                 whole atom count"
            )));
        }
        let n = n / 100;
        if n > 999 {
            return Err(Error::Config(format!(
                "{W}: the formula count {centi}/100 scales to {n} atoms, above a \
                 record's three-digit count field"
            )));
        }
        Ok(n)
    };
    let nc = count(fuel.c_centi)?;
    let nh = count(fuel.h_centi)?;
    let no = count(fuel.o_centi)?;
    let mut els: Vec<(String, u32)> = Vec::new();
    for (sym, n) in [("C", nc), ("H", nh), ("O", no)] {
        if n != 0 {
            els.push((sym.to_string(), n as u32));
        }
    }
    let weight = |sym: &str| -> Result<f64> {
        elements.weight(sym).ok_or_else(|| {
            Error::Config(format!("{W}: the element table has no weight for {sym}"))
        })
    };
    let molar_mass = (nc as f64 * weight("C")? + nh as f64 * weight("H")?
        + no as f64 * weight("O")?)
        / 1000.0;
    let s = scale as f64;
    let coeffs = [s * fuel.a0, s * fuel.a1, 0.0, 0.0, 0.0, s * fuel.a5(), 0.0];
    Ok(SpeciesThermo {
        name: "VOL".to_string(),
        elements: els,
        molar_mass,
        t_low: 200.0,
        t_mid: 1000.0,
        t_high: 6000.0,
        upper: coeffs,
        lower: coeffs,
        line: 0,
    })
}

/// The cloud's four records, in [`MEMBERS`] order, with the element
/// table's carbon (159.19's `W_C`).
pub struct CloudThermo {
    pub vol: SpeciesThermo,
    pub h2o: SpeciesThermo,
    pub co2: SpeciesThermo,
    pub o2: SpeciesThermo,
    pub w_c: f64,
}

impl CloudThermo {
    /// Clones the four records out of a mechanism's thermo, by name.
    /// Refused (Error::Config naming `dust.cloud`): any of them missing.
    pub fn from_set(
        set: &crate::chem::thermo::ThermoSet,
        elements: &crate::chem::thermo::ElementTable,
    ) -> Result<Self> {
        let take = |name: &str| -> Result<SpeciesThermo> {
            set.get(name).cloned().ok_or_else(|| {
                Error::Config(format!(
                    "dust.cloud: the mechanism has no species \"{name}\"; the cloud's \
                     four members VOL, H2O, CO2 and O2 must all be solved (SPEC-LIT \
                     §159.10)"
                ))
            })
        };
        let h2o = take("H2O")?;
        let co2 = take("CO2")?;
        let o2 = take("O2")?;
        let vol = take("VOL")?;
        let Some(w_c) = elements.weight("C") else {
            return Err(Error::Config(
                "dust.cloud: the element table has no weight for C (SPEC-LIT §159.10)"
                    .to_string(),
            ));
        };
        Ok(Self { vol, h2o, co2, o2, w_c: w_c / 1000.0 })
    }

    /// The record of member k, in [`MEMBERS`] order.
    pub fn record(&self, k: usize) -> &SpeciesThermo {
        match k {
            0 => &self.vol,
            1 => &self.h2o,
            2 => &self.co2,
            _ => &self.o2,
        }
    }

    /// J/kg: the record's h(T) over its molar mass.
    pub fn h_mass(&self, k: usize, t: f64) -> Result<f64> {
        let r = self.record(k);
        Ok(r.h(t)? / r.molar_mass)
    }

    /// kg/mol, (159.19)'s `W_CO2 = W_C + W_O2`.
    pub fn w_co2(&self) -> f64 {
        self.w_c + self.o2.molar_mass
    }
}

/// The four stage datums of (159.18), J/kg: what a kilogram moving from
/// its stage to the gas carries besides its sensible enthalpy.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StageDatums {
    pub h_w: f64,
    pub h_v: f64,
    pub h_c: f64,
    pub h_a: f64,
}

impl StageDatums {
    /// (159.18): `h_w = h_H2O(T°)/W_H2O - h_v(T°)` with h_v the water
    /// `Drying`'s Watson latent heat, `h_v = h_VOL(T°)/W_VOL - dH_devol`,
    /// and `h_c = h_a = 0` (graphite's reference state; ash is inert).
    pub fn new(
        th: &CloudThermo,
        drying: &Drying,
        devol_dh: f64,
    ) -> Result<Self> {
        Ok(Self {
            h_w: th.h_mass(1, T_DATUM)? - latent_heat(drying, T_DATUM),
            h_v: th.h_mass(0, T_DATUM)? - devol_dh,
            h_c: 0.0,
            h_a: 0.0,
        })
    }
}

/// (159.18): a particle's enthalpy from its stored state alone - the four
/// stages' sensible enthalpy on one constant `cp`, plus each stage's mass
/// times its datum. Evaluated left to right.
pub fn particle_enthalpy(s: &ParticleState, cp: f64, dat: &StageDatums) -> f64 {
    (s.m_w + s.m_v + s.m_c + s.m_a) * cp * (s.t - T_DATUM)
        + s.m_w * dat.h_w
        + s.m_v * dat.h_v
        + s.m_c * dat.h_c
        + s.m_a * dat.h_a
}

/// One cell's gains over a step (159.19): kg per member (MEMBERS order)
/// and J.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct CellGain {
    pub m: [f64; 4],
    pub e: f64,
}

/// (159.19): every parcel's step, from its state before (`before`) and
/// after (`after`) it - both the stored words, widened to f64 - gives its
/// cell mass and energy, in slot order. The volatile leaves as VOL, the
/// moisture as H2O, and the char as CO2 taking W_O2/W_C of O2, so carbon
/// and oxygen balance by construction as (159.11) does for the particle.
/// Refused (Error::Config naming `dust.cloud`): unequal slice lengths, or
/// a cell index at or above `n_cells`.
pub fn gather(
    before: &[ParticleState],
    after: &[ParticleState],
    n_p: &[f64],
    cell: &[usize],
    n_cells: usize,
    cp: f64,
    dat: &StageDatums,
    th: &CloudThermo,
) -> Result<Vec<CellGain>> {
    const W: &str = "dust.cloud: the gather (SPEC-LIT §159.10)";
    if after.len() != before.len() || n_p.len() != before.len() || cell.len() != before.len() {
        return Err(Error::Config(format!(
            "{W}: the four parcel slices hold {}, {}, {} and {} entries; they must be \
             equal",
            before.len(),
            after.len(),
            n_p.len(),
            cell.len()
        )));
    }
    let r_co2 = th.w_co2() / th.w_c;
    let r_o2 = th.o2.molar_mass / th.w_c;
    let mut out = vec![CellGain::default(); n_cells];
    for i in 0..before.len() {
        let c = cell[i];
        if c >= n_cells {
            return Err(Error::Config(format!(
                "{W}: parcel {i} names cell {c}, which is not below the {n_cells} cells"
            )));
        }
        let (b, a) = (&before[i], &after[i]);
        let n = n_p[i];
        let dc = b.m_c - a.m_c;
        let g = &mut out[c];
        g.m[0] += n * (b.m_v - a.m_v);
        g.m[1] += n * (b.m_w - a.m_w);
        g.m[2] += n * dc * r_co2;
        g.m[3] -= n * dc * r_o2;
        g.e += n * (particle_enthalpy(b, cp, dat) - particle_enthalpy(a, cp, dat));
    }
    Ok(out)
}

/// (159.20)'s cell sources: kg/(m3 s) per member (MEMBERS order) and the
/// temperature equation's q in W/m3.
#[derive(Clone, Debug)]
pub struct CloudSources {
    pub mdot: [Vec<f64>; 4],
    pub q: Vec<f64>,
}

/// (159.20): the gather's gains spread over the cell's volume and the
/// step. `q` is what the cell's `rho cp dT` gains after the new species'
/// enthalpies at the gas temperature `t_gas` (the step's opening) are
/// paid back out of `dE`. Refused (Error::Config naming `dust.cloud`):
/// unequal lengths, a `dt` that is not finite and above zero, or a volume
/// that is not finite and above zero.
pub fn sources(
    gain: &[CellGain],
    volume: &[f64],
    t_gas: &[f64],
    dt: f64,
    th: &CloudThermo,
) -> Result<CloudSources> {
    const W: &str = "dust.cloud: the sources (SPEC-LIT §159.10)";
    if volume.len() != gain.len() || t_gas.len() != gain.len() {
        return Err(Error::Config(format!(
            "{W}: the gains, the volumes and the gas temperatures hold {}, {} and {} \
             cells; they must be equal",
            gain.len(),
            volume.len(),
            t_gas.len()
        )));
    }
    if !(dt.is_finite() && dt > 0.0) {
        return Err(Error::Config(format!(
            "{W}: dt = {dt} is not a finite step above zero"
        )));
    }
    for (c, &v) in volume.iter().enumerate() {
        if !(v.is_finite() && v > 0.0) {
            return Err(Error::Config(format!(
                "{W}: cell {c}'s volume {v} is not a finite volume above zero"
            )));
        }
    }
    let mut mdot = [
        vec![0.0; gain.len()],
        vec![0.0; gain.len()],
        vec![0.0; gain.len()],
        vec![0.0; gain.len()],
    ];
    let mut q = vec![0.0; gain.len()];
    for (c, g) in gain.iter().enumerate() {
        let vdt = volume[c] * dt;
        let mut mh = 0.0;
        for k in 0..4 {
            mdot[k][c] = g.m[k] / vdt;
            mh += g.m[k] * th.h_mass(k, t_gas[c])?;
        }
        q[c] = (g.e - mh) / vdt;
    }
    Ok(CloudSources { mdot, q })
}

/// The cloud's charge (159.21): its concentration C in kg/m3, the four
/// mass fractions (moisture, volatile, char, ash), the particle's `cp`
/// and the stage datums of (159.18).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CloudCharge {
    pub concentration: f64,
    pub fractions: [f64; 4],
    pub cp: f64,
    pub datums: StageDatums,
}

/// (159.21)'s end state: the opening state, the AICC end and the end
/// partial densities in `gas`'s order.
#[derive(Clone, Debug)]
pub struct CloudAicc {
    pub t0: f64,
    pub p0: f64,
    pub u0: f64,
    pub t_v: f64,
    pub p_v: f64,
    /// kg/m3, one entry per `gas` species in its order; VOL's is 0.
    pub rho_end: Vec<f64>,
}

/// (159.21): the cloud's own AICC. `gas` is every gas species' record
/// with its initial partial density (kg/m3); VOL, H2O, CO2 and O2 must be
/// among them by name. Complete conversion - the moisture becomes H2O,
/// `n_v` moles of VOL become CO2 and H2O, the char becomes CO2, the ash
/// stays - ends the box at the T where `U_e(T) = u0`, solved by
/// bisection on the records' 200-6000 K range (200 halvings, or until the
/// midpoint equals an end). Refused: a member missing from `gas`
/// (Error::Config), a negative end O2 (Error::Refused - a rich cloud has
/// no complete conversion), or a `u0` outside the bracket.
pub fn cloud_aicc(
    gas: &[(SpeciesThermo, f64)],
    t0: f64,
    charge: &CloudCharge,
    th: &CloudThermo,
) -> Result<CloudAicc> {
    const W: &str = "dust.cloud: the cloud's AICC (SPEC-LIT §159.10)";
    let mut ix = [None; 4];
    for (k, name) in MEMBERS.iter().enumerate() {
        match gas.iter().position(|(r, _)| r.name == *name) {
            Some(p) => ix[k] = Some(p),
            None => {
                return Err(Error::Config(format!(
                    "{W}: the gas holds no {name}; the cloud's four members must all \
                     be among its species (SPEC-LIT §159.10)"
                )));
            }
        }
    }
    let rho0: Vec<f64> = gas.iter().map(|(_, r)| *r).collect();
    let sum_w = |rho: &[f64]| rho.iter().zip(gas).fold(0.0, |a, (r, (rec, _))| a + r / rec.molar_mass);
    let sum_hw = |rho: &[f64], t: f64| -> Result<f64> {
        let mut s = 0.0;
        for ((rec, _), r) in gas.iter().zip(rho) {
            s += r * rec.h(t)? / rec.molar_mass;
        }
        Ok(s)
    };
    let p0 = crate::chem::thermo::R_GAS * t0 * sum_w(&rho0);
    let f = &charge.fractions;
    let d = &charge.datums;
    let c = charge.concentration;
    let cloud_u = c
        * (charge.cp * (t0 - T_DATUM) + f[0] * d.h_w + f[1] * d.h_v + f[2] * d.h_c
            + f[3] * d.h_a);
    let u0 = sum_hw(&rho0, t0)? - p0 + cloud_u;
    // The end densities of (159.21), with VOL's counts n_C, n_H, n_O.
    let count = |sym: &str| {
        th.vol.elements.iter().find(|(s, _)| s == sym).map_or(0.0, |&(_, n)| f64::from(n))
    };
    let (nc, nh, no) = (count("C"), count("H"), count("O"));
    let n_v = c * f[1] / th.vol.molar_mass;
    let mut rho_e = rho0.clone();
    rho_e[ix[0].unwrap()] = 0.0;
    rho_e[ix[1].unwrap()] = rho0[ix[1].unwrap()] + c * f[0] + n_v * (nh / 2.0) * th.h2o.molar_mass;
    rho_e[ix[2].unwrap()] = rho0[ix[2].unwrap()]
        + n_v * nc * th.w_co2()
        + c * f[2] * th.w_co2() / th.w_c;
    let o2_end = rho0[ix[3].unwrap()]
        - n_v * (nc + nh / 4.0 - no / 2.0) * th.o2.molar_mass
        - c * f[2] * th.o2.molar_mass / th.w_c;
    if o2_end < 0.0 {
        return Err(Error::Refused(format!(
            "{W}: the cloud is richer than complete conversion allows - the end O2 \
             would be {o2_end} kg/m3 (SPEC-LIT §159.10)"
        )));
    }
    rho_e[ix[3].unwrap()] = o2_end;
    let u_e = |t: f64| -> Result<f64> {
        Ok(sum_hw(&rho_e, t)? - crate::chem::thermo::R_GAS * t * sum_w(&rho_e)
            + c * f[3] * charge.cp * (t - T_DATUM))
    };
    let (u_lo, u_hi) = (u_e(200.0)?, u_e(6000.0)?);
    if !(u0 >= u_lo && u0 <= u_hi) {
        return Err(Error::Refused(format!(
            "{W}: u0 = {u0} J/m3 lies outside [U_e(200 K), U_e(6000 K)] = [{u_lo}, \
             {u_hi}] J/m3; the end state is off the records' range"
        )));
    }
    let (mut lo, mut hi) = (200.0f64, 6000.0f64);
    let mut t_v = 0.5 * (lo + hi);
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        t_v = mid;
        if mid == lo || mid == hi {
            break;
        }
        if u_e(mid)? < u0 {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let p_v = crate::chem::thermo::R_GAS * t_v * sum_w(&rho_e);
    Ok(CloudAicc { t0, p0, u0, t_v, p_v, rho_end: rho_e })
}

#[cfg(test)]
#[path = "tests_cloud.rs"]
mod tests;
