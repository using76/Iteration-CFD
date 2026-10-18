// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! The weighted sum of grey gases (WSGG, SPEC-LIT §136.4 and §136.9): the
//! grey gases' weights `a_i` and absorption coefficients `kappa_i` from `T`,
//! `X_H2O` and `X_CO2`, the grey mean `wsggGrey` uses, and the non-grey
//! loop that solves (136.10) once per grey gas and sums the gases.
//!
//! * [`WsggCoefficients`] reads a user-supplied Bordbar et al. 2021
//!   coefficient file (the files are non-commercial and not to be
//!   distributed, so none is vendored, §136.4), checks its sha256 against
//!   [`WSGG_DIGESTS`], and evaluates (136.8)-(136.9) in f64, refusing
//!   `M_r` and `T` outside the fitted range by name.
//! * [`GreyGasSet`] is any `(a_i, kappa_i)` set: (136.7)'s emissivity and
//!   the grey mean `kappa = -ln(1 - eps(L_m))/L_m`, `L_m = 3.6 V/A`
//!   ([`mean_beam_length`]).
//! * [`WsggP1`] and [`WsggDom`] are the non-grey loop: one grey P1 or fvDOM
//!   solve per gas in one storage ([`P1::set_grey_gas`],
//!   [`DomSolve::set_grey_gas`]: `kappa -> kappa_i`, every emission weighted
//!   by `a_i`), with `G`, `q` and the wall flux summed over the gases in
//!   `ofacc` (§118.1) and rounded once. The transparent window (`kappa_0 =
//!   0`) carries no medium source and is not solved (§136.4).
//!
//! Provenance: ORIGINAL - the reader, the refusals and the loop are this
//! project's own. The model is Bordbar, Coelho, Fraga, França & Hostikka,
//! Int. J. Heat Mass Transfer 173 (2021) 121207 (CC BY), eqs. (7)-(11) and
//! (13), via SPEC-LIT §136.4; the coefficient files' row layout was read
//! from the article's supplementary material, whose data are never copied
//! into this tree. SHA-256 is FIPS 180-4 (NIST, 2015, US Government work).
//! No GPL-licensed source was consulted.

use std::path::Path;

use crate::device::{cfg_for, DevBuf, Gpu, KernelSet};
use crate::energy::EnergySources;
use crate::error::{Error, Result};
use crate::field::GpuScalarField;
use crate::mesh::{GpuMesh, HostMesh, PatchKind};
use crate::solver::{SolverControls, SolverPerformance};
use crate::{Acc, Label, Scalar, Vec3};
use cudarc::driver::{CudaFunction, PushKernelArg};

use super::dom::{DomControls, DomMedium, DomPatch, DomReport, DomSolve};
use super::dom_quad::Quadrature;
use super::p1::{MarshakPatch, P1Medium, P1};

// ==========================================================================
//  The model's constants (SPEC-LIT §136.4)
// ==========================================================================

/// Bordbar 2021's grey gases, the window `i = 0` not counted.
pub const WSGG_N_GASES: usize = 4;

/// The temperature range the 2021 model is fitted for, K (its §2.1).
pub const WSGG_T_RANGE_K: (f64, f64) = (300.0, 3000.0);

/// The `M_r = X_H2O/X_CO2` range the 2021 model is fitted for (its §2.1).
pub const WSGG_MR_RANGE: (f64, f64) = (0.1, 4.0);

/// (136.8)'s reference temperature, `T̂ = T / 1500 K`.
pub const WSGG_T_REF_K: f64 = 1500.0;

/// The ten supplementary files of Bordbar 2021: the total pressure in atm,
/// the file's name and its sha256, as SPEC-LIT §136.4 records them.
pub const WSGG_DIGESTS: [(f64, &str, &str); 10] = [
    (0.1, "Final_PWSGG_Model_Coef_Pt001.dat", "695af1dbebd964be71f8479c67093aaaa231c5e65211045ae701eb7dc9160539"),
    (0.5, "Final_PWSGG_Model_Coef_Pt002.dat", "e99bd2c0538e4e9b4628332564dc500ae0bae9901279892af3f6b3cf99d716eb"),
    (1.0, "Final_PWSGG_Model_Coef_Pt003.dat", "6848956545a301212dbb7d4e2e3744bfb6e9c8f418b8a8bcd5ce95832bca4bed"),
    (2.0, "Final_PWSGG_Model_Coef_Pt004.dat", "ddfde896b882e98d4605cff8b7b6125a420211195e2db2e26b9ba043d03df240"),
    (5.0, "Final_PWSGG_Model_Coef_Pt005.dat", "14424163b0443c30f7fd53c547baa558678c3db2e4ae1b57f4a3e1afccca6d4f"),
    (10.0, "Final_PWSGG_Model_Coef_Pt006.dat", "68a632e033339849dd490a781f7060a2bb4612ac249c481480c97f7c4135cd50"),
    (20.0, "Final_PWSGG_Model_Coef_Pt007.dat", "118eaa366e380376e6278ddc60b68ec0f795e2b3846d3e0bfce89965aa6f9cf9"),
    (30.0, "Final_PWSGG_Model_Coef_Pt008.dat", "b43e171035d0efeba0abd7479bdbac1209c0a72e0ab951651f42e5010e5300d5"),
    (50.0, "Final_PWSGG_Model_Coef_Pt009.dat", "72f6e541144e9560b635c81f4677a659cdad4c5f5b99e5ebd4493f1bf8b75c86"),
    (80.0, "Final_PWSGG_Model_Coef_Pt010.dat", "3927e11da0611b2119637bec83de7dd7a61161762a79bdcca3d4ac84438e1cfa"),
];

// ==========================================================================
//  SHA-256 (FIPS 180-4) - the digest check of §136.4
// ==========================================================================

/// SHA-256 of `msg` (FIPS 180-4, NIST 2015, section 6.2) as 64 lower-case hex
/// digits - what `sha256sum` prints.
pub fn sha256_hex(msg: &[u8]) -> String {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];
    let mut h: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];
    // FIPS 180-4 section 5.1.1: 0x80, zeros to 56 mod 64, then the bit length, big-endian.
    let mut m = msg.to_vec();
    let bits = (msg.len() as u64).wrapping_mul(8);
    m.push(0x80);
    while m.len() % 64 != 56 {
        m.push(0);
    }
    m.extend_from_slice(&bits.to_be_bytes());
    for block in m.chunks_exact(64) {
        let mut w = [0u32; 64];
        for t in 0..16 {
            w[t] = u32::from_be_bytes([block[4 * t], block[4 * t + 1], block[4 * t + 2], block[4 * t + 3]]);
        }
        for t in 16..64 {
            let s0 = w[t - 15].rotate_right(7) ^ w[t - 15].rotate_right(18) ^ (w[t - 15] >> 3);
            let s1 = w[t - 2].rotate_right(17) ^ w[t - 2].rotate_right(19) ^ (w[t - 2] >> 10);
            w[t] = w[t - 16].wrapping_add(s0).wrapping_add(w[t - 7]).wrapping_add(s1);
        }
        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh] = h;
        for t in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ (!e & g);
            let t1 = hh.wrapping_add(s1).wrapping_add(ch).wrapping_add(K[t]).wrapping_add(w[t]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(maj);
            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(t1);
            d = c;
            c = b;
            b = a;
            a = t1.wrapping_add(t2);
        }
        for (hi, v) in h.iter_mut().zip([a, b, c, d, e, f, g, hh]) {
            *hi = hi.wrapping_add(v);
        }
    }
    h.iter().map(|v| format!("{v:08x}")).collect()
}

// ==========================================================================
//  Grey gas sets: (136.7) and the grey mean
// ==========================================================================

/// One grey gas: its weight `a_i` (dimensionless) and its absorption
/// coefficient `kappa_i = kappa_p,i (p_H2O + p_CO2)`, 1/m.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GreyGas {
    pub a: f64,
    pub kappa: f64,
}

/// A set of grey gases, the transparent window not counted: its weight is
/// `a_0 = 1 - sum a_i` ([`GreyGasSet::window`]) and its `kappa_0 = 0`.
#[derive(Debug, Clone, PartialEq)]
pub struct GreyGasSet {
    gases: Vec<GreyGas>,
}

impl GreyGasSet {
    /// A set from its gases. At least one gas; every `a_i` finite and every
    /// `kappa_i` finite and positive - a zero `kappa` is the window, which
    /// is not a member. A negative `a_i` is accepted: Bordbar 2021's fits
    /// give some inside their own range, and the model is used as
    /// published (SPEC-LIT §136.9).
    pub fn new(gases: Vec<GreyGas>) -> Result<Self> {
        if gases.is_empty() {
            return Err(Error::Config(
                "radiation::wsgg::GreyGasSet: no grey gas - a WSGG set needs at \
                 least one (SPEC-LIT §136.4)"
                    .into(),
            ));
        }
        for (i, g) in gases.iter().enumerate() {
            if !g.a.is_finite() {
                return Err(Error::Config(format!(
                    "radiation::wsgg::GreyGasSet: gas {} has weight a = {} - the \
                     weights must be finite (SPEC-LIT §136.4)",
                    i + 1,
                    g.a
                )));
            }
            if !g.kappa.is_finite() || g.kappa <= 0.0 {
                return Err(Error::Config(format!(
                    "radiation::wsgg::GreyGasSet: gas {} has kappa = {} 1/m - a \
                     grey gas's kappa must be positive and finite; the window \
                     (kappa_0 = 0) is not a member of the set (SPEC-LIT §136.4)",
                    i + 1,
                    g.kappa
                )));
            }
        }
        Ok(Self { gases })
    }

    /// The grey gases, `i = 1..N_g`.
    pub fn gases(&self) -> &[GreyGas] {
        &self.gases
    }

    /// The window's weight `a_0 = 1 - sum_i a_i` of (136.8).
    pub fn window(&self) -> f64 {
        1.0 - self.gases.iter().map(|g| g.a).sum::<f64>()
    }

    /// (136.7): `eps(L) = sum_i a_i (1 - exp(-kappa_i L))`, by `expm1`.
    pub fn emissivity(&self, l: f64) -> f64 {
        self.gases.iter().map(|g| g.a * -(-g.kappa * l).exp_m1()).sum()
    }

    /// `wsggGrey`'s one grey `kappa = -ln(1 - eps(L_m))/L_m` (SPEC-LIT
    /// §136.4, DESIGN). `L_m` must be finite and positive, and `eps(L_m)`
    /// must lie in `(0, 1)` - outside it the logarithm has no grey answer,
    /// and the call is refused naming the value.
    pub fn grey_kappa(&self, l_m: f64) -> Result<f64> {
        if !l_m.is_finite() || l_m <= 0.0 {
            return Err(Error::Config(format!(
                "radiation::wsgg::GreyGasSet::grey_kappa: the mean beam length \
                 {l_m} m is refused - it must be finite and positive (SPEC-LIT \
                 §136.4)"
            )));
        }
        let eps = self.emissivity(l_m);
        if !(eps > 0.0 && eps < 1.0) {
            return Err(Error::Config(format!(
                "radiation::wsgg::GreyGasSet::grey_kappa: eps(L_m = {l_m} m) = \
                 {eps} lies outside (0, 1), so -ln(1 - eps)/L_m has no grey \
                 value (SPEC-LIT §136.4)"
            )));
        }
        Ok(-(-eps).ln_1p() / l_m)
    }
}

/// `L_m = 3.6 V/A` of the domain (SPEC-LIT §136.4, DESIGN): `V` the sum of
/// the cell volumes and `A` the area of the enclosure's own surface - the
/// faces of `wall` and plain patches. Empty, symmetry and coupled faces are
/// not a surface of the enclosure and are not counted. A mesh with no such
/// face is refused.
// `f64::from` widens `Scalar`: a no-op in the f64 build, needed under `single`.
#[allow(clippy::useless_conversion)]
pub fn mean_beam_length(hm: &HostMesh) -> Result<f64> {
    let v: f64 = hm.v.iter().map(|&x| f64::from(x)).sum();
    let mut a = 0.0f64;
    for p in &hm.patches {
        if matches!(p.kind, PatchKind::Wall | PatchKind::Generic) {
            a += hm.b_mag_sf[p.start..p.start + p.size].iter().map(|&x| f64::from(x)).sum::<f64>();
        }
    }
    if a <= 0.0 || v <= 0.0 {
        return Err(Error::Config(format!(
            "radiation::wsgg::mean_beam_length: V = {v} m3 and A = {a} m2 - the \
             mean beam length 3.6 V/A needs a volume and a wall or plain patch \
             surface (SPEC-LIT §136.4)"
        )));
    }
    Ok(3.6 * v / a)
}

/// `wsgg` and `wsggGrey` need `H2O` and `CO2` among the species (SPEC-LIT
/// §136.4): returns their indices in `species`, or refuses naming the one
/// that is missing.
pub fn require_h2o_co2(species: &[&str]) -> Result<(usize, usize)> {
    let find = |n: &str| species.iter().position(|s| *s == n);
    match (find("H2O"), find("CO2")) {
        (Some(h), Some(c)) => Ok((h, c)),
        (h, c) => {
            let missing: Vec<&str> = [("H2O", h), ("CO2", c)]
                .iter()
                .filter(|(_, i)| i.is_none())
                .map(|(n, _)| *n)
                .collect();
            Err(Error::Config(format!(
                "absorptionEmissionModel wsgg/wsggGrey is refused: the species \
                 carry no {} - WSGG's weights and kappas are functions of X_H2O \
                 and X_CO2 (SPEC-LIT §136.4)",
                missing.join(" and no ")
            )))
        }
    }
}

// ==========================================================================
//  Bordbar 2021's coefficient file
// ==========================================================================

/// One Bordbar 2021 coefficient file, read: `c[i][j][k]` multiplies
/// `M_r^(4-k) T̂^j` in gas `i`'s weight (136.8)-(136.9), and `d[i][k]`
/// multiplies `M_r^(4-k)` in its `kappa_p,i` (1/(atm m)).
#[derive(Debug, Clone, PartialEq)]
pub struct WsggCoefficients {
    pressure_atm: f64,
    c: [[[f64; 5]; 5]; WSGG_N_GASES],
    d: [[f64; 5]; WSGG_N_GASES],
}

impl WsggCoefficients {
    /// Parse the file's text for the total pressure `pressure_atm`. A data
    /// row is eight whitespace-separated numbers `Pt i j b4 b3 b2 b1 b0`
    /// (the supplement's layout): `i = 1..4` the grey gas, `j = 1..5` the
    /// row of `T̂^(j-1)`, `j = 0` the row of `kappa_p,i`, and `b4..b0` the
    /// coefficients of `M_r^4..M_r^0`. Every other line is a header. All 24
    /// rows must be present exactly once, and every row's `Pt` must equal
    /// `pressure_atm` exactly; anything else is refused by name.
    pub fn parse(text: &str, pressure_atm: f64) -> Result<Self> {
        let mut c = [[[0.0f64; 5]; 5]; WSGG_N_GASES];
        let mut d = [[0.0f64; 5]; WSGG_N_GASES];
        let mut seen = [[false; 6]; WSGG_N_GASES];
        for (ln, line) in text.lines().enumerate() {
            let tok: Vec<&str> = line.split_whitespace().collect();
            if tok.len() != 8 {
                continue;
            }
            let Ok(v) = tok.iter().map(|t| t.parse::<f64>()).collect::<std::result::Result<Vec<f64>, _>>()
            else {
                continue;
            };
            let (Ok(i), Ok(j)) = (tok[1].parse::<usize>(), tok[2].parse::<usize>()) else {
                return Err(Error::Config(format!(
                    "WSGG file, line {}: the gas and row indices '{} {}' are not \
                     integers (SPEC-LIT §136.9)",
                    ln + 1,
                    tok[1],
                    tok[2]
                )));
            };
            if !(1..=WSGG_N_GASES).contains(&i) || j > 5 {
                return Err(Error::Config(format!(
                    "WSGG file, line {}: gas {i} row {j} is outside the model's \
                     4 gases and rows 0-5 (SPEC-LIT §136.9)",
                    ln + 1
                )));
            }
            if v[0] != pressure_atm {
                return Err(Error::Config(format!(
                    "WSGG file, line {}: the row is for Pt = {} atm, and the case \
                     names {pressure_atm} atm - the file belongs to another \
                     pressure (SPEC-LIT §136.4)",
                    ln + 1,
                    v[0]
                )));
            }
            if seen[i - 1][j] {
                return Err(Error::Config(format!(
                    "WSGG file, line {}: gas {i} row {j} appears twice (SPEC-LIT \
                     §136.9)",
                    ln + 1
                )));
            }
            seen[i - 1][j] = true;
            let row = [v[3], v[4], v[5], v[6], v[7]];
            if let Some(bad) = row.iter().find(|x| !x.is_finite()) {
                return Err(Error::Config(format!(
                    "WSGG file, line {}: coefficient {bad} is not finite (SPEC-LIT \
                     §136.9)",
                    ln + 1
                )));
            }
            if j == 0 {
                d[i - 1] = row;
            } else {
                c[i - 1][j - 1] = row;
            }
        }
        for (i, s) in seen.iter().enumerate() {
            if let Some(j) = s.iter().position(|x| !x) {
                return Err(Error::Config(format!(
                    "WSGG file: gas {} row {j} is missing - the file must carry \
                     rows 0-5 for each of the 4 gases (SPEC-LIT §136.9)",
                    i + 1
                )));
            }
        }
        Ok(Self { pressure_atm, c, d })
    }

    /// Read `path` for `pressure_atm` (one of
    /// [`super::props::WSGG_PRESSURES_ATM`]) and check its sha256 against
    /// [`WSGG_DIGESTS`] (SPEC-LIT §136.4). The file recorded for this
    /// pressure is accepted silently (`None`); a digest that is recorded
    /// for ANOTHER pressure is refused; an unknown digest is accepted with
    /// the warning returned and printed, naming it - a user may hold a later
    /// revision.
    pub fn read(path: &Path, pressure_atm: f64) -> Result<(Self, Option<String>)> {
        if !super::props::WSGG_PRESSURES_ATM.contains(&pressure_atm) {
            return Err(Error::Config(format!(
                "WSGG: {pressure_atm} atm is refused - the model exists at the \
                 ten pressures {:?} atm only (SPEC-LIT §136.4)",
                super::props::WSGG_PRESSURES_ATM
            )));
        }
        let bytes = std::fs::read(path).map_err(|e| {
            Error::Config(format!("WSGG file '{}': {e} (SPEC-LIT §136.4)", path.display()))
        })?;
        let digest = sha256_hex(&bytes);
        let mut warning = None;
        match WSGG_DIGESTS.iter().find(|(_, _, h)| *h == digest) {
            Some((p, _, _)) if *p == pressure_atm => {}
            Some((p, name, _)) => {
                return Err(Error::Config(format!(
                    "WSGG file '{}' is {name} (sha256 {digest}), the file for {p} \
                     atm, and the case names {pressure_atm} atm (SPEC-LIT §136.4)",
                    path.display()
                )));
            }
            None => {
                let w = format!(
                    "WSGG file '{}': sha256 {digest} is not one SPEC-LIT §136.4 \
                     records - accepted as a user revision, not the published file",
                    path.display()
                );
                eprintln!("{w}");
                warning = Some(w);
            }
        }
        let text = String::from_utf8_lossy(&bytes);
        let coef = Self::parse(&text, pressure_atm).map_err(|e| {
            Error::Config(format!("WSGG file '{}': {e}", path.display()))
        })?;
        Ok((coef, warning))
    }

    /// The total pressure the file is for, atm.
    pub fn pressure_atm(&self) -> f64 {
        self.pressure_atm
    }

    /// (136.8)-(136.10) at one state: the four grey gases' `a_i` and
    /// `kappa_i = kappa_p,i (X_H2O + X_CO2) P_t`, 1/m. Refused by name
    /// (§136.4): mole fractions that are not finite or lie outside `[0, 1]`,
    /// `X_CO2 = 0` (`M_r` is undefined), and `T` or `M_r` outside the fitted
    /// range - the curves are not extrapolated (§128.1's rule).
    pub fn gases(&self, t: f64, x_h2o: f64, x_co2: f64) -> Result<GreyGasSet> {
        for (n, x) in [("X_H2O", x_h2o), ("X_CO2", x_co2)] {
            if !x.is_finite() || !(0.0..=1.0).contains(&x) {
                return Err(Error::Config(format!(
                    "WSGG: {n} = {x} is refused - a mole fraction lies in [0, 1] \
                     (SPEC-LIT §136.4)"
                )));
            }
        }
        if x_co2 == 0.0 {
            return Err(Error::Config(format!(
                "WSGG: X_CO2 = 0 (X_H2O = {x_h2o}) is refused - M_r = X_H2O/X_CO2 \
                 is undefined (SPEC-LIT §136.4)"
            )));
        }
        let (t_lo, t_hi) = WSGG_T_RANGE_K;
        if !t.is_finite() || t < t_lo || t > t_hi {
            return Err(Error::Config(format!(
                "WSGG: T = {t} K is outside the fitted range {t_lo} <= T <= {t_hi} \
                 K - not extrapolated (SPEC-LIT §136.4)"
            )));
        }
        let mr = x_h2o / x_co2;
        let (m_lo, m_hi) = WSGG_MR_RANGE;
        if mr < m_lo || mr > m_hi {
            return Err(Error::Config(format!(
                "WSGG: M_r = X_H2O/X_CO2 = {mr} is outside the fitted range {m_lo} \
                 <= M_r <= {m_hi} - not extrapolated (SPEC-LIT §136.4)"
            )));
        }
        // (136.9)'s polynomials in M_r, b4 first (Horner); (136.8)'s in T̂.
        let poly_mr = |r: &[f64; 5]| r.iter().fold(0.0f64, |acc, &b| acc * mr + b);
        let th = t / WSGG_T_REF_K;
        let pp = (x_h2o + x_co2) * self.pressure_atm;
        let mut gases = Vec::with_capacity(WSGG_N_GASES);
        for i in 0..WSGG_N_GASES {
            let a = self.c[i].iter().rev().fold(0.0f64, |acc, r| acc * th + poly_mr(r));
            gases.push(GreyGas { a, kappa: poly_mr(&self.d[i]) * pp });
        }
        GreyGasSet::new(gases)
    }
}

// ==========================================================================
//  The non-grey loop: the gas sums
// ==========================================================================

/// `G`, `q` and the wall flux summed over the grey gases in `ofacc`, with
/// their one rounding (SPEC-LIT §136.9, §118.1) - every buffer allocated at
/// construction, so the sums are launches only.
struct GasSums {
    accum_cells: CudaFunction,
    accum_faces: CudaFunction,
    round_cells: CudaFunction,
    round_faces: CudaFunction,
    n_cells: usize,
    n_bf: usize,
    g_acc: DevBuf<Acc>,
    q_acc: DevBuf<Acc>,
    wf_acc: DevBuf<Acc>,
    g: DevBuf<Scalar>,
    q: DevBuf<Vec3>,
    wall_flux: DevBuf<Scalar>,
}

impl GasSums {
    fn new(gpu: &Gpu, n_cells: usize, n_bf: usize) -> Result<Self> {
        let k = KernelSet::new(gpu, crate::kernels::RADIATION)?;
        Ok(Self {
            accum_cells: k.func("wsggAccumCells")?,
            accum_faces: k.func("wsggAccumFaces")?,
            round_cells: k.func("domRoundCells")?,
            round_faces: k.func("domRoundFaces")?,
            n_cells,
            n_bf,
            g_acc: gpu.zeros(n_cells)?,
            q_acc: gpu.zeros(3 * n_cells)?,
            wf_acc: gpu.zeros(n_bf)?,
            g: gpu.zeros(n_cells)?,
            q: gpu.zeros(n_cells)?,
            wall_flux: gpu.zeros(n_bf)?,
        })
    }

    fn zero(&mut self, gpu: &Gpu) -> Result<()> {
        gpu.fill_zero(&mut self.g_acc)?;
        gpu.fill_zero(&mut self.q_acc)?;
        gpu.fill_zero(&mut self.wf_acc)?;
        Ok(())
    }

    fn add(&mut self, gpu: &Gpu, g: &DevBuf<Scalar>, q: &DevBuf<Vec3>, wf: &DevBuf<Scalar>) -> Result<()> {
        let (n, nb) = (self.n_cells, self.n_bf);
        let (nl, nbl) = (n as Label, nb as Label);
        if n > 0 {
            unsafe {
                gpu.stream()
                    .launch_builder(&self.accum_cells)
                    .arg(&mut self.g_acc)
                    .arg(&mut self.q_acc)
                    .arg(g)
                    .arg(q)
                    .arg(&nl)
                    .launch(cfg_for(n))?;
            }
        }
        if nb > 0 {
            unsafe {
                gpu.stream()
                    .launch_builder(&self.accum_faces)
                    .arg(&mut self.wf_acc)
                    .arg(wf)
                    .arg(&nbl)
                    .launch(cfg_for(nb))?;
            }
        }
        Ok(())
    }

    fn round(&mut self, gpu: &Gpu) -> Result<()> {
        let (n, nb) = (self.n_cells, self.n_bf);
        let (nl, nbl) = (n as Label, nb as Label);
        if n > 0 {
            unsafe {
                gpu.stream()
                    .launch_builder(&self.round_cells)
                    .arg(&mut self.g)
                    .arg(&mut self.q)
                    .arg(&self.g_acc)
                    .arg(&self.q_acc)
                    .arg(&nl)
                    .launch(cfg_for(n))?;
            }
        }
        if nb > 0 {
            unsafe {
                gpu.stream()
                    .launch_builder(&self.round_faces)
                    .arg(&mut self.wall_flux)
                    .arg(&self.wf_acc)
                    .arg(&nbl)
                    .launch(cfg_for(nb))?;
            }
        }
        Ok(())
    }
}

/// The non-grey P1 loop (SPEC-LIT §136.9): one [`P1`] reused for every
/// grey gas of the set - `kappa -> kappa_i`, every emission weighted by
/// `a_i` - with `G`, `q_r` and the wall flux summed over the gases.
pub struct WsggP1 {
    p1: P1,
    set: GreyGasSet,
    sums: GasSums,
}

impl WsggP1 {
    /// The loop for `set` in a medium scattering with `sigma_s` and `a1`,
    /// with [`P1::new`]'s walls, controls and refusals.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        gpu: &Gpu,
        hm: &HostMesh,
        m: &GpuMesh,
        set: GreyGasSet,
        sigma_s: f64,
        a1: f64,
        walls: &[MarshakPatch],
        ctrl: SolverControls,
    ) -> Result<Self> {
        let medium = P1Medium { kappa: set.gases()[0].kappa, sigma_s, a1 };
        let p1 = P1::new(gpu, hm, m, &medium, walls, ctrl)?;
        let sums = GasSums::new(gpu, hm.n_cells, hm.n_boundary_faces)?;
        Ok(Self { p1, set, sums })
    }

    /// Solve every grey gas and sum `G = sum_i G_i`, `q_r = sum_i q_r,i` and
    /// the wall flux in `ofacc`, rounded once. With `sources`, each gas
    /// registers its own (137.2) pair right after its solve, so the energy
    /// equation receives `sum_i kappa_i (G_i - 4 a_i sigma T^4)` linearised
    /// (§136.4). Returns each gas's solver performance, in order. Launches
    /// only, so the loop captures (SPEC-LIT §81) when the controls allow
    /// [`P1::solve`] to.
    pub fn solve(
        &mut self,
        gpu: &Gpu,
        m: &GpuMesh,
        t: &GpuScalarField,
        mut sources: Option<&mut EnergySources>,
    ) -> Result<Vec<SolverPerformance>> {
        self.sums.zero(gpu)?;
        let mut perf = Vec::with_capacity(self.set.gases().len());
        for j in 0..self.set.gases.len() {
            let gas = self.set.gases[j];
            self.p1.set_grey_gas(gpu, gas.kappa, gas.a)?;
            perf.push(self.p1.solve(gpu, m, t)?);
            if let Some(s) = sources.as_deref_mut() {
                self.p1.register_energy_source(gpu, t, s)?;
            }
            self.sums.add(gpu, &self.p1.g().f, self.p1.qr(), self.p1.wall_flux())?;
        }
        self.sums.round(gpu)?;
        Ok(perf)
    }

    /// The set the loop solves.
    pub fn set(&self) -> &GreyGasSet {
        &self.set
    }

    /// `[n_cells]` `G = sum_i G_i`.
    pub fn g(&self) -> &DevBuf<Scalar> {
        &self.sums.g
    }

    /// `[n_cells]` `q_r = sum_i q_r,i`.
    pub fn qr(&self) -> &DevBuf<Vec3> {
        &self.sums.q
    }

    /// `[n_bf]` the wall flux summed over the gases, 0 off the Marshak faces.
    pub fn wall_flux(&self) -> &DevBuf<Scalar> {
        &self.sums.wall_flux
    }
}

/// The non-grey fvDOM loop (SPEC-LIT §136.9): one [`DomSolve`] reused for
/// every grey gas of the set - the same rows with `kappa -> kappa_i` and
/// emission `a_i kappa_i I_b`, the intensity storage shared (§136.4) - with
/// `G`, `q` and the wall flux summed over the gases.
pub struct WsggDom {
    dom: DomSolve,
    set: GreyGasSet,
    sums: GasSums,
}

impl WsggDom {
    /// The loop for `set` in a medium scattering with `sigma_s` and `a1`,
    /// with [`DomSolve::new`]'s quadrature, patches, controls and refusals.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        gpu: &Gpu,
        hm: &HostMesh,
        m: &GpuMesh,
        set: GreyGasSet,
        sigma_s: f64,
        a1: f64,
        quad: &Quadrature,
        patches: &[DomPatch],
        ctrl: DomControls,
    ) -> Result<Self> {
        let medium = DomMedium { kappa: set.gases()[0].kappa, sigma_s, a1 };
        let dom = DomSolve::new(gpu, hm, m, &medium, quad, patches, ctrl)?;
        let sums = GasSums::new(gpu, hm.n_cells, hm.n_boundary_faces)?;
        Ok(Self { dom, set, sums })
    }

    /// Solve every grey gas to [`DomSolve::solve`]'s own convergence and sum
    /// `G`, `q` and the wall flux over the gases in `ofacc`, rounded once.
    /// Returns each gas's report, in order - an unconverged gas is reported
    /// there (and printed by [`DomSolve::solve`]), never accepted silently.
    /// [`DomSolve::solve`] reads `G` back every sweep, so the loop is
    /// refused inside a capture, as it is.
    pub fn solve(&mut self, gpu: &Gpu, m: &GpuMesh, t: &GpuScalarField) -> Result<Vec<DomReport>> {
        self.sums.zero(gpu)?;
        let mut reports = Vec::with_capacity(self.set.gases().len());
        for j in 0..self.set.gases.len() {
            let gas = self.set.gases[j];
            self.dom.set_grey_gas(gpu, gas.kappa, gas.a)?;
            reports.push(self.dom.solve(gpu, m, t)?);
            self.sums.add(gpu, self.dom.g(), self.dom.q(), self.dom.wall_flux())?;
        }
        self.sums.round(gpu)?;
        Ok(reports)
    }

    /// The set the loop solves.
    pub fn set(&self) -> &GreyGasSet {
        &self.set
    }

    /// The solve the loop reuses.
    pub fn dom(&self) -> &DomSolve {
        &self.dom
    }

    /// `[n_cells]` `G = sum_i G_i`.
    pub fn g(&self) -> &DevBuf<Scalar> {
        &self.sums.g
    }

    /// `[n_cells]` `q = sum_i q_i`.
    pub fn q(&self) -> &DevBuf<Vec3> {
        &self.sums.q
    }

    /// `[n_bf]` `q_w . n` summed over the gases; 0 on empty faces.
    pub fn wall_flux(&self) -> &DevBuf<Scalar> {
        &self.sums.wall_flux
    }
}

// ==========================================================================
//  Tests
// ==========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::blockgen::{self, BlockSpec, GradedAxis};
    use crate::radiation::p1::WallTemperature;
    use crate::solver::LinearSolverKind;

    fn gpu() -> Option<Gpu> {
        Gpu::new(0).ok()
    }

    /// `wsgg_p1_replays_under_capture` (SPEC-LIT §81, §136.9): the non-grey
    /// P1 loop - two grey gases on an 8-cell slab, walls at 500 K, each gas
    /// registering its energy source - captures and replays bitwise.
    #[test]
    fn wsgg_p1_replays_under_capture() {
        let Some(gpu) = gpu() else { return };
        let report = crate::capture::capture_replays_bitwise(
            &gpu,
            "the WSGG non-grey P1 loop (SPEC-LIT 136.9)",
            || {
                let axis = |hi: f64, n: usize| GradedAxis {
                    lo: 0.0,
                    hi: hi as Scalar,
                    n,
                    expansion: 1.0,
                    two_sided: false,
                };
                let spec = BlockSpec {
                    x: axis(1.0, 8),
                    y: axis(0.125, 1),
                    z: axis(0.125, 1),
                    patch_name: BlockSpec::default().patch_name,
                    patch_type: ["wall", "wall", "empty", "empty", "empty", "empty"]
                        .map(String::from),
                    windows: Vec::new(),
                    cyclic: Vec::new(),
                };
                let hm = blockgen::build_mesh(&spec)?;
                let m = GpuMesh::upload(&gpu, &hm)?;
                let mut t = GpuScalarField::zeros(&gpu, &m, "T")?;
                gpu.write(&mut t.f, &vec![1000.0 as Scalar; hm.n_cells])?;
                gpu.write(&mut t.bf, &vec![1000.0 as Scalar; hm.n_boundary_faces])?;
                let walls: Vec<MarshakPatch> = ["xMin", "xMax"]
                    .iter()
                    .map(|p| MarshakPatch {
                        patch: (*p).to_string(),
                        emissivity: 1.0,
                        temperature: WallTemperature::Fixed(500.0),
                    })
                    .collect();
                let set = GreyGasSet::new(vec![
                    GreyGas { a: 0.6, kappa: 0.5 },
                    GreyGas { a: 0.3, kappa: 4.0 },
                ])?;
                let ctrl = SolverControls {
                    solver: LinearSolverKind::PCG,
                    fixed_iters: true,
                    report_residuals: false,
                    max_iter: 20,
                    ..SolverControls::default()
                };
                let wp = WsggP1::new(&gpu, &hm, &m, set, 0.0, 0.0, &walls, ctrl)?;
                let sources = EnergySources::new(&gpu, &m)?;
                Ok((wp, m, t, sources))
            },
            |(wp, m, t, sources): &mut (WsggP1, GpuMesh, GpuScalarField, EnergySources)| {
                sources.clear(&gpu)?;
                wp.solve(&gpu, m, t, Some(sources)).map(|_| ())
            },
            |(wp, _m, _t, sources): &(WsggP1, GpuMesh, GpuScalarField, EnergySources)| {
                let qr = gpu.download(wp.qr())?;
                Ok(vec![
                    ("G", gpu.download(wp.g())?),
                    ("wall flux", gpu.download(wp.wall_flux())?),
                    ("qr x", qr.iter().map(|v| v.x).collect()),
                    ("qr y", qr.iter().map(|v| v.y).collect()),
                    ("qr z", qr.iter().map(|v| v.z).collect()),
                    ("sources q", gpu.download(sources.q())?),
                    ("sources sp", gpu.download(sources.sp())?),
                ])
            },
        )
        .expect("SPEC-LIT 81.7: the WSGG P1 loop must capture and replay bitwise");
        println!("  WSGG P1: {report}");
    }
}
