// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

//! The CHEMRAD sections' shared fixture (docs/17 CHR-27): the TM-4513
//! `ThermoSet` and the ten-species mechanism built over it, the two
//! stoichiometric mole recipes, and the small pure helpers the chr_* gate
//! files port their source tests through. No gate of its own: no GATE const,
//! no rows - everything here is setup a gate reaches with `?`.
//!
//! Written from:
//!   ofgpu `SPEC-LIT.md` §128.6/§130.7 (the TM-4513 fixture and its species
//!     code table), §134.5-§134.7 (the TCI tests the EDM section ports)
//!   tests/chem_equilibrium.rs (`tm_set`, `mech`, `y_from_x`, the mole
//!     recipes), src/chem/tests_thermo.rs and tests/radiation_dom.rs
//!     (`e1`/`e3`/`psi_exact`), tests/chem_device.rs (`SplitMix64`),
//!     tests/bc_atmospheric.rs (`w64`)
//! No GPL-licensed source was consulted.

use ofgpu::chem::{Mechanism, SpeciesThermo, ThermoSet, P_BAR};
use ofgpu::{Result, Scalar};

use crate::{key, Checks};

/// The table's species codes: code, name and elements (SPEC-LIT §128.6's
/// code list), in CSV order - exactly `tests/chem_equilibrium.rs`'s `CODES`.
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

/// The TM-4513 Table II key through the manifest: its digest line printed
/// once through `c`, its verified text returned. Every CHEMRAD section that
/// builds the ten-species fixture comes through here, so the number its
/// rows hold is the number the manifest pinned.
pub(crate) fn table_ii(c: &Checks) -> Result<String> {
    let (line, text) = key::load_text("mcbride-gordon-reno1993-table-II")?;
    c.note(&line);
    Ok(text)
}

/// The records as a set at `P_BAR`, parsed exactly as
/// `tests/chem_equilibrium.rs::tm_set` parses its CSV: TM-4513 tabulates at
/// 1 bar (SPEC-LIT §128.4) and the mechanism reader moves the set to `P_ATM`
/// itself; the row whose `t_lo` is 1000 is `upper`, the one whose `t_lo` is
/// 200 is `lower`; the bounds are 200/1000/6000 K for all ten species; and
/// `molar_mass` starts as the table's own column, the mechanism reader
/// rebuilding it from the element weights.
pub(crate) fn tm_set(text: &str) -> Result<ThermoSet> {
    let mut rows: Vec<(String, f64, f64, [f64; 7])> = Vec::new();
    for line in text.lines() {
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
    ThermoSet::from_species(v, P_BAR, "TM-4513")
}

/// The loaded key as its set: the CHEMRAD fixture every Gibbs, reactor and
/// TCI row of sections 86-88 runs on.
pub(crate) fn tm4513(c: &Checks) -> Result<ThermoSet> {
    tm_set(&table_ii(c)?)
}

/// A mechanism of the given reaction lines over the ten TM-4513 species, as
/// `tests/chem_equilibrium.rs::mech` builds it; the concentration order is
/// H2 O2 H2O OH O H CO CO2 CH4 N2 (0..9).
pub(crate) fn mech(set: &ThermoSet, rx: &[&str]) -> Result<Mechanism> {
    let mut text: Vec<&str> = vec![
        "ELEMENTS H O C N END",
        "SPECIES H2 O2 H2O OH O H CO CO2 CH4 N2 END",
        "REACTIONS",
    ];
    text.extend_from_slice(rx);
    text.push("END");
    Mechanism::read_chemkin(&text.join("\n"), "kin-fixture", Some(set))
}

/// Mole fractions to mass fractions against `set`: y_j = x_j W_j / sum x W.
pub(crate) fn y_from_x(set: &ThermoSet, x: &[f64]) -> Vec<f64> {
    let xw: Vec<f64> = x
        .iter()
        .zip(set.species())
        .map(|(xj, s)| xj * s.molar_mass)
        .collect();
    let s: f64 = xw.iter().sum();
    xw.iter().map(|v| v / s).collect()
}

/// Stoichiometric H2-air by moles: 2 H2 + 1 O2 + 3.76 N2 (phi = 1).
pub(crate) fn h2_air(set: &ThermoSet) -> Vec<f64> {
    let mut x = vec![0.0; set.species().len()];
    x[0] = 2.0;
    x[1] = 1.0;
    x[9] = 3.76;
    y_from_x(set, &x)
}

/// Stoichiometric CH4-air by moles: 1 CH4 + 2 O2 + 7.52 N2 (phi = 1).
pub(crate) fn ch4_air(set: &ThermoSet) -> Vec<f64> {
    let mut x = vec![0.0; set.species().len()];
    x[1] = 2.0;
    x[8] = 1.0;
    x[9] = 7.52;
    y_from_x(set, &x)
}

/// The build's `Scalar` widened to the f64 the answer keys are computed in.
pub(crate) fn w64(x: Scalar) -> f64 {
    x as f64
}

/// `E_1(x)`, x > 0: Abramowitz & Stegun 5.1.11's series for x <= 1, the
/// continued fraction 5.1.22 (evaluated backward, 200 levels) for x > 1,
/// exactly as `tests/radiation_dom.rs` evaluates it.
pub(crate) fn e1(x: f64) -> f64 {
    if x <= 1.0 {
        let euler = 0.577_215_664_901_532_9_f64;
        let (mut sum, mut term) = (0.0f64, 1.0f64);
        for k in 1..60 {
            let kf = k as f64;
            term *= -x / kf; // (-x)^k / k!
            sum -= term / kf; // + (-1)^(k+1) x^k / (k k!)
        }
        -euler - x.ln() + sum
    } else {
        let mut f = x;
        for k in (1..=200).rev() {
            let kf = k as f64;
            f = x + kf / (1.0 + kf / f);
        }
        (-x).exp() / f
    }
}

/// `E_3(x)` by A&S 5.1.14 twice: `E_(n+1) = (e^-x - x E_n)/n`.
pub(crate) fn e3(x: f64) -> f64 {
    let ex = (-x).exp();
    let e2 = ex - x * e1(x);
    (ex - x * e2) / 2.0
}

/// (135.6): the exact slab's wall flux over `sigma T^4`, `1 - 2 E_3(tau)`.
pub(crate) fn psi_exact(tau: f64) -> f64 {
    1.0 - 2.0 * e3(tau)
}

/// SplitMix64, seeded per gate; `u()` is uniform on [0, 1) as
/// (x >> 11) 2^-53 (tests/chem_device.rs's generator).
pub(crate) struct SplitMix64(pub u64);

impl SplitMix64 {
    pub(crate) fn u(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        (z >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }
}

