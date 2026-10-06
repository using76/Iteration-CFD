// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! Chemistry: the NASA seven-coefficient species thermochemistry read from
//! a CHEMKIN THERMO block, and the ideal-gas mixture built on it. Host only,
//! and f64 in every type, so the module and its tests behave identically in
//! the default build and in `--features single`; SPEC-LIT §128-§133 are its
//! contract.

pub mod thermo;
#[cfg(test)]
mod tests_thermo;
pub use thermo::{ElementTable, Interval, SpeciesThermo, ThermoSet, P_ATM, P_BAR, R_GAS};
