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
//! contract. A CHEMKIN mechanism (ELEMENTS, SPECIES, THERMO and REACTIONS) is
//! read into a validated [`Mechanism`] whose every reaction balances exactly
//! in rational arithmetic (SPEC-LIT §129). The finite-rate kinetics built on
//! it - the rate of progress, fall-off and the reverse rate from equilibrium
//! (SPEC-LIT §130) - lives in [`kinetics`]. Chemical equilibrium by Gibbs
//! minimisation (SPEC-LIT §131) lives in [`equilibrium`]. The stiff
//! Rosenbrock integrator (SPEC-LIT §132) lives in [`rosenbrock`].

mod ck_lexer;
pub mod equilibrium;
pub mod kinetics;
pub mod mechanism;
pub mod rosenbrock;
pub mod thermo;
#[cfg(test)]
mod tests_mechanism;
#[cfg(test)]
mod tests_thermo;
pub use kinetics::{production_rates, production_rates_tpy, rates_of_progress, Progress};
pub use equilibrium::{equilibrate_hp, equilibrate_tp, EqOptions, Equilibrium};
pub use mechanism::{Arrhenius, Blending, Collider, Falloff, Mechanism, Rational, Reaction, ThirdBody};
pub use thermo::{ElementTable, Interval, SpeciesThermo, ThermoSet, P_ATM, P_BAR, R_GAS};
pub use rosenbrock::{integrate, Method, RosOptions, RosStats};
