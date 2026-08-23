// meteor-cfd - Copyright (c) 2026 주식회사 메테오시뮬레이션 (Meteo Simulation Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

//! Turbulence models - SPEC-LIT §6.1 and §6.2.
//!
//! Written from:
//!   Launder & Spalding, *Comput. Methods Appl. Mech. Eng.* 3 (1974) 269-289
//!   Wilcox, *Turbulence Modeling for CFD*, DCW Industries - the 1988 form
//!   ofgpu `SPEC-LIT.md` §6
//! No GPL-licensed source was consulted.
//!
//! Each model is its own type, and the two share no trait. That is
//! deliberate: a `dyn` dispatch has no place in a solver's inner loop, and
//! the two models do not have the same interface anyway - one carries
//! `epsilon` and the other `omega`, and pretending otherwise would mean a
//! `dissipation_field()` accessor that means two different things. Where a
//! caller genuinely wants uniformity - `src/bin/bench.rs` times them side by
//! side - it declares a two-method trait of its own and pays one virtual call
//! per *outer iteration*, which disappears into the hundred kernel launches
//! it wraps.
//!
//! Which of them a case gets is [`registry`]'s job, and it is a real
//! dispatch: `RAS { model ...; }` used to be read and thrown away, so the
//! model that ran was whichever one the binary had been compiled around.
//!
//! What the two DO share is [`crate::turbulence::RasCore`]: the matrix, the
//! solver workspace, the wall-function tables, `nu_t`, and the
//! `ddt + div - laplacian` assembly that every eddy-viscosity transport
//! equation has in common. What is left in these files is the source terms,
//! which is exactly what a model *is*.

pub mod k_epsilon;
pub mod k_omega;
pub mod k_omega_sst;
pub mod les;
pub mod registry;

pub use k_epsilon::{KEpsilon, KEpsilonCoeffs};
pub use k_omega::{KOmega, KOmegaCoeffs};
pub use k_omega_sst::{KOmegaSst, KOmegaSstCoeffs};
pub use les::{Les, LesCoeffs, LesModel};
pub use registry::{available_models, select_turbulence_model, RasModel, TurbulenceSelection};
