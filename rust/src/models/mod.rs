// meteor-cfd - Copyright (c) 2026 주식회사 메테오시뮬레이션 (Meteo Simulation Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

//! Turbulence models - SPEC-LIT §6.1, §6.2, §33, §40 and §41.
//!
//! Written from:
//!   Launder & Spalding, *Comput. Methods Appl. Mech. Eng.* 3 (1974) 269-289
//!   Wilcox, *Turbulence Modeling for CFD*, DCW Industries - the 1988 form
//!   Launder & Sharma, *Letters in Heat and Mass Transfer* 1 (1974) 131-138 -
//!     the low-Reynolds-number extension of the first, SPEC-LIT §33
//!   Shih, Liou, Shabbir, Yang & Zhu, NASA TM-106721 (1994) - the realizable
//!     variant of the first, SPEC-LIT §40
//!   Yakhot, Orszag, Thangam, Gatski & Speziale, ICASE 91-65 / NASA CR-187611
//!     (1991) - the RNG variant of the first, SPEC-LIT §41
//!   ofgpu `SPEC-LIT.md` §6, §33, §40, §41
//! No GPL-licensed source was consulted.
//!
//! Each model is its own type, and none of them share a trait with each
//! other. That is deliberate: a `dyn` dispatch has no place in a solver's
//! inner loop, and the models do not have the same interface anyway - one
//! carries `epsilon`, another `omega`, and pretending otherwise would mean a
//! `dissipation_field()` accessor that means different things depending on
//! which model answers it (`LaunderSharmaKE` carries `epsilon` too, but the
//! quantity under that name is `epsilon_tilde` - see its own module doc).
//! Where a caller genuinely wants uniformity - `src/bin/bench.rs` times them
//! side by side - it declares a two-method trait of its own and pays one
//! virtual call per *outer iteration*, which disappears into the hundred
//! kernel launches it wraps.
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

pub mod coupled;
pub mod k_epsilon;
pub mod ke_variants;
pub mod k_omega;
pub mod k_omega_sst;
pub mod launder_sharma;
pub mod les;
pub mod registry;

pub use coupled::{
    BuoyancySettings, CombustionMixing, CoupledKEpsilon, CoupledKOmega, CoupledKOmegaSst,
    CoupledLaminar, CoupledLaunderSharmaKE, CoupledLes, CoupledRealizableKe, CoupledRngKe,
    CoupledTurbulence, ThermalCtx,
};
pub use k_epsilon::{KEpsilon, KEpsilonCoeffs};
pub use ke_variants::{RealizableKe, RealizableKeCoeffs, RngKe, RngKeCoeffs};
pub use k_omega::{KOmega, KOmegaCoeffs};
pub use launder_sharma::{f2, f_mu, mesh_resolution_report, LaunderSharmaKE, MeshResolutionReport};
pub use k_omega_sst::{KOmegaSst, KOmegaSstCoeffs};
pub use les::{Les, LesCoeffs, LesModel};
pub use registry::{
    available_models, build_coupled, buoyancy_settings, realizable_ke_coeffs,
    refuse_realizable_ke_buoyancy, rng_ke_coeffs, select_turbulence_model, RasModel,
    TurbulenceSelection,
};
