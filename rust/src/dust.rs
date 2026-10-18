// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! The equivalent premixed gas of a corn-starch dust flame: unburned gas U
//! (the dust as a "vapour" plus air at equivalence ratio phi) and burned gas
//! B (the complete products), each one NASA-7 pseudo-species carried by the
//! CHEMRAD thermo unchanged, on host and device alike. The formula mode
//! (REMBE) and the prescribed mode (Leeds), the isobaric burned state, the
//! AICC state and the laminar burning velocity S_L(phi) are SPEC-LIT
//! section 152's contract; everything here is `f64`, never [`crate::Scalar`],
//! so the two builds agree bit for bit.
//!
//! Written from:
//!   SPEC-LIT §152 - the global reaction (152.1), phi (152.2), the starch
//!     polynomial and its derived a5 (152.4)-(152.5), the heat of reaction
//!     (152.6), the NASA-7 form and the two pseudo-species (152.7)-(152.8),
//!     the state (152.9), the isobaric and AICC burned states (152.10)-(152.11),
//!     the heat release (152.12) and S_L (152.13)-(152.14)
//!   SPEC-LIT §153 - the laminar-limit eigenproblem of `psi`: the published
//!     closure (153.1), the reaction time (153.2), the eigenproblem (153.3),
//!     the integral identity (153.4) and the ZFK asymptote (153.5)
//!   SPEC-LIT §154 - the FSC and TFC closures of `fsc` and their kernels in
//!     `device` (`cuda/dust.cu`): (154.1)-(154.11), the f32-safe rate
//!     constant and its zero rule
//!   SPEC-LIT §155 - the progress-variable equation (155.3) of `progress`
//!     on the device, its ignition source (155.4) in `ignition`, the
//!     flame-Courant number (155.9) in `courant`, the BML density, the
//!     Reynolds mean and rho_u(p) (155.1), (155.2), (155.10), and the 1-D
//!     flame tube `tube1d` that holds it against (155.5)
//!   SPEC-LIT §159 - the reacting parcel of Phase B in `particle`: the
//!     particle energy (159.1) with Ranz-Marshall (159.2) and Bird (159.3),
//!     drying by §76's evaporation, SFOM (159.4), the radiation temperature
//!     (159.7) and the accumulators of §159.7; its kernel `cuda/dustparticle.cu`;
//!     and §159.8 - the char reaction (159.5) with its heat retention in `char`
//!   A. Islas, A. Rodriguez Fernandez, C. Betegon, E. Martinez-Paneda,
//!     A. Pandal, arXiv:2207.13149 (2022) - eqs. (13)-(17), (20)-(23)
//!   C. Huang, M. Bloching & A. N. Lipatnikov, J. Loss Prev. Process Ind. 75
//!     (2022) 104707, DOI 10.1016/j.jlp.2021.104707 - eqs. 1-3 and Table 1
//!   C. Huang, A. N. Lipatnikov & K. Nessvi, J. Loss Prev. Process Ind. 67
//!     (2020) 104237, DOI 10.1016/j.jlp.2020.104237 - Table 1
//!   NASA TM-4513 Table II through §128
//! No GPL-licensed source was consulted.

pub mod age;
pub mod char;
pub mod cloud;
pub mod courant;
pub mod device;
pub mod fsc;
pub mod gatemesh;
pub mod ignition;
pub mod laminar;
pub mod particle;
pub mod progress;
pub mod psi;
pub mod seeding;
pub mod source;
pub mod thermo;
pub mod tube1d;
pub mod vent;
pub mod vessel0d;

pub use fsc::{CombustionModel, FscConstants, FscInputs, FscOut, FscState};
pub use ignition::IgnitionSource;
pub use laminar::{LaminarSpeed, SlFactors};
pub use progress::{ProgressCoupling, ProgressEquation, ProgressSolve};
pub use psi::{AdiabaticCurve, FlameCurve, PsiOptions, PsiProblem, PsiSolution, PublishedCurve};
pub use thermo::{
    AiccState, BurnedState, EquivalentGas, FormulaInputs, PrescribedInputs, StarchFuel, ThermoMode,
};
