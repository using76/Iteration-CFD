// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! DST-04's gates: the progress-variable equation of the dust flame
//! (SPEC-LIT §155) on the device, held in the 1-D flame tube of
//! `src/dust/tube1d.rs` against the exact planar solution in frozen
//! turbulence, the laminar limit at S_L (regenerated Psi) and at the ZFK
//! speed, the flame-Courant independence and refusal, the ignition kernel
//! and the bounds counter (§155.8). The same case file runs as the unit
//! tests `dust::progress::tests`, so both builds run it.
//!
//! An integration test, so it links the build's library: under `single`
//! the equation is solved in float.
//! The specification was informed by a review of FSCDustFoam (Huang &
//! Lipatnikov, GPL) and no code was copied.
//! No GPL-licensed source was consulted.

mod progress {
    include!("shared/dust_progress_cases.rs");
}
