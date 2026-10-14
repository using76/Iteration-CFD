// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! DST-03's gates: the FSC and TFC closures of the dust flame (SPEC-LIT
//! §154) - the kernels of `cuda/dust.cu` against their f64 host twins in
//! `src/dust/fsc.rs`, the developed and early-time limits, the TFC limit,
//! the extended FSC's switch, the f32-safe Arrhenius factor and the setup's
//! refusals (§154.9). The same case file runs as the unit tests
//! `dust::fsc::tests`, so both builds run it.
//!
//! An integration test, so it links the build's library: under `single`
//! the kernels store float words and the device rows count ulps in float.
//! The specification was informed by a review of FSCDustFoam (Huang &
//! Lipatnikov, GPL) and no code was copied.
//! No GPL-licensed source was consulted.

mod fsc {
    include!("shared/dust_fsc_cases.rs");
}
