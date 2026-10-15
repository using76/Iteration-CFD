// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! DST-13's gates: the char stage of the Phase B parcel (SPEC-LIT §159.5,
//! §159.8) on the host and the device - the d^2 law and the linear
//! shrinking of (159.6), the limits of (159.5)'s harmonic combination, the
//! carbon and oxygen ledger, the heat-retention split h_s, the device step
//! against its host twin, and the refusals. The same case file runs as the
//! unit tests `dust::char::tests`, so both builds run it.
//!
//! An integration test, so it links the build's library: under `single`
//! the device state is float.
//! The specification was informed by a review of FSCDustFoam (Huang &
//! Lipatnikov, GPL) and no code was copied.
//! No GPL-licensed source was consulted.

mod char_stage {
    include!("shared/dust_char_cases.rs");
}
