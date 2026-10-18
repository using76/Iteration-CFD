// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! The flame-age rows of SPEC-LIT §160.5 as an integration test, so they
//! link the build's library: under `single` the equations are solved in
//! float. The same case file runs as the unit tests `dust::age::tests`, so
//! both builds run it.
//!
//! The specification was informed by a review of FSCDustFoam (Huang &
//! Lipatnikov, GPL) and no code was copied.
//! No GPL-licensed source was consulted.

mod age {
    include!("shared/dust_age_cases.rs");
}
