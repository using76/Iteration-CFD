// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! DST-15 part 1's gates: the cloud's library of SPEC-LIT §159.10 - the
//! volatile record (159.16) as the starch times 50, the mechanism file's
//! read-back, the stage datums and the particle enthalpy (159.18), the
//! gather's telescoping ledgers (159.19), the sources' temperature form
//! (159.20) and the cloud's AICC (159.21). The same case file runs as the
//! unit tests `dust::cloud::tests`, so both builds run it.
//!
//! An integration test, so it links the build's library; everything here
//! is host f64 in both builds.
//! The specification was informed by a review of FSCDustFoam (Huang &
//! Lipatnikov, GPL) and no code was copied.
//! No GPL-licensed source was consulted.

mod cloud_lib {
    include!("shared/dust_cloud_cases.rs");
}
