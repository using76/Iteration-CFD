// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! SPEC-LIT §157.4 (DST-06 part 1): the `combustion` FSC/TFC keys and the
//! `dust` case block lower in BOTH builds.
//!
//! The library tests `io::case_dust::tests` hold every key, every default
//! and every refusal in the f64 build; the single lib test crate does not
//! compile, so this file is the single build's arm (class D). Host only: no
//! card is needed. No GPL-licensed source was consulted.

include!("shared/case_dust_cases.rs");
