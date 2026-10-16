// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! CMP-25's gate, run in BOTH builds: the three tests of
//! `tests/shared/sa_conservative_cases.rs` are SPEC-LIT §127.1/§127.1.1's
//! rows, verbatim, for the conservative rho-weighted Spalart-Allmaras form.
//! The same file is included by the lib tests of
//! `src/models/spalart_allmaras/tests.rs`, under `mod conservative`.

include!("shared/sa_conservative_cases.rs");
