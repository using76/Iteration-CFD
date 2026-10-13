// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! CMP-10's gate table, run in BOTH builds: the four tests of
//! `tests/shared/compressible_pressure_cases.rs` are SPEC-LIT §124.4's
//! rows, verbatim, for the all-speed pressure equation of the compressible
//! solver. The same file is included by the lib tests of
//! `src/compressible/pressure.rs`, where the capture gate sits beside them.

include!("shared/compressible_pressure_cases.rs");
