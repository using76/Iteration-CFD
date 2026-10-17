// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! CMP-25's coupling gate, run in BOTH builds: the three tests of
//! `tests/shared/compressible_sa_cases.rs` are SPEC-LIT §127.1.2's rows for
//! the Spalart-Allmaras coupling on the compressible solver. The same file
//! is included by the lib tests of `src/compressible/tests.rs`, where the
//! names sit under `mod sa`.

include!("shared/compressible_sa_cases.rs");
