// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! DST-10's gate table, run in BOTH builds: the tests of
//! `tests/shared/compressible_dust_cases.rs` are SPEC-LIT §158.6's rows for
//! the Phase A dust model in the compressible solver. The same file is
//! included by `src/compressible/tests.rs` (module `dust`), the lib tests of
//! the module root.
//!
//! The specification was informed by a review of FSCDustFoam (Huang &
//! Lipatnikov, GPL) and no code was copied.
//! No GPL-licensed source was consulted.

include!("shared/compressible_dust_cases.rs");
