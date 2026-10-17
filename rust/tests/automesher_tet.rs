// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! TET-12's gates (SPEC-LIT §116.16, §116.18) as an integration test, so
//! the `--features single` build runs them too: the library's own test
//! crate does not compile under `single` (SPEC-LIT §112.3). The cases are
//! tests/shared/automesher_tet_cases.rs, which src/automesher/driver.rs
//! includes as well.
//!
//! Provenance: ORIGINAL. No GPL-licensed source was consulted.

include!("shared/automesher_tet_cases.rs");
