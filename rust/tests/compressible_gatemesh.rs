// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! CMP-26's gate table, run in BOTH builds: the ten tests of
//! `tests/shared/compressible_gatemesh_cases.rs` are SPEC-LIT §127.5.1's
//! gate, verbatim, for the RAE 2822 airfoil C-mesh of the compressible
//! unit. The same file is included by the lib tests of
//! `src/compressible/gatemesh.rs`, where the gate sits beside the code.

include!("shared/compressible_gatemesh_cases.rs");
