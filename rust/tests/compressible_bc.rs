// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! CMP-16's gate table, run in BOTH builds: the five tests of
//! `tests/shared/compressible_bc_cases.rs` are SPEC-LIT §126.8's rows for
//! the compressible inlets. The same file is included by
//! `src/compressible/bc.rs`, the lib tests of the module.

include!("shared/compressible_bc_cases.rs");
