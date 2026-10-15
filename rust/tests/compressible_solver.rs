// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! CMP-12's gate table, run in BOTH builds: the four tests of
//! `tests/shared/compressible_solver_cases.rs` are SPEC-LIT §125.6's rows
//! for the outer loop of the compressible solver. The same file is included
//! by `src/compressible/tests.rs`, the lib tests of the module root.

include!("shared/compressible_solver_cases.rs");
