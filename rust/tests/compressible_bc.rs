// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! CMP-16's, CMP-17's, CMP-18's and BC-20's gate tables, run in BOTH builds: the
//! tests of `tests/shared/compressible_bc_cases.rs` are SPEC-LIT §126.8's
//! rows for the compressible inlets, §126.9's for the outlets (Gate CMP-NR
//! among them), §126.10's for the characteristic far-field and §126.11's
//! for the free-stream family. The same file is included by `src/compressible/bc.rs`, the lib
//! tests of the module.

include!("shared/compressible_bc_cases.rs");
