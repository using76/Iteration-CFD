// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! CMP-11's gate table, run in BOTH builds: the five tests of
//! `tests/shared/compressible_energy_cases.rs` are SPEC-LIT §125.5's rows
//! for the energy equation of the compressible solver. The same file is
//! included by the lib tests of `src/compressible/energy.rs`, where the
//! capture gate sits beside them.

include!("shared/compressible_energy_cases.rs");
