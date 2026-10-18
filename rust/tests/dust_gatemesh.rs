// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! DST-07's sector-mesh gate, run in BOTH builds: the five tests of
//! `tests/shared/dust_gatemesh_cases.rs` are SPEC-LIT §157.3's contract,
//! verbatim - the radii, the closure and r² face areas, the exact frusta on
//! the host mesh, the Leeds case lowering, and the refusals. The same file
//! is included by the lib tests of `src/dust/gatemesh.rs`, where the gate
//! sits beside the code.

include!("shared/dust_gatemesh_cases.rs");
