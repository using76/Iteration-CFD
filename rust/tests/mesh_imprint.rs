// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! BC-18's gate, run in BOTH builds: the nine tests of
//! `tests/shared/mesh_imprint_cases.rs` are SPEC-LIT §120.18's gate,
//! verbatim, for the static non-conformal interface imprint of
//! `mesh::imprint` (matching grids, the 3:2 counts, the 7-degree rotated
//! grid, the area sums, the one-to-one pairing, per-cell closure, the
//! moved pair and the refusals). The same file is included by the lib
//! tests of `src/mesh/imprint.rs`, where the gate sits beside the code.

include!("shared/mesh_imprint_cases.rs");
