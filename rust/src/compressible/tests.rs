// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! The lib tests of the module root: the outer loop's gate table of
//! SPEC-LIT §125.6 (CMP-12), written once in
//! `tests/shared/compressible_solver_cases.rs` and included here and by
//! `tests/compressible_solver.rs`, which runs the same tests in both builds.
//!
//! Provenance: ORIGINAL. No GPL-licensed source was consulted.

use crate as ofgpu;

include!("../../tests/shared/compressible_solver_cases.rs");

/// DST-10's rows of SPEC-LIT §158.6 (the dust model in this solver),
/// written once in `tests/shared/compressible_dust_cases.rs` and run in both
/// builds by `tests/compressible_dust.rs`.
mod dust {
    use crate as ofgpu;

    include!("../../tests/shared/compressible_dust_cases.rs");
}

/// SPEC-LIT §127.1.2's gate: the SA coupling of the module root, written
/// once in `tests/shared/compressible_sa_cases.rs` and included here and by
/// `tests/compressible_sa.rs`, which runs the same tests in both builds.
mod sa {
    use crate as ofgpu;

    include!("../../tests/shared/compressible_sa_cases.rs");
}
