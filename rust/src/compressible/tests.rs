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

/// SPEC-LIT 112.1: `cuda/compressible.cu` defines no floor - its kernels
/// state "no floor and no clip" (122.3, 122.5), and a `double`-only literal
/// in the file would be one, hiding in the one place the `single` build does
/// not re-type. The constants arrive already formed by (122.9); nothing here
/// may need a double's range to say so.
#[test]
fn the_compressible_kernels_carry_no_double_only_literal() {
    let code = crate::models::spalart_allmaras::tests::strip_c_comments(include_str!(
        "../../cuda/compressible.cu"
    ));
    for literal in [
        "1e-300",
        "1e300",
        "1e+300",
        "2.2250738585072014e-308",
        "DBL_MIN",
        "DBL_MAX",
    ] {
        assert!(
            !code.contains(literal),
            "cuda/compressible.cu states {literal}: SPEC-LIT 112.1 gives the file no \
             floor to state, and a double-only literal would not survive the single build"
        );
    }
}
