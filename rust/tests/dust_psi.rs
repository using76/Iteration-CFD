// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! DST-02's gates: the laminar-limit eigenproblem of the dust flame
//! (SPEC-LIT §153) — Psi regenerated from its definition by
//! `src/dust/psi.rs`, for the published closure (153.1) and the case's own
//! adiabatic line — against the oracle values of §153.5-§153.6. The same
//! case files run as the unit tests `dust::psi::tests`, so both builds run
//! them.
//!
//! An integration test, so it links the normal library, which is f64 in
//! every build.
//! No GPL-licensed source was consulted.

mod psi {
    include!("shared/dust_psi_cases.rs");
}
