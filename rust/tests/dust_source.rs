// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! DST-06 part 2's gates: one dust step of `src/dust/source.rs` (SPEC-LIT
//! §157.4) - the three thermo kernels of `cuda/dust.cu` against the host
//! formulas, the setup's constants against §152.11, one laminar step of a
//! box against its closed-form answer, and the predicted flame-Courant
//! refusal. The library tests `dust::source::tests` run the same cases in
//! the f64 build; this file is both builds' arm (class R, D-F32-1). A
//! machine without a card passes vacuously.
//! The specification was informed by a review of FSCDustFoam (Huang &
//! Lipatnikov, GPL) and no code was copied.
//! No GPL-licensed source was consulted.

include!("shared/dust_source_cases.rs");
