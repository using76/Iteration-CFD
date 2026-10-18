// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! DST-08 part 1's gates: the host thin-flame vessel of `src/dust/vessel0d.rs`
//! (SPEC-LIT §157.6) - the vessel from the case's own thermo, the (157.2)-
//! (157.3) end points, the (157.4) integration's time-to-half against the
//! quadrature and its fourth order, the (157.5) cube-root law on two volumes,
//! the negative control's miss, and every refusal naming its input. The
//! library tests `dust::vessel0d::tests` run the same cases in the f64 build;
//! this file is both builds' arm. No GPU: the vessel is host `f64` in both.
//! The specification was informed by a review of FSCDustFoam (Huang &
//! Lipatnikov, GPL) and no code was copied.
//! No GPL-licensed source was consulted.

include!("shared/dust_vessel0d_cases.rs");
