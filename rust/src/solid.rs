// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! Solid mechanics - the thermo-elastic solid.
//!
//! Written from: nothing yet. This file is **ORIGINAL** - it is the module
//! root and carries no numerics of its own; every child declares its own
//! provenance in its own header. No GPL-licensed source was consulted.
//!
//! # What is here today, and what is not
//!
//! Exactly one child, [`prototype`], and it is not part of any solver. It is
//! the **risk-1 experiment** of `docs/09-thermal-structural-plan.md` §F: the
//! segregated displacement outer loop written host-side, before a kernel
//! exists, so that the contraction the plan *derives* -
//! `|deferred|/|implicit| = (mu + lambda)/(2 mu + lambda) = 1/(2(1 - nu))` -
//! is **measured** rather than asserted. SPEC-LIT §46.4 is the precedent that
//! made that the order of work: an estimate written into a specification,
//! corrected afterwards by the test that measured it.
//!
//! The displacement equation itself has **no SPEC-LIT section yet**. The plan
//! reserves the ninety-fifth for it, and nothing in this module may cite that
//! number until the section exists - §80's audit fails the build on a
//! citation whose address is vacant, and the whole point of §0 rule 6 is that
//! an address is either occupied or it is not. What the prototype cites are the sections it
//! genuinely reuses: §1's LDU storage, §2.4's over-relaxed non-orthogonal
//! correction, §3.2's Gauss laplacian, §3.5's Green-Gauss gradient, §4's one
//! mixed boundary triple, §8.2's conjugate gradients and §8.4's residual
//! normalisation.

pub mod prototype;
