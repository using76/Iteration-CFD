// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! DST-09's gate, run in BOTH builds: the tests of
//! `tests/shared/dust_vent_cases.rs` are SPEC-LIT §158.5's rows for the
//! one-domain vent - the closed membrane as a wall, the membrane open at
//! t = 0, Sod through an opening membrane, the trigger, the split mesh, the
//! hand-over and the refusals of §158.8. The same file is included by
//! `src/dust/vent.rs`, the lib tests of the module.

include!("shared/dust_vent_cases.rs");
