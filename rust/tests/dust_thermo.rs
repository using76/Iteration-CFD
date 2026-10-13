// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! DST-01's gates: the equivalent premixed gas of the corn-starch dust
//! (SPEC-LIT §152) — the thermo of `src/dust/thermo.rs` and the laminar
//! burning velocity of `src/dust/laminar.rs` — against the oracle values of
//! §152.10 and the printed values of H22 and H20. The same case files run as
//! the unit tests `dust::thermo::tests` and `dust::laminar::tests`, so both
//! builds run them.
//!
//! An integration test, so it links the normal library, which is f64 in
//! every build.
//! No GPL-licensed source was consulted.

mod thermo {
    include!("shared/dust_thermo_cases.rs");
}

mod laminar {
    include!("shared/dust_laminar_cases.rs");
}
