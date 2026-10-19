// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! DST-12's gates: the reacting dust parcel of SPEC-LIT §159 on the host
//! and the device - isothermal devolatilisation and the lumped heat-up
//! against (159.6), Bird's limits (159.3), the stage order moisture ->
//! volatiles -> char, and the device step against its host twin (§159.5).
//! The same case file runs as the unit tests `dust::particle::tests`, so
//! both builds run it.
//! DST-18 adds the particle-radiation rows of SPEC-LIT 159.13: a cloud in
//! P1, the uniform-field heating, the energy closure and the off switch.
//!
//! An integration test, so it links the build's library: under `single`
//! the device state is float.
//! The specification was informed by a review of FSCDustFoam (Huang &
//! Lipatnikov, GPL) and no code was copied.
//! No GPL-licensed source was consulted.

mod particle {
    include!("shared/dust_particle_cases.rs");
}
