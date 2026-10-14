// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! DST-05's gates: the rho-weighted k-omega SST-2003 and SST-sust (SPEC-LIT
//! §156) - the rho == 1 row bit for bit against the kinematic SST, the
//! homogeneous-decay closed form (156.11), the sustained ambient state, the
//! `∫rho k dV` budget, both dilatation terms of the production (156.3) and
//! the `limitOmegaProduction` setting (156.8). The same case file runs as the
//! unit tests `k_omega_sst::tests`, so both builds run it.
//!
//! An integration test, so it links the build's library: under `single` the
//! fields are float and class R is 1e-5. Written from SPEC-LIT §156 (the
//! NASA Turbulence Modeling Resource SST page). The specification was
//! informed by a review of FSCDustFoam (Huang & Lipatnikov, GPL) and no code
//! was copied.
//! No GPL-licensed source was consulted.

mod sst {
    include!("shared/sst_rho_cases.rs");
}
