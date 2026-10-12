// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! Native aggregation multigrid (SPEC-LIT §141): AMG-01 contributes the
//! deterministic double pairwise aggregation of one LDU level. AMG-02..04a
//! add the Galerkin operator, the smoothers and the cycles on top of it.
//! AMG-02 (§141.2) contributes the Galerkin coarse operator and the
//! transfers, as gather lists with a host twin and four device kernels.
//!
//! Provenance: ORIGINAL - written from Notay, ETNA 37 (2010) 123-146,
//! Algorithms 2.1 and 2.2 (SPEC-LIT §141.1). No GPL-licensed source was
//! consulted.

pub mod aggregate;
pub mod galerkin;

pub use aggregate::{
    aggregate, aggregate_level, Aggregation, LevelView, BETA, INTERIOR, UNCOUPLED,
};
pub use galerkin::{
    GalerkinPlan, GpuGalerkin, GpuLevel, HostLevel, KIND_BOUNDARY, KIND_DIAG, KIND_LOWER,
    KIND_UPPER, MAX_INDEX,
};
