// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! The pressure-based all-speed compressible solver of SPEC-LIT §122 to
//! §127 (`ofgpu-compressible`): one segregated `(U, p, T)` solver in a
//! self-contained module, on the `vof.rs` precedent, with its kernels in a
//! `cuda/compressible.cu` of its own.
//!
//! This unit ships only [`exact`], the host answer keys of SPEC-LIT §127.2.
//! Later units add these submodules to the module:
//!
//! - `thermo` (§122.1-§122.3), the perfect-gas state, CMP-08;
//! - `momentum` (§123), momentum in conservative form, CMP-09;
//! - `pressure` (§124), the all-speed pressure equation, CMP-10;
//! - `energy` (§125), the energy equation solved for `T`, CMP-11;
//! - `bc` (§126), the compressible boundary conditions, CMP-16 to CMP-18;
//! - `lts` (§125), the local time step of a steady run, CMP-13;
//! - `exact` (§127.2), the analytic answer keys, CMP-06, with
//!   `exact/riemann.rs`, Ringleb flow (§127.3) and compressible Couette
//!   flow (§127.4) to come with CMP-07;
//! - `gatemesh` (§127.5), the gate meshes;
//! - `tests`, the module's integration tests.
//!
//! Provenance: ORIGINAL. The skeleton carries no numerics: it declares
//! [`exact`] and names the submodules the later units add. No GPL-licensed
//! source was consulted.

pub mod exact;
