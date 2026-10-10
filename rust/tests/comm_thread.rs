// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! SPEC-LIT §138.1.5 — the Comm tests in BOTH builds.
//!
//! An integration test because the library's own test target does not
//! compile under `--features single`; an integration test links the normal
//! (non-test) library, which does. This file copies nothing: it includes
//! `src/comm/tests.rs` itself, so the SAME T0-T14 run in the default build
//! and under `single` with the same assertions — class D of docs/17 §5.1:
//! exact bytes and exact counts. Host only, no GPU.
//!
//! This file supplies shims only, no logic: the names `src/comm/tests.rs`
//! reaches for at its crate root (`use super::*;` finds `Comm`,
//! `ThreadComm`, `MAX_TAG`, `DEFAULT_TIMEOUT` from `ofgpu::comm`;
//! `crate::error::Error` is `ofgpu::Error`; `crate::Result` is
//! `ofgpu::Result`). The tests' generators are SplitMix64 (Steele, Lea &
//! Flood 2014, DOI 10.1145/2660193.2660195) and FNV-1a 64, both public
//! domain.

use ofgpu::comm::{Comm, ThreadComm, MAX_TAG};
use ofgpu::Result;

// The included file's header names this in its contract even though no test
// references it by name; supplied so `use super::*;` finds it.
#[allow(unused_imports)]
use ofgpu::comm::DEFAULT_TIMEOUT;

/// `src/comm/tests.rs`'s `crate::error::Error` is `ofgpu::Error` here.
mod error {
    pub use ofgpu::Error;
}

#[path = "../src/comm/tests.rs"]
mod tests;
