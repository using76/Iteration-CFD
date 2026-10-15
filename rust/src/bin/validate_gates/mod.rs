// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

//! `ofgpu-validate`'s gates in files of their own (docs/17 section 5.3 rule 2,
//! VAL decision D10): one file per gate, each reached from `validate.rs`
//! through one `c.section` / `enter_gate` call appended to its dispatch block.
//! Each file is a child of the binary's root, so it uses the root's `Checks`,
//! `GateReport` and answer-key reader as they are, and adds no second registry.
//!
//! Written from:
//!   ofgpu `SPEC-LIT.md` §69 (the gate registry), §94 (the mesh study) and
//!     the section each gate file names
//! No GPL-licensed source was consulted.

/// SPEC-LIT §145.2, Gate 145-A: the dam break against Martin & Moyce (1952).
pub(crate) mod martin_moyce;

/// SPEC-LIT §146.2, Gate 78-D's measured leg: the two splash criteria of §78.4
/// against Riboux & Gordillo's (2014) threshold speeds.
pub(crate) mod splash;

/// SPEC-LIT §144.2, Gate 132-L: the laminar flat plate against Blasius.
pub(crate) mod blasius_plate;

/// SPEC-LIT §144.3: the T3A plate on the flat-plate harness, leg 1 in 2-D
/// (recorded runs, replayed).
pub(crate) mod t3a;
