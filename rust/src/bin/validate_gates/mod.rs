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

/// SPEC-LIT §144.5, Gate 132-Z: the TMR zero-pressure-gradient plate with SA
/// and SST against Karman-Schoenherr and Coles (recorded runs, replayed).
pub(crate) mod zpg;

/// SPEC-LIT §143.5, Gate 131-P: TG0's turbulent pipe with kOmegaSST on the
/// true wedge against its four bands (recorded ofgpu-lowmach runs, replayed).
pub(crate) mod pipe;

/// SPEC-LIT §110.2/§110.5, Gate 110-A: the plane channel's recorded runs at
/// three `Re_tau` (data replayed by `validate.rs`'s channel gate).
pub(crate) mod channel_dns;

/// SPEC-LIT §110.4/§110.5, Gate 110-C: the hot-gas plume's recorded runs at
/// three levels (data replayed by `validate.rs`'s plume gate).
pub(crate) mod plume;

/// SPEC-LIT §110.3/§110.5, Gate 110-B: the backward-facing step's recorded
/// runs at three `dx/H` levels and three single builds (data replayed by
/// `validate.rs`'s step gate).
pub(crate) mod backstep;

/// SPEC-LIT §127.6, Gate CMP-SOD: Sod's shock tube on three meshes against the
/// exact Riemann solution (CMP-19).
pub(crate) mod sod;

/// SPEC-LIT §127.6, Gate CMP-AC: a small acoustic pulse's speed against
/// (122.4)'s `c_0` (CMP-20).
pub(crate) mod acoustic;

/// SPEC-LIT §127.6, Gate CMP-LM: the lid-driven cavity at lid Mach 0.05
/// against Ghia, Ghia & Shin (1982) with Gate 94-D's tolerance (CMP-20).
pub(crate) mod cavity_lm;

/// SPEC-LIT §127.6, Gate CMP-COU: compressible Couette with viscous
/// heating against (127.12) (CMP-22).
pub(crate) mod couette;

/// SPEC-LIT §127.6, Gate CMP-NOZ: the quasi-1-D nozzle, subsonic,
/// supersonic started and shocked, against (127.23)-(127.25) (CMP-21).
pub(crate) mod nozzle;
