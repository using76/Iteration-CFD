// meteor-cfd - Copyright (c) 2026 주식회사 메테오시뮬레이션 (Meteo Simulation Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

//! ofgpu - GPU-native finite-volume CFD, resident on the device.
//!
//! The host side is Rust; the kernels are CUDA C++ compiled to CUBIN by
//! `build.rs` and loaded through `cudarc`. The line is drawn there because
//! Rust buys nothing inside a kernel body - indexing a raw device pointer is
//! `unsafe` in either language - while everything around it (memory ownership,
//! stream and module lifetimes, the OpenFOAM parser, the mesh, the model
//! orchestration) is where the memory bugs actually live. See `../README.md`.
//!
//! Layout:
//!
//! * [`types`], [`device`], [`error`] - primitives, the GPU handle, errors
//! * [`mesh`] - `HostMesh` / `GpuMesh` and the cell -> face CSR
//! * [`field`], [`ldu`] - fields with their universal mixed BC, and the
//!   `fvScalarMatrix` equivalent
//! * [`io`] - the OpenFOAM ASCII reader/writer and case setup
//! * [`potential_flow`] - solves `laplacian(Phi) = 0` for a flux that is
//!   discretely conservative, for cases with no `phi` on disk
//! * [`pressure`] - the pressure Poisson equation's interchangeable linear
//!   backends (PBiCGStab, a direct cuFFT solve, AMGX) and the selector that
//!   measures them against each other before choosing
//! * [`sources`] - volumetric source terms over a geometrically selected cell
//!   set: heat release, body forces, Darcy-Forchheimer porous drag
//! * [`species`] - N-1 transported mass fractions with the inert one closed by
//!   `1 - sum`, all advected by the one conservative flux
//! * [`blockgen`] - a structured mesh generator, so test cases need no OpenFOAM
//! * [`reference`] - an independent CPU transcription used only to validate

pub mod device;
pub mod error;
pub mod types;

pub mod field;
pub mod field_ops;
pub mod field_setup;
pub mod fv;
pub mod ldu;
pub mod ldu_ops;
pub mod les;
pub mod mesh;
pub mod momentum;
pub mod models;
pub mod potential_flow;
pub mod precon;
pub mod pressure;
pub mod scalar_transport;
pub mod simple;
pub mod solver;
pub mod sources;
pub mod species;
pub mod timescheme;
pub mod turbulence;
pub mod vof;
pub mod walldistance;
pub mod wallfunctions;

pub mod blockgen;
pub mod io;
pub mod reference;

pub use device::{cfg_for, DevBuf, Gpu, Graph, KernelSet, BLOCK};
pub use error::{Error, Result};
pub use field::{BcKind, GpuScalarField, GpuSurfaceScalarField, GpuVectorField};
pub use ldu::{CsrPattern, GpuCsrMatrix, GpuLduMatrix};
pub use mesh::{GpuMesh, HostMesh, MeshReport, PatchInfo, PatchKind};
pub use types::{Tensor, Vec3};

/// Compiled device code for every kernel unit, embedded at build time.
///
/// CUBIN rather than PTX - see the note in `build.rs`.
pub mod kernels {
    include!(concat!(env!("OUT_DIR"), "/kernels.rs"));
}

/// The scalar type the whole stack solves in. Mirrors `ofscalar` in
/// `cuda/ofgpu_device.cuh`; both are switched by the `single` feature, and
/// `types::tests::layout_matches_device` pins them together.
#[cfg(feature = "single")]
pub type Scalar = f32;
#[cfg(not(feature = "single"))]
pub type Scalar = f64;

/// A mesh index. `i32` matches what the ASCII case format carries and
/// is what the kernels index with.
pub type Label = i32;
