// meteor-cfd - Copyright (c) 2026 주식회사 메테오시뮬레이션 (Meteo Simulation Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

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
//! * [`s2s`] - surface-to-surface radiation: deterministic view factors, the
//!   enclosure radiosity system, and the one rewritten Robin triple that is
//!   the whole of its contact with the finite-volume solver
//! * [`cht`] - conjugate heat transfer: solid regions, anisotropic
//!   conduction, contact resistance, and the concatenated thermal mesh whose
//!   fluid/solid interface is a cyclic couple with a zero transform
//! * [`blockgen`] - a structured mesh generator, so test cases need no OpenFOAM
//! * [`reference`] - an independent CPU transcription used only to validate
//!
//! Provenance: ORIGINAL - the crate root. Module declarations, the crate-wide
//! `Scalar` alias and the re-exports; it carries no numerics of its own, and
//! each module named above declares its own provenance in its own header.
//! `PROVENANCE.md` is the per-file record for the whole tree. No GPL-licensed
//! source was consulted.

pub mod adapt;
pub mod decompose;
pub mod device;
pub mod distsolve;
pub mod error;
pub mod exactsum;
pub mod types;

pub mod field;
pub mod field_ops;
pub mod field_setup;
pub mod fv;
pub mod halo;
pub mod ldu;
pub mod ldu_ops;
pub mod les;
pub mod mesh;
pub mod momentum;
pub mod models;
pub mod parcels;
pub mod potential_flow;
pub mod restart;
pub mod rheology;
pub mod cht;
pub mod dcmetrics;
pub mod fan;
pub mod psychro;
pub mod combustion;
pub mod twostep;
pub mod contact_angle;
pub mod energy;
pub mod fvdom;
pub mod radiation;
pub mod s2s;
pub mod soot;
pub mod wsgg;
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
pub mod surface;

/// The citation audit of SPEC-LIT §80: every `§NN.M`, `SNN.M` and `(NN.M)` in
/// a comment, doc comment or error message must name something that exists in
/// the document the citation belongs to. Test-only, so the release build is
/// unchanged by construction.
#[cfg(test)]
mod xref;

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

/// The provenance surface, checked by test rather than asserted in prose.
///
/// `NOTICE` and `PROVENANCE.md` both claim that EVERY source file under
/// `rust/` carries the copyright header and the line "No GPL-licensed source
/// was consulted", and both quote a file count. Those were manual claims, and
/// a manual claim drifts: at the time this module was written 85 of 105 files
/// carried the line while both documents said 105. An acquirer auditing file
/// by file would have found the gap before we did.
///
/// So the claim is now a test. It walks the same four roots the documents
/// name (`src/`, `cuda/`, `tests/`, `build.rs`), and it reads the count back
/// out of `NOTICE` and `PROVENANCE.md` so that adding a file without updating
/// them fails here rather than in an audit.
#[cfg(test)]
mod provenance_audit {
    use std::fs;
    use std::path::{Path, PathBuf};

    const GPL_LINE: &str = "No GPL-licensed source was consulted";
    const COPYRIGHT: &str = "Meteo Simulation Co., Ltd.";

    fn root() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).to_path_buf()
    }

    /// Every `.rs`/`.cu`/`.cuh` under the roots `NOTICE` names, sorted.
    fn sources() -> Vec<PathBuf> {
        fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
            let Ok(rd) = fs::read_dir(dir) else { return };
            for e in rd.flatten() {
                let p = e.path();
                if p.is_dir() {
                    walk(&p, out);
                } else if matches!(
                    p.extension().and_then(|s| s.to_str()),
                    Some("rs") | Some("cu") | Some("cuh")
                ) {
                    out.push(p);
                }
            }
        }
        let mut out = Vec::new();
        for d in ["src", "cuda", "tests"] {
            walk(&root().join(d), &mut out);
        }
        out.push(root().join("build.rs"));
        out.sort();
        out
    }

    fn rel(p: &Path) -> String {
        let s = p.strip_prefix(root()).unwrap_or(p).display().to_string();
        s.replace(std::path::MAIN_SEPARATOR, "/")
    }

    #[test]
    fn every_source_file_declares_its_provenance() {
        let mut missing_gpl = Vec::new();
        let mut missing_copyright = Vec::new();
        for p in sources() {
            let t = fs::read_to_string(&p).unwrap_or_default();
            if !t.contains(GPL_LINE) {
                missing_gpl.push(rel(&p));
            }
            if !t.contains(COPYRIGHT) {
                missing_copyright.push(rel(&p));
            }
        }
        assert!(
            missing_gpl.is_empty(),
            "{} source file(s) do not carry \"{GPL_LINE}\", which NOTICE and \
             PROVENANCE.md both claim every file carries:\n  {}",
            missing_gpl.len(),
            missing_gpl.join("\n  ")
        );
        assert!(
            missing_copyright.is_empty(),
            "{} source file(s) do not carry the copyright header:\n  {}",
            missing_copyright.len(),
            missing_copyright.join("\n  ")
        );
    }

    /// The COUNT both documents publish has to be the count on disk. This is
    /// what stops the next added file from making the published number stale.
    #[test]
    fn notice_and_provenance_quote_the_real_file_count() {
        let n = sources().len();
        for (name, path) in [
            ("NOTICE", root().join("../NOTICE")),
            ("PROVENANCE.md", root().join("PROVENANCE.md")),
        ] {
            let t = fs::read_to_string(&path)
                .unwrap_or_else(|e| panic!("cannot read {name}: {e}"));
            assert!(
                t.contains(&format!("{n} source files"))
                    || t.contains(&format!("{n}/{n}")),
                "{name} does not quote the real source-file count {n}; update it \
                 (it is the number an acquirer checks first)"
            );
        }
    }
}
