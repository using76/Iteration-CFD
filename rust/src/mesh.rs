// meteor-cfd - Copyright (c) 2026 주식회사 메테오시뮬레이션 (Meteo Simulation Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

//! The mesh, on the host and on the device.
//!
//! A finite-volume mesh is a face-based graph: two parallel arrays `owner`
//! and `neighbour` naming the cells either side of each internal face, plus
//! geometry. That is already the right shape for a GPU - flat, static, and for
//! a fixed mesh uploaded exactly once.
//!
//! Two things are added on top:
//!
//! 1. A **cell -> face CSR** (`cf_offset`/`cf_face`/`cf_own`). The obvious way
//!    to assemble a diagonal is to *scatter* over faces
//!    (`diag[owner[f]] -= ...`), which on a GPU needs atomics on `f64` and
//!    gives non-deterministic rounding. Inverting the map lets every kernel
//!    *gather* instead: one thread per cell, no atomics, bitwise reproducible.
//!
//! 2. Boundary faces of every patch flattened into one contiguous range with a
//!    patch offset table, so one kernel covers all patches.
//!
//! This file defines the types. The geometry itself is in `mesh/geometry.rs`.

use crate::device::{DevBuf, Gpu};
use crate::error::Result;
use crate::{Label, Scalar, Vec3};

/// What a patch *is*, topologically. The physical boundary condition applied
/// to a given field is stored per face in `field.rs`; this only describes the
/// patch's geometric role.
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PatchKind {
    /// Ordinary patch; whatever the field's BC says goes.
    Generic = 0,
    /// Wall: wall functions may target it.
    Wall = 1,
    /// 2-D front/back. Contributes nothing to any surface integral.
    Empty = 2,
    /// symmetry / symmetryPlane / slip.
    Symmetry = 3,
    /// Coupled to another patch on the same GPU.
    Cyclic = 4,
    /// Coupled across MPI. Unused in this single-GPU build.
    Processor = 5,
}

impl PatchKind {
    /// Map an OpenFOAM `type` string from `constant/polyMesh/boundary`.
    pub fn from_type(t: &str) -> Self {
        match t {
            "wall" | "mappedWall" => Self::Wall,
            "empty" => Self::Empty,
            "symmetry" | "symmetryPlane" | "wedge" => Self::Symmetry,
            "cyclic" | "cyclicAMI" | "cyclicSlip" => Self::Cyclic,
            "processor" | "processorCyclic" => Self::Processor,
            _ => Self::Generic,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Generic => "generic",
            Self::Wall => "wall",
            Self::Empty => "empty",
            Self::Symmetry => "symmetry",
            Self::Cyclic => "cyclic",
            Self::Processor => "processor",
        }
    }
}

#[derive(Debug, Clone)]
pub struct PatchInfo {
    pub name: String,
    /// Raw OpenFOAM patch type string, kept verbatim for round-tripping.
    pub type_name: String,
    pub kind: PatchKind,
    /// Offset into the FLATTENED boundary-face arrays - not the global face
    /// index. `startFace - nInternalFaces` in OpenFOAM terms.
    pub start: usize,
    pub size: usize,
    /// For a cyclic patch, the index of the patch it couples to.
    pub nbr_patch: Option<usize>,
}

/// The mesh on the host, after reading polyMesh and computing the
/// finite-volume geometry. Exists only during setup and for the CPU reference.
#[derive(Debug, Default, Clone)]
pub struct HostMesh {
    pub n_cells: usize,
    pub n_internal_faces: usize,
    pub n_boundary_faces: usize,
    pub n_points: usize,

    // ---- lduAddressing ----------------------------------------------------
    /// `[n_internal_faces]`, equals `lduAddr().lowerAddr()`
    pub owner: Vec<Label>,
    /// `[n_internal_faces]`, equals `lduAddr().upperAddr()`
    pub neighbour: Vec<Label>,

    // ---- cell geometry ----------------------------------------------------
    /// `[n_cells]` cell volumes
    pub v: Vec<Scalar>,
    /// `[n_cells]` cell centres
    pub c: Vec<Vec3>,

    // ---- internal face geometry ------------------------------------------
    /// `[n_if]` outward area vector, owner -> neighbour
    pub sf: Vec<Vec3>,
    pub mag_sf: Vec<Scalar>,
    pub cf: Vec<Vec3>,
    /// `[n_if]` linear interpolation weight of the OWNER
    pub weights: Vec<Scalar>,
    /// `[n_if]` NON-ORTHOGONAL delta coefficients - what the laplacian and
    /// snGrad use. Coincides with `1/|d|` on an orthogonal mesh.
    pub delta_coeffs: Vec<Scalar>,
    pub non_orth_corr: Vec<Vec3>,

    // ---- boundary faces, all patches flattened ---------------------------
    pub b_face_cells: Vec<Label>,
    pub b_sf: Vec<Vec3>,
    pub b_mag_sf: Vec<Scalar>,
    pub b_cf: Vec<Vec3>,
    pub b_delta_coeffs: Vec<Scalar>,
    /// `[n_bf]` wall-normal distance of the adjacent cell centre, `nf . delta`
    pub b_y: Vec<Scalar>,
    /// `[n_bf]` cyclic: the cell across the couple; `-1` otherwise
    pub b_nbr_cell: Vec<Label>,
    /// `[n_bf]` cyclic: interpolation weight; `1` otherwise
    pub b_weights: Vec<Scalar>,
    /// `[n_bf]` `PatchKind` of the owning patch, as `i32`
    pub b_kind: Vec<Label>,
    /// `[n_bf]` index of the owning patch
    pub b_patch: Vec<Label>,

    pub patches: Vec<PatchInfo>,

    // ---- cell -> face CSR (built, not read) ------------------------------
    /// `[n_cells + 1]`
    pub cf_offset: Vec<Label>,
    /// `[2 * n_internal_faces]` internal face id
    pub cf_face: Vec<Label>,
    /// `[2 * n_internal_faces]` 1 if this cell is the face's owner
    pub cf_own: Vec<Label>,

    // ---- cell -> boundary face CSR ---------------------------------------
    /// `[n_cells + 1]`
    pub bcf_offset: Vec<Label>,
    /// `[n_boundary_faces]`
    pub bcf_face: Vec<Label>,
}

/// What `HostMesh::check` found. Printing it is the first thing every binary
/// does, because a mesh that does not close is not worth solving on.
#[derive(Debug, Clone)]
pub struct MeshReport {
    pub total_volume: Scalar,
    pub min_volume: Scalar,
    pub max_volume: Scalar,
    pub min_volume_cell: usize,
    /// Maximum face non-orthogonality, degrees.
    pub max_non_orth_deg: Scalar,
    pub mean_non_orth_deg: Scalar,
    /// `max |sum_f s*Sf| / V^(2/3)` over cells. A correct mesh closes to
    /// round-off; anything above ~1e-10 means the face winding is wrong.
    pub max_closure_error: Scalar,
    pub max_closure_cell: usize,
    /// `true` when owner < neighbour everywhere and faces are sorted by
    /// (owner, neighbour) - the upper-triangular order the LDU addressing and
    /// every gather kernel assume.
    pub ldu_ordered: bool,
}

impl HostMesh {
    /// Invert `owner`/`neighbour` into the two CSR maps.
    ///
    /// Within each cell the faces are kept in ascending face index so the
    /// gather order is deterministic, which is what makes results bitwise
    /// reproducible across runs.
    pub fn build_cell_face_maps(&mut self) {
        crate::mesh::topology::build_cell_face_maps(self)
    }

    /// Compute `v`, `c`, `sf`, `mag_sf`, `cf`, `weights`, `delta_coeffs`,
    /// `non_orth_corr` and the boundary metrics from raw points and faces.
    /// Mirrors `primitiveMesh` + `surfaceInterpolation`.
    pub fn compute_geometry(
        &mut self,
        points: &[Vec3],
        faces: &[Vec<Label>],
    ) -> Result<()> {
        crate::mesh::geometry::compute(self, points, faces)
    }

    pub fn check(&self) -> MeshReport {
        crate::mesh::geometry::check(self)
    }

    /// Human-readable summary, in the same shape the C++ version printed so
    /// the two can be diffed.
    pub fn print_report(&self) {
        crate::mesh::geometry::print_report(self)
    }

    /// `+1` when `cell` owns `face`, `-1` when it is the neighbour.
    #[inline]
    pub fn face_sign(&self, cell: usize, face: usize) -> Scalar {
        if self.owner[face] as usize == cell {
            1.0
        } else {
            -1.0
        }
    }
}

/// The device-resident mirror. Uploaded once; immutable thereafter.
///
/// Every field is a separate `DevBuf` rather than a packed struct of raw
/// pointers: `cudarc` tracks stream dependencies per buffer when they are
/// passed with `.arg(&buf)`, and hand-rolling a pointer struct would throw
/// that tracking away - which is precisely the safety this port is for.
pub struct GpuMesh {
    pub n_cells: usize,
    pub n_internal_faces: usize,
    pub n_boundary_faces: usize,

    pub owner: DevBuf<Label>,
    pub neighbour: DevBuf<Label>,

    pub v: DevBuf<Scalar>,
    pub c: DevBuf<Vec3>,

    pub sf: DevBuf<Vec3>,
    pub mag_sf: DevBuf<Scalar>,
    pub cf: DevBuf<Vec3>,
    pub weights: DevBuf<Scalar>,
    pub delta_coeffs: DevBuf<Scalar>,
    pub non_orth_corr: DevBuf<Vec3>,

    pub b_face_cells: DevBuf<Label>,
    pub b_sf: DevBuf<Vec3>,
    pub b_mag_sf: DevBuf<Scalar>,
    pub b_cf: DevBuf<Vec3>,
    pub b_delta_coeffs: DevBuf<Scalar>,
    pub b_y: DevBuf<Scalar>,
    pub b_nbr_cell: DevBuf<Label>,
    pub b_weights: DevBuf<Scalar>,
    pub b_kind: DevBuf<Label>,
    pub b_patch: DevBuf<Label>,

    pub cf_offset: DevBuf<Label>,
    pub cf_face: DevBuf<Label>,
    pub cf_own: DevBuf<Label>,
    pub bcf_offset: DevBuf<Label>,
    pub bcf_face: DevBuf<Label>,

    /// Kept on the host for reporting only.
    pub patches: Vec<PatchInfo>,
    pub total_volume: Scalar,
}

impl GpuMesh {
    pub fn upload(gpu: &Gpu, m: &HostMesh) -> Result<Self> {
        Ok(Self {
            n_cells: m.n_cells,
            n_internal_faces: m.n_internal_faces,
            n_boundary_faces: m.n_boundary_faces,

            owner: gpu.upload(&m.owner)?,
            neighbour: gpu.upload(&m.neighbour)?,

            v: gpu.upload(&m.v)?,
            c: gpu.upload(&m.c)?,

            sf: gpu.upload(&m.sf)?,
            mag_sf: gpu.upload(&m.mag_sf)?,
            cf: gpu.upload(&m.cf)?,
            weights: gpu.upload(&m.weights)?,
            delta_coeffs: gpu.upload(&m.delta_coeffs)?,
            non_orth_corr: gpu.upload(&m.non_orth_corr)?,

            b_face_cells: gpu.upload(&m.b_face_cells)?,
            b_sf: gpu.upload(&m.b_sf)?,
            b_mag_sf: gpu.upload(&m.b_mag_sf)?,
            b_cf: gpu.upload(&m.b_cf)?,
            b_delta_coeffs: gpu.upload(&m.b_delta_coeffs)?,
            b_y: gpu.upload(&m.b_y)?,
            b_nbr_cell: gpu.upload(&m.b_nbr_cell)?,
            b_weights: gpu.upload(&m.b_weights)?,
            b_kind: gpu.upload(&m.b_kind)?,
            b_patch: gpu.upload(&m.b_patch)?,

            cf_offset: gpu.upload(&m.cf_offset)?,
            cf_face: gpu.upload(&m.cf_face)?,
            cf_own: gpu.upload(&m.cf_own)?,
            bcf_offset: gpu.upload(&m.bcf_offset)?,
            bcf_face: gpu.upload(&m.bcf_face)?,

            patches: m.patches.clone(),
            total_volume: m.v.iter().copied().sum(),
        })
    }
}

pub mod geometry;
pub mod topology;
