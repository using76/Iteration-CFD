// meteor-cfd - Copyright (c) 2026 주식회사 메테오시뮬레이션 (Meteo Simulation Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

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
//!
//! Provenance: ORIGINAL - the mesh types and the LDU -> CSR inversion. The
//! addressing CONVENTION it stores (upper-triangular face order) is a property
//! of the case format it reads, not of anyone's source; the inversion itself is
//! designed here. `PROVENANCE.md`, *GPU plumbing and tooling - original*. No
//! GPL-licensed source was consulted.

use crate::device::{DevBuf, Gpu};
use crate::error::{Error, Result};
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
    /// A conformal fluid/solid (or solid/solid) conjugate interface -
    /// SPEC-LIT S47.4. Topologically a `Cyclic` couple with a zero transform:
    /// `b_nbr_cell` names the cell on the other side, in the CONCATENATED
    /// thermal-mesh numbering, so `lduAmul` and
    /// `lduAddBoundaryContributions` - which test `bNbrCell >= 0`, never
    /// "cyclic" - solve it implicitly with no new matrix code.
    ///
    /// It is a SEPARATE discriminant from [`Self::Cyclic`] for two reasons,
    /// both in S47.3. `fvLapNonOrth`'s cyclic branch interpolates the two
    /// cells' gradients across the couple, which is meaningless where `kappa`
    /// and `grad T` are both discontinuous, so an interface face is skipped
    /// there instead. And every OTHER kernel's cyclic branch reads
    /// `psi[nbr]` in place of the evaluated face value, which cannot
    /// represent the `R_c` temperature jump; an interface face falls into the
    /// uncoupled branch and is read from `bf[bf]`, the Robin face value of
    /// (S47.5), which can.
    ///
    /// No mesh this crate read before SPEC-LIT S47 carries it, so every
    /// existing case is unmoved by construction.
    Interface = 6,
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
            Self::Interface => "interface",
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
    /// `[n_if]` the SKEWNESS vector of SPEC-LIT S74.4,
    /// `s_f = Cf - (C_P + (1 - w) d)`: the offset from the point the
    /// interpolation weight actually places `psi_f` at - where the face plane
    /// cuts the line `P-N` - to the face CENTROID, which is the point
    /// Green-Gauss and the midpoint rule both assume it is.
    ///
    /// Exactly zero on a mesh whose faces are unskewed, because there `w` is
    /// `1/2` and `Cf` is the midpoint of `C_P C_N`. Non-zero at a 2:1
    /// refinement interface, where it is `0.1421 |d|`.
    pub skew_corr: Vec<Vec3>,

    // ---- boundary faces, all patches flattened ---------------------------
    pub b_face_cells: Vec<Label>,
    pub b_sf: Vec<Vec3>,
    pub b_mag_sf: Vec<Scalar>,
    pub b_cf: Vec<Vec3>,
    pub b_delta_coeffs: Vec<Scalar>,
    /// `[n_bf]` the over-relaxed explicit non-orthogonal correction vector
    /// (SPEC-LIT section 2.4), computed through the cyclic delta for a
    /// coupled face; `Vec3::ZERO` on every uncoupled boundary face, where the
    /// boundary condition is imposed directly on the face rather than
    /// interpolated across a `d` that could be non-collinear with `Sf`.
    pub b_non_orth_corr: Vec<Vec3>,
    /// `[n_bf]` wall-normal distance of the adjacent cell centre, `nf . delta`
    pub b_y: Vec<Scalar>,
    /// `[n_bf]` cyclic: the cell across the couple; `-1` otherwise
    pub b_nbr_cell: Vec<Label>,
    /// `[n_bf]` coupled: the boundary FACE on the other side of the couple;
    /// `-1` otherwise - SPEC-LIT §48.3.
    ///
    /// The pairing has always existed (`mesh/geometry.rs::cyclic_pairing`
    /// computes it, and `b_nbr_cell` is derived from it); it was thrown away
    /// after that one use. It is kept now because a coupled face's two
    /// `boundary_coeffs` are the two halves of ONE matrix entry pair, and
    /// `solver::matrix_is_symmetric` cannot compare them without knowing
    /// which face is which - which is the blind spot §48.3 closes.
    pub b_nbr_face: Vec<Label>,
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

    // ---- the merged, GLOBAL-face-ordered row map - SPEC-LIT §70 ----------
    /// `[n_internal_faces + n_boundary_faces]` the GLOBAL id of every face,
    /// indexed by SLOT: slot `f` for internal face `f`, slot
    /// `n_internal_faces + bf` for boundary face `bf` - the polyMesh face
    /// numbering.
    ///
    /// **Empty means the identity**, which is what an undecomposed mesh has
    /// and what every mesh this crate can currently read gets. It exists so
    /// that the order a row of `A psi` is summed in is a property of the MESH
    /// rather than of the partition: cut an internal face and it becomes a
    /// boundary face on both sides, which moves its term between the two CSRs
    /// above and renumbers everything after it. See §70.1.
    pub global_face: Vec<Label>,

    /// `[n_cells + 1]` offsets into [`Self::rf_face`] - SPEC-LIT §70.2.
    pub rf_offset: Vec<Label>,
    /// `[2 * n_internal_faces + n_boundary_faces]` the face's index in its OWN
    /// array: an internal face id when [`Self::rf_flags`] has
    /// [`topology::RF_BOUNDARY`] clear, a boundary face id when it is set.
    ///
    /// Every cell's slice is ascending in GLOBAL face id. Under the identity
    /// map that is bit-for-bit `cf_face`'s slice followed by `bcf_face`'s, and
    /// §70.3 is the argument that it must be.
    pub rf_face: Vec<Label>,
    /// `[2 * n_internal_faces + n_boundary_faces]`
    /// [`topology::RF_OWNS`] | [`topology::RF_BOUNDARY`].
    pub rf_flags: Vec<Label>,
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
    /// Invert `owner`/`neighbour` into the two CSR maps, and build the merged
    /// global-face-ordered row map on top of them.
    ///
    /// Within each cell the faces are kept in ascending face index so the
    /// gather order is deterministic, which is what makes results bitwise
    /// reproducible across runs. The merged map (SPEC-LIT §70) additionally
    /// makes that order a property of the MESH rather than of the partition,
    /// by keying it on the global face id instead of the local one.
    pub fn build_cell_face_maps(&mut self) {
        crate::mesh::topology::build_cell_face_maps(self)
    }

    /// Compute `v`, `c`, `sf`, `mag_sf`, `cf`, `weights`, `delta_coeffs`,
    /// `non_orth_corr`, `b_non_orth_corr` and the boundary metrics from raw
    /// points and faces.
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
    /// SPEC-LIT S74.4. Zero everywhere on an unskewed mesh, and read only by
    /// the `skewCorrected` path, which is not the default.
    pub skew_corr: DevBuf<Vec3>,

    pub b_face_cells: DevBuf<Label>,
    pub b_sf: DevBuf<Vec3>,
    pub b_mag_sf: DevBuf<Scalar>,
    pub b_cf: DevBuf<Vec3>,
    pub b_delta_coeffs: DevBuf<Scalar>,
    pub b_non_orth_corr: DevBuf<Vec3>,
    pub b_y: DevBuf<Scalar>,
    pub b_nbr_cell: DevBuf<Label>,
    /// `[n_bf]` the face across the couple, `-1` otherwise - SPEC-LIT §48.3.
    pub b_nbr_face: DevBuf<Label>,
    pub b_weights: DevBuf<Scalar>,
    pub b_kind: DevBuf<Label>,
    pub b_patch: DevBuf<Label>,

    pub cf_offset: DevBuf<Label>,
    pub cf_face: DevBuf<Label>,
    pub cf_own: DevBuf<Label>,
    pub bcf_offset: DevBuf<Label>,
    pub bcf_face: DevBuf<Label>,

    /// The merged, global-face-ordered row map - SPEC-LIT §70. One list per
    /// cell covering its internal AND boundary faces, ascending in global face
    /// id, so that the summation order of a row of `A psi` is a property of
    /// the mesh and not of how it was cut up.
    pub rf_offset: DevBuf<Label>,
    pub rf_face: DevBuf<Label>,
    pub rf_flags: DevBuf<Label>,

    /// Kept on the host for reporting only.
    pub patches: Vec<PatchInfo>,
    pub total_volume: Scalar,
}

/// Everything a [`GpuMesh`] carries that is **not** geometry: the addressing,
/// the boundary bookkeeping and the §70 row map.
///
/// It exists so that there is exactly one of it. A `GpuMesh` can now be built
/// two ways - from a `HostMesh` that has run the host sweep, or from a
/// [`gpugeom::GpuGeometry`] the device sweep just filled - and §70's guarantee
/// is that a row's summation order is a property of the MESH. Two constructors
/// that each built their own row map would be two rules, and the second one
/// would be wrong the first time someone edited the first. SPEC-LIT §83.4.
struct GpuTopology {
    owner: DevBuf<Label>,
    neighbour: DevBuf<Label>,
    b_face_cells: DevBuf<Label>,
    b_nbr_cell: DevBuf<Label>,
    b_nbr_face: DevBuf<Label>,
    b_kind: DevBuf<Label>,
    b_patch: DevBuf<Label>,
    cf_offset: DevBuf<Label>,
    cf_face: DevBuf<Label>,
    cf_own: DevBuf<Label>,
    bcf_offset: DevBuf<Label>,
    bcf_face: DevBuf<Label>,
    rf_offset: DevBuf<Label>,
    rf_face: DevBuf<Label>,
    rf_flags: DevBuf<Label>,
}

impl GpuTopology {
    fn upload(gpu: &Gpu, m: &HostMesh) -> Result<Self> {
        // SPEC-LIT §70.3. Half the meshes in this crate's tests are written
        // out by hand and never call `build_cell_face_maps`, so the merged row
        // map may not be there. Uploading an empty one would give every row a
        // slice of length zero and make `amul` return `diag*psi` - wrong, and
        // wrong QUIETLY, which is the failure mode this project has been bitten
        // by before. Build it here instead, from the addressing the mesh does
        // carry. Three vectors, not a clone of the mesh.
        //
        // A device-built mesh reaches this by the same route and gets the same
        // answer BY CONSTRUCTION, because there is one copy of these lines:
        // the row map is a function of `owner`, `neighbour`, `b_face_cells`
        // and `global_face`, none of which the geometry sweep touches, so
        // where the geometry was computed cannot move it. §83.4.
        let rebuilt = if m.rf_offset.len() == m.n_cells + 1
            && m.rf_face.len() == 2 * m.n_internal_faces + m.n_boundary_faces
            && m.rf_flags.len() == m.rf_face.len()
        {
            None
        } else {
            Some(topology::row_face_map(
                m.n_cells,
                m.n_internal_faces,
                m.n_boundary_faces,
                &m.owner,
                &m.neighbour,
                &m.b_face_cells,
                &m.global_face,
            ))
        };
        let (rf_offset, rf_face, rf_flags) = match &rebuilt {
            Some((o, f, g)) => (o.as_slice(), f.as_slice(), g.as_slice()),
            None => (
                m.rf_offset.as_slice(),
                m.rf_face.as_slice(),
                m.rf_flags.as_slice(),
            ),
        };

        Ok(Self {
            owner: gpu.upload(&m.owner)?,
            neighbour: gpu.upload(&m.neighbour)?,
            b_face_cells: gpu.upload(&m.b_face_cells)?,
            b_nbr_cell: gpu.upload(&m.b_nbr_cell)?,
            // A mesh built before §48.3, or by hand in a test, may have no
            // pairing array at all. `-1` everywhere is "no couple", which is
            // what every consumer already means by it.
            b_nbr_face: if m.b_nbr_face.len() == m.n_boundary_faces {
                gpu.upload(&m.b_nbr_face)?
            } else {
                gpu.upload(&vec![-1 as Label; m.n_boundary_faces])?
            },
            b_kind: gpu.upload(&m.b_kind)?,
            b_patch: gpu.upload(&m.b_patch)?,

            cf_offset: gpu.upload(&m.cf_offset)?,
            cf_face: gpu.upload(&m.cf_face)?,
            cf_own: gpu.upload(&m.cf_own)?,
            bcf_offset: gpu.upload(&m.bcf_offset)?,
            bcf_face: gpu.upload(&m.bcf_face)?,

            rf_offset: gpu.upload(rf_offset)?,
            rf_face: gpu.upload(rf_face)?,
            rf_flags: gpu.upload(rf_flags)?,
        })
    }
}

impl GpuMesh {
    pub fn upload(gpu: &Gpu, m: &HostMesh) -> Result<Self> {
        let t = GpuTopology::upload(gpu, m)?;

        Ok(Self {
            n_cells: m.n_cells,
            n_internal_faces: m.n_internal_faces,
            n_boundary_faces: m.n_boundary_faces,

            owner: t.owner,
            neighbour: t.neighbour,

            v: gpu.upload(&m.v)?,
            c: gpu.upload(&m.c)?,

            sf: gpu.upload(&m.sf)?,
            mag_sf: gpu.upload(&m.mag_sf)?,
            cf: gpu.upload(&m.cf)?,
            weights: gpu.upload(&m.weights)?,
            delta_coeffs: gpu.upload(&m.delta_coeffs)?,
            non_orth_corr: gpu.upload(&m.non_orth_corr)?,
            // A mesh written out by hand in a test may never have run the
            // geometry sweep. Zero is what an unskewed mesh's skewness vector
            // IS, so an absent array and a uniform mesh give the same answer -
            // and a buffer of the right length, which the kernels index
            // unconditionally.
            skew_corr: if m.skew_corr.len() == m.n_internal_faces {
                gpu.upload(&m.skew_corr)?
            } else {
                gpu.upload(&vec![Vec3::ZERO; m.n_internal_faces])?
            },

            b_face_cells: t.b_face_cells,
            b_sf: gpu.upload(&m.b_sf)?,
            b_mag_sf: gpu.upload(&m.b_mag_sf)?,
            b_cf: gpu.upload(&m.b_cf)?,
            b_delta_coeffs: gpu.upload(&m.b_delta_coeffs)?,
            b_non_orth_corr: gpu.upload(&m.b_non_orth_corr)?,
            b_y: gpu.upload(&m.b_y)?,
            b_nbr_cell: t.b_nbr_cell,
            b_nbr_face: t.b_nbr_face,
            b_weights: gpu.upload(&m.b_weights)?,
            b_kind: t.b_kind,
            b_patch: t.b_patch,

            cf_offset: t.cf_offset,
            cf_face: t.cf_face,
            cf_own: t.cf_own,
            bcf_offset: t.bcf_offset,
            bcf_face: t.bcf_face,

            rf_offset: t.rf_offset,
            rf_face: t.rf_face,
            rf_flags: t.rf_flags,

            patches: m.patches.clone(),
            total_volume: m.v.iter().copied().sum(),
        })
    }

    /// The device-resident constructor: the topology from `m`, the geometry
    /// **moved** out of a [`gpugeom::GpuGeometry`] the device sweep just
    /// filled, and not one geometric array through the host.
    ///
    /// SPEC-LIT §83. `GpuMesh::upload` is how a mesh reaches the device when
    /// the geometry was computed on the host; this is how it reaches the
    /// device when the geometry never left it. §82.2 measured the two costs
    /// that removes at 13824 cells - the sixteen-array download inside
    /// `gpu_compute_geometry` and the sixteen-array upload inside
    /// `GpuMesh::upload` - and §83.3 measures what is left.
    ///
    /// The sixteen `DevBuf`s are moved, not copied: `g` is consumed, so there
    /// is no moment at which the same geometry exists twice on the device and
    /// no way to hand the same buffers to two meshes.
    ///
    /// **The row map is the same rule.** Both constructors go through the one
    /// private `GpuTopology::upload`, which is the only copy of §70.3's
    /// rebuild, so an adapted mesh cannot get a different row from a
    /// generated one.
    ///
    /// **Refused by name.** `m.v` must already hold the cell volumes.
    /// `total_volume` is `v.iter().sum()` - a fold in ascending cell id - and
    /// this constructor will not silently spend a device-to-host copy to get
    /// it, nor quietly substitute a tree reduction that would not be the same
    /// bits. **Alternative: `m.v = g.download_volumes(gpu)?` before the
    /// call**, which is the one array §83.3 accounts for and the same array
    /// `adapt::plan` needs anyway for the conservative weights of §75.6.
    pub fn from_device_geometry(
        gpu: &Gpu,
        m: &HostMesh,
        g: gpugeom::GpuGeometry,
    ) -> Result<Self> {
        let want = [
            ("cells", g.n_cells, m.n_cells),
            ("internal faces", g.n_internal_faces, m.n_internal_faces),
            ("boundary faces", g.n_boundary_faces, m.n_boundary_faces),
        ];
        for (what, got, exp) in want {
            if got != exp {
                return Err(Error::Mesh(format!(
                    "GpuMesh::from_device_geometry: the device geometry was computed for \
                     {got} {what} and the host mesh has {exp}. The two must be the same \
                     mesh; call mesh::gpugeom::gpu_geometry_resident on THIS mesh, or use \
                     GpuMesh::upload, which needs no device geometry"
                )));
            }
        }
        if m.v.len() != m.n_cells {
            return Err(Error::Mesh(format!(
                "GpuMesh::from_device_geometry: the host mesh carries {} cell volumes for \
                 {} cells, and `total_volume` is the ascending-cell-id fold of them. This \
                 constructor will not spend a hidden download to get it. Set `mesh.v = \
                 geometry.download_volumes(gpu)?` first - one array of the sixteen, which \
                 SPEC-LIT §83.3 accounts for - or use GpuMesh::upload",
                m.v.len(),
                m.n_cells
            )));
        }

        let t = GpuTopology::upload(gpu, m)?;

        Ok(Self {
            n_cells: m.n_cells,
            n_internal_faces: m.n_internal_faces,
            n_boundary_faces: m.n_boundary_faces,

            owner: t.owner,
            neighbour: t.neighbour,

            v: g.v,
            c: g.c,

            sf: g.sf,
            mag_sf: g.mag_sf,
            cf: g.cf,
            weights: g.weights,
            delta_coeffs: g.delta_coeffs,
            non_orth_corr: g.non_orth_corr,
            skew_corr: g.skew_corr,

            b_face_cells: t.b_face_cells,
            b_sf: g.b_sf,
            b_mag_sf: g.b_mag_sf,
            b_cf: g.b_cf,
            b_delta_coeffs: g.b_delta_coeffs,
            b_non_orth_corr: g.b_non_orth_corr,
            b_y: g.b_y,
            b_nbr_cell: t.b_nbr_cell,
            b_nbr_face: t.b_nbr_face,
            b_weights: g.b_weights,
            b_kind: t.b_kind,
            b_patch: t.b_patch,

            cf_offset: t.cf_offset,
            cf_face: t.cf_face,
            cf_own: t.cf_own,
            bcf_offset: t.bcf_offset,
            bcf_face: t.bcf_face,

            rf_offset: t.rf_offset,
            rf_face: t.rf_face,
            rf_flags: t.rf_flags,

            patches: m.patches.clone(),
            total_volume: m.v.iter().copied().sum(),
        })
    }
}

pub mod geometry;
pub mod gpugeom;
pub mod refined;
pub mod topology;
