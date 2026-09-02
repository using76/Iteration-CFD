// meteor-cfd - Copyright (c) 2026 주식회사 메테오시뮬레이션 (Meteo Simulation Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

//! The finite-volume geometry sweep on the device.
//!
//! Written from:
//!   ofgpu SPEC-LIT.md sections 2 and 82
//!   this crate's own `src/mesh/geometry.rs`, of which every kernel in
//!     `cuda/meshgeom.cu` is a line-by-line restatement
//! No GPL-licensed source was consulted.
//!
//! # Why this exists
//!
//! Section 75.8 measured what an adapt costs and found the answer was not the
//! CUDA graph. Capture and instantiate are under 0.08 ms and do not grow with
//! the mesh; the host mesh rebuild is 2.3 / 12.8 / 30.7 ms at 512 / 4096 /
//! 13824 cells and grows linearly. An adapt that costs thirty milliseconds can
//! only happen every few thousand steps, and a flame moves in tens.
//!
//! Section 75.8 then named `mesh/geometry.rs::compute` as the binding
//! constraint. **Section 82.2 measures that claim and it is wrong**: the
//! geometry sweep is 31 % of the rebuild and the emitter's face-grouping loop
//! is 54 %. This module removes most of the 31 %, and 82.9 says what is left.
//!
//! # The claim
//!
//! **Bitwise identity with [`crate::mesh::geometry::compute`], on every mesh.**
//! Not "to a tolerance" - this is a refactor of WHERE the work happens, and a
//! geometry array that differs in the last bit changes every operator
//! downstream of it. Two things make that claim true and both are fragile
//! enough to be guarded by a test:
//!
//! 1. **`-fmad=false` on `meshgeom.cu`.** nvcc contracts `a*b + c` into one
//!    fused multiply-add and rustc never does. `build.rs::FMAD_OFF_UNITS`
//!    carries the flag by name and
//!    [`tests::the_translation_unit_is_compiled_with_fmad_off`] reads build.rs
//!    to check it is still there. Section 82.4 measures what happens without
//!    it, so the flag is a measurement and not a precaution.
//! 2. **The gather order.** The host walks faces and scatters into cells; a
//!    cell therefore accumulates its internal faces in ascending face id and
//!    then its boundary faces in ascending face id. That is exactly the order
//!    of its `cf_face` slice followed by its `bcf_face` slice, which
//!    `mesh::topology` fills in ascending face id for precisely this kind of
//!    reason. The kernel gathers in that order and in no other.
//!
//! # What stays on the host, and why
//!
//! The patch-level bookkeeping: validation, the cyclic pairing, and the
//! derivation of `b_kind`/`b_patch`/`b_nbr_cell`. Two of the three are loops
//! over PATCHES - six of them on a box; the third, `geometry::validate`, walks
//! every vertex of every face. Section 82.2 BOUNDS the three together at under
//! a millisecond against a 9.4 ms sweep - bounds and not measures, because the
//! figure is a residual of three separate timings, and section 82.1 says what
//! happened when an earlier draft of this module read one such residual as a
//! measurement.

use cudarc::driver::{CudaFunction, PushKernelArg};

use crate::device::{cfg_for, Gpu, KernelSet};
use crate::error::{Error, Result};
use crate::mesh::{geometry, HostMesh, PatchKind};
use crate::{DevBuf, Label, Scalar, Vec3};

/// The four kernels of `cuda/meshgeom.cu`.
pub struct MeshGeomKernels {
    pub face_geometry: CudaFunction,
    pub cell_geometry: CudaFunction,
    pub internal_metrics: CudaFunction,
    pub boundary_metrics: CudaFunction,
}

impl MeshGeomKernels {
    pub fn new(gpu: &Gpu) -> Result<Self> {
        Self::from_cubin(gpu, crate::kernels::MESHGEOM)
    }

    /// Load the same four kernels from a named module.
    ///
    /// There is exactly one other module: `kernels::MESHGEOM_FMAD`, the same
    /// source compiled with nvcc's default multiply-add contraction, which
    /// `tests::the_contraction_this_unit_turns_off_is_real` runs so that the
    /// `-fmad=false` flag is held by a measurement rather than by a belief.
    /// Nothing else should call this.
    pub(crate) fn from_cubin(gpu: &Gpu, cubin: &[u8]) -> Result<Self> {
        let ks = KernelSet::new(gpu, cubin)?;
        Ok(Self {
            face_geometry: ks.func("meshFaceGeometry")?,
            cell_geometry: ks.func("meshCellGeometry")?,
            internal_metrics: ks.func("meshInternalFaceMetrics")?,
            boundary_metrics: ks.func("meshBoundaryFaceMetrics")?,
        })
    }
}

/// The face -> point CSR: the flattened form of the polyMesh face list.
///
/// The host sweep indexes `faces: &[Vec<Label>]`, one heap allocation per
/// face, which no kernel can read. Flattening is one pass with a running
/// offset and no arithmetic; section 82.2 measures it at 0.76 ms against the
/// 9.4 ms sweep it feeds, and section 82.9 records that a device-resident
/// emitter would produce this shape directly and skip it.
#[derive(Debug, Clone, Default)]
pub struct FacePointCsr {
    /// `[n_faces + 1]`, ascending, `offset[0] == 0`.
    pub offset: Vec<Label>,
    /// `[offset[n_faces]]` point ids, face by face, in winding order.
    pub point: Vec<Label>,
}

/// Flatten a polyMesh face list into [`FacePointCsr`].
pub fn flatten_faces(faces: &[Vec<Label>]) -> FacePointCsr {
    let total: usize = faces.iter().map(|f| f.len()).sum();
    let mut offset = Vec::with_capacity(faces.len() + 1);
    let mut point = Vec::with_capacity(total);
    offset.push(0 as Label);
    for f in faces {
        point.extend_from_slice(f);
        offset.push(point.len() as Label);
    }
    FacePointCsr { offset, point }
}

/// Every geometric array of a mesh, resident on the device.
///
/// The names are `HostMesh`'s, so [`GpuGeometry::download_into`] is a
/// field-for-field copy and a reader can check the correspondence by eye.
pub struct GpuGeometry {
    pub n_cells: usize,
    pub n_internal_faces: usize,
    pub n_boundary_faces: usize,

    pub v: DevBuf<Scalar>,
    pub c: DevBuf<Vec3>,

    pub sf: DevBuf<Vec3>,
    pub cf: DevBuf<Vec3>,
    pub mag_sf: DevBuf<Scalar>,
    pub weights: DevBuf<Scalar>,
    pub delta_coeffs: DevBuf<Scalar>,
    pub non_orth_corr: DevBuf<Vec3>,
    pub skew_corr: DevBuf<Vec3>,

    pub b_sf: DevBuf<Vec3>,
    pub b_mag_sf: DevBuf<Scalar>,
    pub b_cf: DevBuf<Vec3>,
    pub b_delta_coeffs: DevBuf<Scalar>,
    pub b_non_orth_corr: DevBuf<Vec3>,
    pub b_y: DevBuf<Scalar>,
    pub b_weights: DevBuf<Scalar>,
}

/// Upload a slice, padding an empty one to a single zeroed element.
///
/// A zero-length `cuMemAlloc` is legal and returns a null pointer, which a
/// kernel that never dereferences it would survive - but only by accident, and
/// only until someone adds a bounds check that reads `offset[0]`. One element
/// costs nothing and removes the question.
fn upload_padded<T>(gpu: &Gpu, v: &[T]) -> Result<DevBuf<T>>
where
    T: cudarc::driver::DeviceRepr + cudarc::driver::ValidAsZeroBits + Default + Clone,
{
    if v.is_empty() {
        return gpu.zeros(1);
    }
    gpu.upload(v)
}

impl GpuGeometry {
    /// Run the four kernels. `pair` is the cyclic pairing and `b_weights_in`
    /// the boundary weights the caller already has - both derived on the host,
    /// see the module header.
    ///
    /// Nothing is downloaded here. A device-resident adapt keeps this struct
    /// and hands its buffers straight to `GpuMesh`; the host round trip lives
    /// in [`GpuGeometry::download_into`] and is paid only by a caller that
    /// wants a `HostMesh` back.
    pub fn compute(
        gpu: &Gpu,
        k: &MeshGeomKernels,
        m: &HostMesh,
        points: &[Vec3],
        csr: &FacePointCsr,
        pair: &[Label],
        b_weights_in: &[Scalar],
    ) -> Result<Self> {
        let (n_cells, n_if, n_bf) = (m.n_cells, m.n_internal_faces, m.n_boundary_faces);
        let n_faces = n_if + n_bf;

        if csr.offset.len() != n_faces + 1 {
            return Err(Error::Mesh(format!(
                "gpu_compute_geometry: the face CSR has {} offsets for {n_faces} faces",
                csr.offset.len()
            )));
        }
        if pair.len() != n_bf || b_weights_in.len() != n_bf {
            return Err(Error::Mesh(format!(
                "gpu_compute_geometry: the cyclic pairing has {} entries and the \
                 boundary weights {}, for {n_bf} boundary faces",
                pair.len(),
                b_weights_in.len()
            )));
        }

        let d_points = upload_padded(gpu, points)?;
        let d_foff = gpu.upload(&csr.offset)?;
        let d_fpt = upload_padded(gpu, &csr.point)?;
        let d_owner = upload_padded(gpu, &m.owner)?;
        let d_nbr = upload_padded(gpu, &m.neighbour)?;
        let d_bfc = upload_padded(gpu, &m.b_face_cells)?;
        let d_pair = upload_padded(gpu, pair)?;
        let d_cfo = gpu.upload(&m.cf_offset)?;
        let d_cff = upload_padded(gpu, &m.cf_face)?;
        let d_cfw = upload_padded(gpu, &m.cf_own)?;
        let d_bcfo = gpu.upload(&m.bcf_offset)?;
        let d_bcff = upload_padded(gpu, &m.bcf_face)?;

        let mut f_sf: DevBuf<Vec3> = gpu.zeros(n_faces.max(1))?;
        let mut f_cf: DevBuf<Vec3> = gpu.zeros(n_faces.max(1))?;

        let mut out = Self {
            n_cells,
            n_internal_faces: n_if,
            n_boundary_faces: n_bf,
            v: gpu.zeros(n_cells.max(1))?,
            c: gpu.zeros(n_cells.max(1))?,
            sf: gpu.zeros(n_if.max(1))?,
            cf: gpu.zeros(n_if.max(1))?,
            mag_sf: gpu.zeros(n_if.max(1))?,
            weights: gpu.zeros(n_if.max(1))?,
            delta_coeffs: gpu.zeros(n_if.max(1))?,
            non_orth_corr: gpu.zeros(n_if.max(1))?,
            skew_corr: gpu.zeros(n_if.max(1))?,
            b_sf: gpu.zeros(n_bf.max(1))?,
            b_mag_sf: gpu.zeros(n_bf.max(1))?,
            b_cf: gpu.zeros(n_bf.max(1))?,
            b_delta_coeffs: gpu.zeros(n_bf.max(1))?,
            b_non_orth_corr: gpu.zeros(n_bf.max(1))?,
            b_y: gpu.zeros(n_bf.max(1))?,
            // Uploaded, not zeroed: an uncoupled boundary face keeps the weight
            // the caller supplied, exactly as the host sweep leaves it alone.
            b_weights: upload_padded(gpu, b_weights_in)?,
        };

        let (nl_f, nl_c, nl_if, nl_bf) = (
            n_faces as Label,
            n_cells as Label,
            n_if as Label,
            n_bf as Label,
        );

        unsafe {
            if n_faces > 0 {
                gpu.stream()
                    .launch_builder(&k.face_geometry)
                    .arg(&mut f_sf)
                    .arg(&mut f_cf)
                    .arg(&d_foff)
                    .arg(&d_fpt)
                    .arg(&d_points)
                    .arg(&nl_f)
                    .launch(cfg_for(n_faces))?;
            }
            if n_cells > 0 {
                gpu.stream()
                    .launch_builder(&k.cell_geometry)
                    .arg(&mut out.v)
                    .arg(&mut out.c)
                    .arg(&f_sf)
                    .arg(&f_cf)
                    .arg(&d_cfo)
                    .arg(&d_cff)
                    .arg(&d_cfw)
                    .arg(&d_bcfo)
                    .arg(&d_bcff)
                    .arg(&nl_if)
                    .arg(&nl_c)
                    .launch(cfg_for(n_cells))?;
            }
            if n_if > 0 {
                gpu.stream()
                    .launch_builder(&k.internal_metrics)
                    .arg(&mut out.sf)
                    .arg(&mut out.cf)
                    .arg(&mut out.mag_sf)
                    .arg(&mut out.weights)
                    .arg(&mut out.delta_coeffs)
                    .arg(&mut out.non_orth_corr)
                    .arg(&mut out.skew_corr)
                    .arg(&f_sf)
                    .arg(&f_cf)
                    .arg(&out.c)
                    .arg(&d_owner)
                    .arg(&d_nbr)
                    .arg(&nl_if)
                    .launch(cfg_for(n_if))?;
            }
            if n_bf > 0 {
                gpu.stream()
                    .launch_builder(&k.boundary_metrics)
                    .arg(&mut out.b_sf)
                    .arg(&mut out.b_mag_sf)
                    .arg(&mut out.b_cf)
                    .arg(&mut out.b_delta_coeffs)
                    .arg(&mut out.b_non_orth_corr)
                    .arg(&mut out.b_y)
                    .arg(&mut out.b_weights)
                    .arg(&f_sf)
                    .arg(&f_cf)
                    .arg(&out.c)
                    .arg(&d_bfc)
                    .arg(&d_pair)
                    .arg(&nl_if)
                    .arg(&nl_bf)
                    .launch(cfg_for(n_bf))?;
            }
        }

        Ok(out)
    }

    /// Copy every array back into a `HostMesh`, truncated to its logical
    /// length (the buffers are padded to one element when a mesh has no
    /// boundary faces, or no internal ones).
    pub fn download_into(&self, gpu: &Gpu, m: &mut HostMesh) -> Result<()> {
        fn cut<T: Clone>(mut v: Vec<T>, n: usize) -> Vec<T> {
            v.truncate(n);
            v
        }
        let (nc, nif, nbf) = (self.n_cells, self.n_internal_faces, self.n_boundary_faces);

        m.v = cut(gpu.download(&self.v)?, nc);
        m.c = cut(gpu.download(&self.c)?, nc);

        m.sf = cut(gpu.download(&self.sf)?, nif);
        m.cf = cut(gpu.download(&self.cf)?, nif);
        m.mag_sf = cut(gpu.download(&self.mag_sf)?, nif);
        m.weights = cut(gpu.download(&self.weights)?, nif);
        m.delta_coeffs = cut(gpu.download(&self.delta_coeffs)?, nif);
        m.non_orth_corr = cut(gpu.download(&self.non_orth_corr)?, nif);
        m.skew_corr = cut(gpu.download(&self.skew_corr)?, nif);

        m.b_sf = cut(gpu.download(&self.b_sf)?, nbf);
        m.b_mag_sf = cut(gpu.download(&self.b_mag_sf)?, nbf);
        m.b_cf = cut(gpu.download(&self.b_cf)?, nbf);
        m.b_delta_coeffs = cut(gpu.download(&self.b_delta_coeffs)?, nbf);
        m.b_non_orth_corr = cut(gpu.download(&self.b_non_orth_corr)?, nbf);
        m.b_y = cut(gpu.download(&self.b_y)?, nbf);
        m.b_weights = cut(gpu.download(&self.b_weights)?, nbf);

        Ok(())
    }
}

/// The device twin of [`crate::mesh::geometry::compute`]: same signature, same
/// bits, one host round trip.
///
/// A caller that wants the geometry to STAY on the device calls
/// [`GpuGeometry::compute`] directly and never pays the download; this
/// function exists so that the two sweeps can be compared field for field, and
/// so that an existing host caller can be switched over one line at a time.
pub fn gpu_compute_geometry(
    gpu: &Gpu,
    k: &MeshGeomKernels,
    m: &mut HostMesh,
    points: &[Vec3],
    faces: &[Vec<Label>],
) -> Result<()> {
    let csr = flatten_faces(faces);
    gpu_compute_geometry_csr(gpu, k, m, points, faces, &csr)
}

/// As [`gpu_compute_geometry`], for a caller that already holds the flattened
/// face list. `faces` is still taken, because the host validation names the
/// offending face and a CSR alone cannot say which `Vec` it came from.
pub fn gpu_compute_geometry_csr(
    gpu: &Gpu,
    k: &MeshGeomKernels,
    m: &mut HostMesh,
    points: &[Vec3],
    faces: &[Vec<Label>],
    csr: &FacePointCsr,
) -> Result<()> {
    // ---- the host prologue, identical to `geometry::compute` --------------
    geometry::validate(m, points, faces)?;
    refuse_a_mesh_without_a_cell_face_csr(m)?;
    refuse_a_cell_with_no_faces(m)?;

    let n_bf = m.n_boundary_faces;
    m.n_points = points.len();

    let pair = geometry::cyclic_pairing(m)?;

    if m.b_kind.len() != n_bf {
        m.b_kind = vec![PatchKind::Generic as Label; n_bf];
        for pi in m.patches.iter() {
            for kk in 0..pi.size.min(n_bf.saturating_sub(pi.start)) {
                m.b_kind[pi.start + kk] = pi.kind as Label;
            }
        }
    }
    if m.b_patch.len() != n_bf {
        m.b_patch = vec![-1; n_bf];
        for (p, pi) in m.patches.iter().enumerate() {
            for kk in 0..pi.size.min(n_bf.saturating_sub(pi.start)) {
                m.b_patch[pi.start + kk] = p as Label;
            }
        }
    }
    if m.b_nbr_cell.len() != n_bf {
        m.b_nbr_cell = (0..n_bf)
            .map(|bf| {
                if pair[bf] >= 0 {
                    m.b_face_cells[pair[bf] as usize]
                } else {
                    -1
                }
            })
            .collect();
    }
    m.b_nbr_face = pair.clone();
    if m.b_weights.len() != n_bf {
        m.b_weights = vec![1.0; n_bf];
    }

    // ---- the device sweep -------------------------------------------------
    let b_weights_in = std::mem::take(&mut m.b_weights);
    let g = GpuGeometry::compute(gpu, k, m, points, csr, &pair, &b_weights_in);
    let g = match g {
        Ok(g) => g,
        Err(e) => {
            m.b_weights = b_weights_in;
            return Err(e);
        }
    };
    g.download_into(gpu, m)?;
    Ok(())
}

/// The one precondition the host sweep does not have.
///
/// `geometry::compute` walks faces and scatters, so it needs no addressing at
/// all. Every kernel here gathers, so it needs the cell -> face CSR - which is
/// also what makes the accumulation order, and therefore the bitwise claim,
/// reproducible. `HostMesh::build_cell_face_maps` builds it and every emitter
/// in this crate already calls it first; a caller that has not is refused by
/// name rather than reading a short array.
///
/// SPEC-LIT section 82.7 carries this as a contract row.
fn refuse_a_mesh_without_a_cell_face_csr(m: &HostMesh) -> Result<()> {
    let (nc, n_if, n_bf) = (m.n_cells, m.n_internal_faces, m.n_boundary_faces);
    let want = [
        ("cf_offset", m.cf_offset.len(), nc + 1),
        ("cf_face", m.cf_face.len(), 2 * n_if),
        ("cf_own", m.cf_own.len(), 2 * n_if),
        ("bcf_offset", m.bcf_offset.len(), nc + 1),
        ("bcf_face", m.bcf_face.len(), n_bf),
    ];
    for (name, got, exp) in want {
        if got != exp {
            return Err(Error::Mesh(format!(
                "gpu_compute_geometry: the device sweep gathers over the cell ->                  face CSR, and `{name}` holds {got} entries where the mesh needs                  {exp}. Call HostMesh::build_cell_face_maps() first, or use the                  host sweep mesh::geometry::compute, which needs no addressing"
            )));
        }
    }
    Ok(())
}

/// A cell with no faces is not a cell.
///
/// The host sweep discovers this while averaging the face centroids and
/// returns `Error::Mesh` naming the cell. A kernel cannot return, so the same
/// question is asked here, before any launch, off the same CSR the kernel
/// reads - and it is asked in the same words, because a caller that switches
/// sweeps must not get a different diagnosis of the same broken mesh.
fn refuse_a_cell_with_no_faces(m: &HostMesh) -> Result<()> {
    for c in 0..m.n_cells {
        let internal = m.cf_offset[c + 1] - m.cf_offset[c];
        let boundary = m.bcf_offset[c + 1] - m.bcf_offset[c];
        if internal + boundary == 0 {
            return Err(Error::Mesh(format!(
                "compute_geometry: cell {c} has no faces, so it is not a cell"
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests;
