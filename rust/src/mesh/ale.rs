// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

//! The moving mesh on the device - SPEC-LIT section 105.
//!
//! Written from:
//!   ofgpu SPEC-LIT.md section 105 (the ALE step, its refusals and its gate)
//!   ofgpu SPEC-LIT.md section 2 (the fan about the vertex average, 2.1;
//!     the pyramid decomposition whose fan-volume comparison 105.3 measures)
//!   ofgpu SPEC-LIT.md section 82 (the four geometry kernels this module
//!     re-launches on resident inputs)
//!   Demirdzic & Peric (1988), DOI 10.1002/fld.1650080906 - the space
//!     conservation law
//!   Thomas & Lombard (1979), DOI 10.2514/3.61273 - the geometric
//!     conservation law
//! No GPL-licensed source was consulted.
//!
//! The shared borrow. `Simple<'m>`, `Momentum<'m>`, `Energy` and `RasCore`
//! each hold a `&'m GpuMesh` for their whole life, so an `&mut GpuMesh` would
//! force the next unit to rebuild every solver object on every time step.
//! [`AleMesh::recompute_in_place`] therefore writes the sixteen geometry
//! arrays of `gm` through a SHARED borrow: it is sound because `Gpu::new`
//! disables event tracking before the first allocation and the crate has
//! exactly one stream, so every launch runs in issue order, no host reference
//! points into device memory, and with tracking off `.arg(&buf)` and
//! `.arg(&mut buf)` push the identical device pointer. The mutable state this
//! module owns - points, CSR, scratch, history - lives in [`AleMesh`], which
//! the caller holds `&mut`.
//!
//! The step order of [`AleMesh::advance`]: rotate the history
//! (`v00 <- v0 <- v`, `swept0 <- swept`), sweep the swept volume from
//! `points_old` and `points`, recompute the geometry in place, weight the
//! mesh flux with the scheme's own coefficients, and copy `points` to
//! `points_old` as its last act. Nothing downloads, nothing allocates,
//! nothing syncs - which is what makes the step capturable; its gate is
//! `the_ale_step_replays_bitwise`.
//!
//! NOT here: the relative flux `phi - phi_mesh` routed through the momentum
//! predictor, the pressure equation, turbulence and energy; the point
//! smoother; the case block. Those are the next units'.

use cudarc::driver::{CudaFunction, PushKernelArg};

use crate::device::{cfg_for, Gpu, KernelSet};
use crate::error::{Error, Result};
use crate::field::{BcKind, GpuScalarField, GpuSurfaceScalarField};
use crate::fv::{self, DivScheme, FvKernels};
use crate::ldu::GpuLduMatrix;
use crate::mesh::gpugeom::{FacePointCsr, MeshGeomKernels, flatten_faces};
use crate::mesh::refined::{self, RefinedBox};
use crate::mesh::{geometry, GpuMesh, HostMesh};
use crate::timescheme::{DdtCoeffs, DdtScheme};
use crate::{DevBuf, Label, Scalar, Vec3};

/// The four kernels of `cuda/ale.cu`.
pub struct AleKernels {
    pub swept_volume: CudaFunction,
    pub mesh_flux: CudaFunction,
    pub ddt_v: CudaFunction,
    pub ddt_rho_v: CudaFunction,
}

impl AleKernels {
    pub fn new(gpu: &Gpu) -> Result<Self> {
        Self::from_cubin(gpu, crate::kernels::ALE)
    }

    /// Load the same four kernels from a named module. There is exactly one
    /// other module: `kernels::ALE_FMAD`, the same source compiled with
    /// nvcc's default multiply-add contraction, which
    /// `tests::the_contraction_the_ale_unit_turns_off_is_real` runs so that
    /// the `-fmad=false` flag is held by a measurement rather than a belief.
    /// Nothing else should call this.
    pub(crate) fn from_cubin(gpu: &Gpu, cubin: &[u8]) -> Result<Self> {
        let ks = KernelSet::new(gpu, cubin)?;
        Ok(Self {
            swept_volume: ks.func("aleSweptVolume")?,
            mesh_flux: ks.func("aleMeshFlux")?,
            ddt_v: ks.func("tsDdtGeneralV")?,
            ddt_rho_v: ks.func("tsDdtGeneralRhoV")?,
        })
    }
}

/// The resident state of a moving mesh, beside the `GpuMesh` it moves.
pub struct AleMesh {
    pub n_cells: usize,
    pub n_internal_faces: usize,
    pub n_boundary_faces: usize,
    pub n_points: usize,
    /// `x^{n+1}` - what the caller moves, with `set_points` or a device
    /// kernel of its own.
    pub points: DevBuf<Vec3>,
    /// `x^n` - `advance` copies `points` here as its last act.
    pub points_old: DevBuf<Vec3>,
    /// The face -> point CSR, resident: `[n_faces + 1]` and
    /// `[offset[n_faces]]`.
    pub face_offset: DevBuf<Label>,
    pub face_point: DevBuf<Label>,
    /// `[n_bf]` the cyclic pairing of `geometry::cyclic_pairing`, padded to
    /// one element when empty.
    pub pair: DevBuf<Label>,
    /// `V^n`, the volume at `psi0`'s level; length `gm.v.len()`.
    pub v0: DevBuf<Scalar>,
    /// `V^{n-1}`, the volume at `psi00`'s level; length `gm.v.len()`.
    pub v00: DevBuf<Scalar>,
    /// `[max(n_faces, 1)]` the swept volume of the last `advance`, global
    /// face order (internal first).
    pub swept: DevBuf<Scalar>,
    /// the one before it.
    pub swept0: DevBuf<Scalar>,
    /// `aN*swept - a00*swept0`, named "phiMesh": positive when the owner
    /// grows.
    pub phi_mesh: GpuSurfaceScalarField,
    f_sf: DevBuf<Vec3>,
    f_cf: DevBuf<Vec3>,
    geom: MeshGeomKernels,
    k: AleKernels,
}

/// Upload a slice, padding an empty one to a single zeroed element - the
/// helper of `mesh/gpugeom.rs`, local because that one is private there.
fn upload_padded(gpu: &Gpu, v: &[Label]) -> Result<DevBuf<Label>> {
    if v.is_empty() {
        return gpu.zeros(1);
    }
    gpu.upload(v)
}

impl AleMesh {
    /// Build the moving-mesh state for ONE mesh, at its rest points.
    pub fn new(gpu: &Gpu, m: &HostMesh, gm: &GpuMesh, points: &[Vec3], csr: &FacePointCsr) -> Result<Self> {
        Self::with_cubin(gpu, m, gm, points, csr, crate::kernels::ALE)
    }

    pub(crate) fn with_cubin(
        gpu: &Gpu,
        m: &HostMesh,
        gm: &GpuMesh,
        points: &[Vec3],
        csr: &FacePointCsr,
        cubin: &[u8],
    ) -> Result<Self> {
        if gm.n_cells != m.n_cells
            || gm.n_internal_faces != m.n_internal_faces
            || gm.n_boundary_faces != m.n_boundary_faces
        {
            return Err(Error::Mesh(format!(
                "AleMesh::new: the device mesh is {}x{}+{} faces but the host \
                 mesh is {}x{}+{} - an AleMesh is built for ONE mesh, and this \
                 is two",
                gm.n_cells, gm.n_internal_faces, gm.n_boundary_faces,
                m.n_cells, m.n_internal_faces, m.n_boundary_faces
            )));
        }
        if points.is_empty() || (m.n_points != 0 && points.len() != m.n_points) {
            return Err(Error::Mesh(format!(
                "AleMesh::new: {} points for a mesh of {} - the moving state is \
                 one point per mesh point, no more and no fewer",
                points.len(),
                m.n_points
            )));
        }
        geometry::validate_csr(m, points, csr)?;
        let pair = geometry::cyclic_pairing(m)?;
        let (n_if, n_bf) = (m.n_internal_faces, m.n_boundary_faces);
        let n_faces = n_if + n_bf;

        let mut v0 = gpu.zeros(gm.v.len())?;
        let mut v00 = gpu.zeros(gm.v.len())?;
        gpu.stream().memcpy_dtod(&gm.v, &mut v0)?;
        gpu.stream().memcpy_dtod(&gm.v, &mut v00)?;

        Ok(Self {
            n_cells: m.n_cells,
            n_internal_faces: n_if,
            n_boundary_faces: n_bf,
            n_points: points.len(),
            points: gpu.upload(points)?,
            points_old: gpu.upload(points)?,
            face_offset: gpu.upload(&csr.offset)?,
            face_point: upload_padded(gpu, &csr.point)?,
            pair: upload_padded(gpu, &pair)?,
            v0,
            v00,
            swept: gpu.zeros(n_faces.max(1))?,
            swept0: gpu.zeros(n_faces.max(1))?,
            f_sf: gpu.zeros(n_faces.max(1))?,
            f_cf: gpu.zeros(n_faces.max(1))?,
            phi_mesh: GpuSurfaceScalarField::zeros(gpu, gm, "phiMesh")?,
            geom: MeshGeomKernels::new(gpu)?,
            k: AleKernels::from_cubin(gpu, cubin)?,
        })
    }

    /// Move the points from the host. Setup and prescribed-motion only: this
    /// is a host write, and `Gpu::write` is refused by name inside a CUDA
    /// graph capture (SPEC-LIT 81.3) - a captured step moves its points on
    /// the device instead, as the gate's `memcpy_dtod` does.
    pub fn set_points(&mut self, gpu: &Gpu, points: &[Vec3]) -> Result<()> {
        if points.len() != self.n_points {
            return Err(Error::Mesh(format!(
                "AleMesh::set_points: {} points, and this AleMesh moves {} - the \
                 mesh never gains or loses a point",
                points.len(),
                self.n_points
            )));
        }
        gpu.write(&mut self.points, points)
    }

    /// The counts this module was built against, asked again at every use.
    fn check_shape(&self, gm: &GpuMesh, who: &str) -> Result<()> {
        if gm.n_cells != self.n_cells
            || gm.n_internal_faces != self.n_internal_faces
            || gm.n_boundary_faces != self.n_boundary_faces
        {
            return Err(Error::Mesh(format!(
                "{who}: the GpuMesh is {}x{}+{} faces, but this AleMesh was \
                 built for a mesh of {} cells, {} internal and {} boundary faces",
                gm.n_cells, gm.n_internal_faces, gm.n_boundary_faces,
                self.n_cells, self.n_internal_faces, self.n_boundary_faces
            )));
        }
        Ok(())
    }

    /// Recompute the sixteen geometry arrays of `gm` in place, from the
    /// resident points, through a SHARED borrow of `gm`.
    ///
    /// SPEC-LIT 105.2. The four launches are `mesh/gpugeom.rs`'s `compute`,
    /// in the same order with the same argument order; what changed is where
    /// the inputs live - already resident - and where the outputs go: `gm`'s
    /// own arrays, written in place. With event tracking off and one stream,
    /// `.arg(&buf)` and `.arg(&mut buf)` push the identical device pointer,
    /// so the shared borrow costs nothing on the device and saves the caller
    /// a rebuild of every solver object that holds `&GpuMesh`.
    pub fn recompute_in_place(&mut self, gpu: &Gpu, gm: &GpuMesh) -> Result<()> {
        self.check_shape(gm, "AleMesh::recompute_in_place")?;
        let (n_cells, n_if, n_bf) = (self.n_cells, self.n_internal_faces, self.n_boundary_faces);
        let n_faces = n_if + n_bf;
        let (nl_f, nl_c, nl_if, nl_bf) =
            (n_faces as Label, n_cells as Label, n_if as Label, n_bf as Label);

        unsafe {
            if n_faces > 0 {
                gpu.stream()
                    .launch_builder(&self.geom.face_geometry)
                    .arg(&mut self.f_sf)
                    .arg(&mut self.f_cf)
                    .arg(&self.face_offset)
                    .arg(&self.face_point)
                    .arg(&self.points)
                    .arg(&nl_f)
                    .launch(cfg_for(n_faces))?;
            }
            if n_cells > 0 {
                gpu.stream()
                    .launch_builder(&self.geom.cell_geometry)
                    .arg(&gm.v) // written in place: SPEC-LIT 105.2
                    .arg(&gm.c) // written in place: SPEC-LIT 105.2
                    .arg(&self.f_sf)
                    .arg(&self.f_cf)
                    .arg(&gm.cf_offset)
                    .arg(&gm.cf_face)
                    .arg(&gm.cf_own)
                    .arg(&gm.bcf_offset)
                    .arg(&gm.bcf_face)
                    .arg(&nl_if)
                    .arg(&nl_c)
                    .launch(cfg_for(n_cells))?;
            }
            if n_if > 0 {
                gpu.stream()
                    .launch_builder(&self.geom.internal_metrics)
                    .arg(&gm.sf) // written in place: SPEC-LIT 105.2
                    .arg(&gm.cf) // written in place: SPEC-LIT 105.2
                    .arg(&gm.mag_sf) // written in place: SPEC-LIT 105.2
                    .arg(&gm.weights) // written in place: SPEC-LIT 105.2
                    .arg(&gm.delta_coeffs) // written in place: SPEC-LIT 105.2
                    .arg(&gm.non_orth_corr) // written in place: SPEC-LIT 105.2
                    .arg(&gm.skew_corr) // written in place: SPEC-LIT 105.2
                    .arg(&self.f_sf)
                    .arg(&self.f_cf)
                    .arg(&gm.c)
                    .arg(&gm.owner)
                    .arg(&gm.neighbour)
                    .arg(&nl_if)
                    .launch(cfg_for(n_if))?;
            }
            if n_bf > 0 {
                gpu.stream()
                    .launch_builder(&self.geom.boundary_metrics)
                    .arg(&gm.b_sf) // written in place: SPEC-LIT 105.2
                    .arg(&gm.b_mag_sf) // written in place: SPEC-LIT 105.2
                    .arg(&gm.b_cf) // written in place: SPEC-LIT 105.2
                    .arg(&gm.b_delta_coeffs) // written in place: SPEC-LIT 105.2
                    .arg(&gm.b_non_orth_corr) // written in place: SPEC-LIT 105.2
                    .arg(&gm.b_y) // written in place: SPEC-LIT 105.2
                    .arg(&gm.b_weights) // written in place: SPEC-LIT 105.2
                    .arg(&self.f_sf)
                    .arg(&self.f_cf)
                    .arg(&gm.c)
                    .arg(&gm.b_face_cells)
                    .arg(&self.pair)
                    .arg(&nl_if)
                    .arg(&nl_bf)
                    .launch(cfg_for(n_bf))?;
            }
        }

        Ok(())
    }

    /// One ALE step, end to end, on the one stream.
    ///
    /// SPEC-LIT 105.4. Refuses a steady scheme - a moving mesh needs a
    /// transient one - and inconsistent coefficients. The order is the
    /// bitwise claim the gate replays: history rotation first, then the
    /// swept volume, then the geometry recompute, then the mesh flux, and
    /// `points_old <- points` last of all. Nothing downloads, nothing
    /// allocates, nothing syncs.
    pub fn advance(&mut self, gpu: &Gpu, gm: &GpuMesh, c: DdtCoeffs) -> Result<()> {
        if c == DdtCoeffs::ZERO {
            return Err(Error::Config(
                "AleMesh::advance: a moving mesh needs a transient time scheme; \
                 a steady run must not move its mesh through this module"
                    .to_string(),
            ));
        }
        if !c.is_consistent() {
            return Err(Error::Config(format!(
                "AleMesh::advance: the coefficients {c:?} do not sum to zero, so \
                 the volume balance of the step cannot close"
            )));
        }
        self.check_shape(gm, "AleMesh::advance")?;
        let n_faces = self.n_internal_faces + self.n_boundary_faces;

        gpu.stream().memcpy_dtod(&self.v0, &mut self.v00)?;
        gpu.stream().memcpy_dtod(&gm.v, &mut self.v0)?;
        gpu.stream().memcpy_dtod(&self.swept, &mut self.swept0)?;

        let (nl_f, nl_if, nl_bf) = (
            n_faces as Label,
            self.n_internal_faces as Label,
            self.n_boundary_faces as Label,
        );
        unsafe {
            gpu.stream()
                .launch_builder(&self.k.swept_volume)
                .arg(&mut self.swept)
                .arg(&self.face_offset)
                .arg(&self.face_point)
                .arg(&self.points_old)
                .arg(&self.points)
                .arg(&nl_f)
                .launch(cfg_for(n_faces))?;
        }
        self.recompute_in_place(gpu, gm)?;
        unsafe {
            gpu.stream()
                .launch_builder(&self.k.mesh_flux)
                .arg(&mut self.phi_mesh.f)
                .arg(&mut self.phi_mesh.bf)
                .arg(&self.swept)
                .arg(&self.swept0)
                .arg(&c.a_n)
                .arg(&c.a_00)
                .arg(&nl_if)
                .arg(&nl_bf)
                .launch(cfg_for(n_faces))?;
        }
        gpu.stream().memcpy_dtod(&self.points, &mut self.points_old)?;
        Ok(())
    }

    /// Add `sign . d(psi)/dt` to the matrix, on the moving mesh: the ddt of
    /// `timescheme::fvm_ddt` with each time level carrying its OWN volume
    /// instead of one `V` for all three.
    ///
    /// SPEC-LIT 105.4. `psi0` and `psi00` are the same old levels
    /// `timescheme::fvm_ddt` reads; the volumes come from the history
    /// `advance` rotates.
    pub fn fvm_ddt(
        &self,
        gpu: &Gpu,
        a: &mut GpuLduMatrix,
        gm: &GpuMesh,
        psi0: &DevBuf<Scalar>,
        psi00: &DevBuf<Scalar>,
        c: DdtCoeffs,
        sign: Scalar,
    ) -> Result<()> {
        self.check_shape(gm, "AleMesh::fvm_ddt")?;
        if a.n_cells != gm.n_cells || a.n_internal_faces != gm.n_internal_faces {
            return Err(Error::Config(format!(
                "ale::fvm_ddt: matrix is {}x{} but the mesh is {}x{}",
                a.n_cells, a.n_internal_faces, gm.n_cells, gm.n_internal_faces
            )));
        }
        let n = gm.n_cells;
        if n == 0 || c == DdtCoeffs::ZERO {
            return Ok(());
        }
        for (b, what) in [(psi0, "psi0"), (psi00, "psi00")] {
            if b.len() < n {
                return Err(Error::Config(format!(
                    "ale::fvm_ddt: `{what}` holds {} values, {n} were needed",
                    b.len()
                )));
            }
        }
        if !c.is_consistent() {
            return Err(Error::Config(format!(
                "ale::fvm_ddt: the coefficients {c:?} do not sum to zero, so the \
                 discrete time derivative of a constant field is not zero"
            )));
        }
        let nl = n as Label;
        unsafe {
            gpu.stream()
                .launch_builder(&self.k.ddt_v)
                .arg(&mut a.diag)
                .arg(&mut a.source)
                .arg(&gm.v)
                .arg(&self.v0)
                .arg(&self.v00)
                .arg(psi0)
                .arg(psi00)
                .arg(&c.a_n)
                .arg(&c.a_0)
                .arg(&c.a_00)
                .arg(&sign)
                .arg(&nl)
                .launch(cfg_for(n))?;
        }
        Ok(())
    }

    /// `sign . d(rho psi)/dt` on the moving mesh: each level carries its own
    /// density and its own volume, which is what makes the discrete form
    /// conserve `rho psi` (SPEC-LIT 105.4).
    #[allow(clippy::too_many_arguments)]
    pub fn fvm_ddt_rho(
        &self,
        gpu: &Gpu,
        a: &mut GpuLduMatrix,
        gm: &GpuMesh,
        rho: &DevBuf<Scalar>,
        rho0: &DevBuf<Scalar>,
        rho00: &DevBuf<Scalar>,
        psi0: &DevBuf<Scalar>,
        psi00: &DevBuf<Scalar>,
        c: DdtCoeffs,
        sign: Scalar,
    ) -> Result<()> {
        self.check_shape(gm, "AleMesh::fvm_ddt_rho")?;
        if a.n_cells != gm.n_cells || a.n_internal_faces != gm.n_internal_faces {
            return Err(Error::Config(format!(
                "ale::fvm_ddt_rho: matrix is {}x{} but the mesh is {}x{}",
                a.n_cells, a.n_internal_faces, gm.n_cells, gm.n_internal_faces
            )));
        }
        let n = gm.n_cells;
        if n == 0 || c == DdtCoeffs::ZERO {
            return Ok(());
        }
        for (b, what) in [
            (rho, "rho"),
            (rho0, "rho0"),
            (rho00, "rho00"),
            (psi0, "psi0"),
            (psi00, "psi00"),
        ] {
            if b.len() < n {
                return Err(Error::Config(format!(
                    "ale::fvm_ddt_rho: `{what}` holds {} values, {n} were needed",
                    b.len()
                )));
            }
        }
        if !c.is_consistent() {
            return Err(Error::Config(format!(
                "ale::fvm_ddt_rho: the coefficients {c:?} do not sum to zero, so \
                 the discrete time derivative of a constant field is not zero"
            )));
        }
        let nl = n as Label;
        unsafe {
            gpu.stream()
                .launch_builder(&self.k.ddt_rho_v)
                .arg(&mut a.diag)
                .arg(&mut a.source)
                .arg(&gm.v)
                .arg(&self.v0)
                .arg(&self.v00)
                .arg(rho)
                .arg(rho0)
                .arg(rho00)
                .arg(psi0)
                .arg(psi00)
                .arg(&c.a_n)
                .arg(&c.a_0)
                .arg(&c.a_00)
                .arg(&sign)
                .arg(&nl)
                .launch(cfg_for(n))?;
        }
        Ok(())
    }

    /// The total volume of the mesh AS IT IS NOW, not as it was uploaded:
    /// the ascending-cell-id fold of `gm.v`'s first `n_cells` entries - the
    /// same fold `GpuMesh::upload` used, which `GpuMesh::total_volume` froze
    /// at upload (SPEC-LIT 105.1).
    pub fn total_volume(&self, gpu: &Gpu, gm: &GpuMesh) -> Result<Scalar> {
        let v = gpu.download(&gm.v)?;
        Ok(v.iter().copied().take(self.n_cells).sum())
    }
}

/// The swept volume of every face, on the HOST, in the kernel's association.
///
/// The twin `cuda/ale.cu`'s `aleSweptVolume` is held bitwise against: every
/// operation below has the association and the order of the kernel - the
/// accumulation into `x0`/`x1`, the division by `n`, the half-sums, the
/// cross products, the `(dx + da) + dc` grouping, the `(n0 + 4 nm) + n1`
/// Simpson weights, and the division by 36. SPEC-LIT 105.3.
pub fn host_swept_volumes(points_old: &[Vec3], points_new: &[Vec3], csr: &FacePointCsr) -> Vec<Scalar> {
    let n_faces = csr.offset.len().saturating_sub(1);
    let mut out = Vec::with_capacity(n_faces);
    for f in 0..n_faces {
        let b = csr.offset[f] as usize;
        let n = (csr.offset[f + 1] - csr.offset[f]) as usize;
        if n < 3 {
            out.push(0.0);
            continue;
        }
        let mut x0 = Vec3::ZERO;
        let mut x1 = Vec3::ZERO;
        for i in 0..n {
            x0 += points_old[csr.point[b + i] as usize];
            x1 += points_new[csr.point[b + i] as usize];
        }
        x0 = x0 / n as Scalar;
        x1 = x1 / n as Scalar;
        let xm = (x0 + x1) * 0.5;
        let dx = x1 - x0;

        let mut acc: Scalar = 0.0;
        for i in 0..n {
            let pa = csr.point[b + i] as usize;
            let pc = csr.point[b + ((i + 1) % n)] as usize;
            let a0 = points_old[pa];
            let c0 = points_old[pc];
            let a1 = points_new[pa];
            let c1 = points_new[pc];
            let am = (a0 + a1) * 0.5;
            let cm = (c0 + c1) * 0.5;
            let n0 = (a0 - x0).cross(c0 - x0);
            let nm = (am - xm).cross(cm - xm);
            let n1 = (a1 - x1).cross(c1 - x1);
            let dsum = (dx + (a1 - a0)) + (c1 - c0);
            let nsum = (n0 + nm * 4.0) + n1;
            acc += dsum.dot(nsum) / 36.0;
        }
        out.push(acc);
    }
    out
}

/// Component-wise bounds of a point set - the box the interior motion of
/// `gate_105a` fades out at.
pub fn point_bounds(points: &[Vec3]) -> (Vec3, Vec3) {
    let mut lo = points[0];
    let mut hi = points[0];
    for p in points {
        lo = Vec3::new(lo.x.min(p.x), lo.y.min(p.y), lo.z.min(p.z));
        hi = Vec3::new(hi.x.max(p.x), hi.y.max(p.y), hi.z.max(p.z));
    }
    (lo, hi)
}

/// The prescribed interior motion of the gate: a sinusoid that returns every
/// point on the box's boundary EXACTLY as it arrived - not `rest + 0.0 * ...`
/// but untouched - which is what makes a boundary face sweep exactly zero.
pub fn interior_sinusoid(rest: Vec3, lo: Vec3, hi: Vec3, amp: Scalar, period: Scalar, t: Scalar) -> Vec3 {
    let l = hi - lo;
    let (xi, eta, zeta) = ((rest.x - lo.x) / l.x, (rest.y - lo.y) / l.y, (rest.z - lo.z) / l.z);
    let edge = |s: Scalar| s <= 1e-9 || s >= 1.0 - 1e-9;
    if edge(xi) || edge(eta) || edge(zeta) {
        return rest;
    }
    let pi = std::f64::consts::PI as Scalar;
    let s = amp * (2.0 * pi * t / period).sin();
    Vec3::new(
        rest.x + s * (pi * xi).sin() * (2.0 * pi * eta).sin() * (pi * zeta).sin(),
        rest.y + s * (2.0 * pi * xi).sin() * (pi * eta).sin() * (pi * zeta).sin(),
        rest.z + s * (pi * xi).sin() * (pi * eta).sin() * (2.0 * pi * zeta).sin(),
    )
}

/// What `run_scl` measured. Every `worst_*` is a maximum over all steps and
/// all cells.
#[derive(Debug, Clone)]
pub struct SclRun {
    pub steps: usize,
    pub n_cells: usize,
    /// |(V^{n+1}_c - V^n_c) - sum_f s_f dV_f| / V^{n+1}_c
    pub worst_scl: Scalar,
    /// |a_n V^{n+1} + a_0 V^n + a_00 V^{n-1} - sum_f s_f phi_mesh,f| / (|a_n| V^{n+1})
    pub worst_scheme: Scalar,
    /// |(A 1 - b)_c| / (|a_n| V^{n+1}_c): ALE ddt plus Gauss upwind of
    /// phi - phi_mesh, U uniform, psi = 1
    pub worst_uniform: Scalar,
    /// v0 == V^n and v00 == V^{n-1}, bit for bit, after every step
    pub history_bitwise: bool,
    /// the worst step's |sum_c V^{n+1}_c - sum_c V^0_c| / sum_c V^0_c
    pub volume_drift: Scalar,
    /// max |phi_mesh| over boundary faces
    pub boundary_flux_max: Scalar,
    /// min V^{n+1}_c / V^0_c
    pub min_volume_ratio: Scalar,
    /// where worst_scl was taken
    pub worst_step: usize,
    pub worst_cell: usize,
}

/// The boundary fold of `fv.rs`'s residual tests: `diag[c] += ic`, `source[c]
/// += bc`. Private here so that `fv.rs`'s test helpers can stay private.
fn fold_boundary(diag: &mut [Scalar], source: &mut [Scalar], ic: &[Scalar], bc: &[Scalar], b_face_cells: &[Label]) {
    for bf in 0..ic.len() {
        let c = b_face_cells[bf] as usize;
        diag[c] += ic[bf];
        source[c] += bc[bf];
    }
}

/// `A 1` for the assembled matrix - the diagonal plus, per internal face,
/// the upper coefficient into the owner's row and the lower into the
/// neighbour's. The `amul` of `fv.rs`'s residual test at `x = 1`.
fn amul_ones(
    diag: &[Scalar],
    upper: &[Scalar],
    lower: &[Scalar],
    owner: &[Label],
    neighbour: &[Label],
    n_cells: usize,
) -> Vec<Scalar> {
    let mut y: Vec<Scalar> = diag.iter().copied().take(n_cells).collect();
    for f in 0..upper.len() {
        let o = owner[f] as usize;
        let n = neighbour[f] as usize;
        y[o] += upper[f];
        y[n] += lower[f];
    }
    y
}

/// Run the space-conservation gate body: `steps` ALE steps of the refined
/// box `rb` under `motion`, measuring, per step, the discrete space
/// conservation law, the scheme's own flux form of it, and the uniform-flow
/// residual of the ALE ddt plus Gauss upwind of `phi - phi_mesh`.
///
/// SPEC-LIT 105.5. Refuses a zero-step run and a non-positive `dt`.
pub fn run_scl(
    gpu: &Gpu,
    rb: &RefinedBox,
    motion: &dyn Fn(Vec3, Scalar) -> Vec3,
    dt: Scalar,
    steps: usize,
    scheme: DdtScheme,
) -> Result<SclRun> {
    if steps == 0 {
        return Err(Error::Config(
            "run_scl: a space-conservation run needs at least one step".to_string(),
        ));
    }
    if !(dt > 0.0) {
        return Err(Error::Config(format!(
            "run_scl: dt is {dt}; a transient run needs a positive time step"
        )));
    }
    let m = &rb.mesh;
    let csr = flatten_faces(&rb.faces);
    let gm = GpuMesh::upload(gpu, m)?;
    let mut ale = AleMesh::new(gpu, m, &gm, &rb.points, &csr)?;
    let fvk = FvKernels::new(gpu)?;
    let mut a = GpuLduMatrix::new(gpu, &gm)?;

    // psi = 1, Dirichlet on every boundary face - built exactly as fv.rs's
    // residual test builds one.
    let nbf = m.n_boundary_faces;
    let mut psi = GpuScalarField::zeros(gpu, &gm, "psi")?;
    gpu.write(&mut psi.f, &vec![1.0 as Scalar; m.n_cells])?;
    gpu.write(&mut psi.bf, &vec![1.0 as Scalar; nbf])?;
    gpu.write(&mut psi.fr, &vec![1.0 as Scalar; nbf])?;
    gpu.write(&mut psi.ref_value, &vec![1.0 as Scalar; nbf])?;
    gpu.write(&mut psi.ref_grad, &vec![0.0 as Scalar; nbf])?;
    gpu.write(&mut psi.bc_kind, &vec![BcKind::FixedValue as Label; nbf])?;

    let ones = gpu.upload(&vec![1.0 as Scalar; m.n_cells])?;
    let mut phi_rel = GpuSurfaceScalarField::zeros(gpu, &gm, "phiRel")?;
    let mut w: DevBuf<Scalar> = gpu.zeros(m.n_internal_faces)?;
    let mut bw: DevBuf<Scalar> = gpu.zeros(nbf)?;
    let u = Vec3::new(0.7, -0.4, 0.25);

    let n_faces = m.n_internal_faces + nbf;
    let owner: Vec<Label> = gpu.download(&gm.owner)?;
    let neighbour: Vec<Label> = gpu.download(&gm.neighbour)?;
    let b_face_cells: Vec<Label> = gpu.download(&gm.b_face_cells)?;

    let v_init: Vec<Scalar> = gpu.download(&gm.v)?.into_iter().take(m.n_cells).collect();
    let init_sum: Scalar = v_init.iter().copied().sum();
    let mut v_prev = v_init.clone();
    let mut v_prev2 = v_init.clone();

    let mut run = SclRun {
        steps,
        n_cells: m.n_cells,
        worst_scl: 0.0,
        worst_scheme: 0.0,
        worst_uniform: 0.0,
        history_bitwise: true,
        volume_drift: 0.0,
        boundary_flux_max: 0.0,
        min_volume_ratio: Scalar::MAX,
        worst_step: 0,
        worst_cell: 0,
    };
    let mut pts: Vec<Vec3> = rb.points.clone();

    for n in 0..steps {
        let t = (n + 1) as Scalar * dt;
        for (i, p) in rb.points.iter().enumerate() {
            pts[i] = motion(*p, t);
        }
        ale.set_points(gpu, &pts)?;
        let c = scheme.coeffs(dt, dt, n as u64)?;
        ale.advance(gpu, &gm, c)?;
        gpu.sync()?;

        let v: Vec<Scalar> = gpu.download(&gm.v)?.into_iter().take(m.n_cells).collect();
        let swept: Vec<Scalar> = gpu.download(&ale.swept)?.into_iter().take(n_faces).collect();
        let pm_f: Vec<Scalar> = gpu.download(&ale.phi_mesh.f)?.into_iter().take(m.n_internal_faces).collect();
        let pm_b: Vec<Scalar> = gpu.download(&ale.phi_mesh.bf)?.into_iter().take(nbf).collect();
        let v0: Vec<Scalar> = gpu.download(&ale.v0)?.into_iter().take(m.n_cells).collect();
        let v00: Vec<Scalar> = gpu.download(&ale.v00)?.into_iter().take(m.n_cells).collect();
        run.history_bitwise &= v0 == v_prev && v00 == v_prev2;

        // The per-cell face sums: internal faces first, ascending id, owner
        // + and neighbour -, then the boundary faces, owner +.
        let mut d_v_sum = vec![0.0 as Scalar; m.n_cells];
        let mut flux_sum = vec![0.0 as Scalar; m.n_cells];
        for f in 0..m.n_internal_faces {
            let (o, nn) = (owner[f] as usize, neighbour[f] as usize);
            d_v_sum[o] += swept[f];
            d_v_sum[nn] -= swept[f];
            flux_sum[o] += pm_f[f];
            flux_sum[nn] -= pm_f[f];
        }
        for bf in 0..nbf {
            d_v_sum[b_face_cells[bf] as usize] += swept[m.n_internal_faces + bf];
            flux_sum[b_face_cells[bf] as usize] += pm_b[bf];
            run.boundary_flux_max = run.boundary_flux_max.max(pm_b[bf].abs());
        }
        for c_id in 0..m.n_cells {
            let scl = ((v[c_id] - v_prev[c_id]) - d_v_sum[c_id]).abs() / v[c_id];
            if scl > run.worst_scl {
                run.worst_scl = scl;
                run.worst_step = n;
                run.worst_cell = c_id;
            }
            let scheme_scl = (c.a_n * v[c_id] + c.a_0 * v_prev[c_id] + c.a_00 * v_prev2[c_id]
                - flux_sum[c_id])
                .abs()
                / (c.a_n.abs() * v[c_id]);
            run.worst_scheme = run.worst_scheme.max(scheme_scl);
            run.min_volume_ratio = run.min_volume_ratio.min(v[c_id] / v_init[c_id]);
        }

        // The uniform state: phi = U.Sf - phi_mesh, assembled by the ALE ddt
        // plus Gauss upwind at psi = 1, residual A 1 - b.
        let sf: Vec<Vec3> = gpu.download(&gm.sf)?;
        let b_sf: Vec<Vec3> = gpu.download(&gm.b_sf)?;
        let mut rel_f: Vec<Scalar> = vec![0.0 as Scalar; m.n_internal_faces];
        let mut rel_b: Vec<Scalar> = vec![0.0 as Scalar; nbf];
        for f in 0..m.n_internal_faces {
            rel_f[f] = u.dot(sf[f]) - pm_f[f];
        }
        for bf in 0..nbf {
            rel_b[bf] = u.dot(b_sf[bf]) - pm_b[bf];
        }
        gpu.write(&mut phi_rel.f, &rel_f)?;
        gpu.write(&mut phi_rel.bf, &rel_b)?;

        a.zero(gpu)?;
        ale.fvm_ddt(gpu, &mut a, &gm, &ones, &ones, c, 1.0)?;
        fv::div_scheme_weights(
            gpu, &fvk, Some(&mut w), Some(&mut bw), DivScheme::Upwind, &phi_rel, &psi, None, &gm,
        )?;
        fv::fvm_div_gauss(gpu, &fvk, &mut a, &gm, &phi_rel, &w, &bw, &psi, 1.0)?;
        gpu.sync()?;

        let mut diag: Vec<Scalar> = gpu.download(&a.diag)?;
        let upper: Vec<Scalar> = gpu.download(&a.upper)?;
        let lower: Vec<Scalar> = gpu.download(&a.lower)?;
        let mut source: Vec<Scalar> = gpu.download(&a.source)?;
        let ic: Vec<Scalar> = gpu.download(&a.internal_coeffs)?;
        let bc: Vec<Scalar> = gpu.download(&a.boundary_coeffs)?;
        fold_boundary(&mut diag, &mut source, &ic, &bc, &b_face_cells);
        let ax = amul_ones(&diag, &upper, &lower, &owner, &neighbour, m.n_cells);
        for c_id in 0..m.n_cells {
            let r = (ax[c_id] - source[c_id]).abs() / (c.a_n.abs() * v[c_id]);
            run.worst_uniform = run.worst_uniform.max(r);
        }

        let v_sum: Scalar = v.iter().copied().sum();
        run.volume_drift = run.volume_drift.max(((v_sum - init_sum) / init_sum).abs());
        v_prev2 = v_prev;
        v_prev = v;
    }
    Ok(run)
}

/// Gate 105-A's own fixture and motion: the uniform box 6x5x4 of cell
/// `0.2 x 0.25 x 0.3`, every interior point on the amplitude-0.03,
/// period-0.5 sinusoid, every boundary point exactly fixed, `dt = 0.01`,
/// 100 steps. SPEC-LIT 105.5.
pub fn gate_105a(gpu: &Gpu, scheme: DdtScheme) -> Result<SclRun> {
    let rb = refined::build([6, 5, 4], Vec3::new(0.2, 0.25, 0.3), &[0u32; 120])?;
    let (lo, hi) = point_bounds(&rb.points);
    let motion = |p: Vec3, t: Scalar| interior_sinusoid(p, lo, hi, 0.03, 0.5, t);
    run_scl(gpu, &rb, &motion, 0.01, 100, scheme)
}

#[cfg(test)]
mod tests;
