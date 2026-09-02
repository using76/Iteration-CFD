// meteor-cfd - Copyright (c) 2026 주식회사 메테오시뮬레이션 (Meteo Simulation Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

//! The polyMesh emitter on the device.
//!
//! Written from:
//!   ofgpu SPEC-LIT.md sections 74, 82 and 84
//!   this crate's own `src/adapt.rs::Forest::emit`, of which every kernel in
//!     `cuda/meshemit.cu` is a restatement
//! No GPL-licensed source was consulted.
//!
//! # Why this exists
//!
//! Section 75.8 named `mesh/geometry.rs::compute` as what makes an adapt
//! expensive. Section 82.2 measured that and found it was not: the geometry
//! sweep was 31 % of a rebuild and **`Forest::emit`'s face-grouping loop was
//! 54 %**, because it allocates a `BTreeMap` per (cell, axis) and a `Vec` per
//! face. Section 82.9 specified the port and split it honestly - the topology
//! is the easy half, the point numbering is the hard one. Section 84 is the
//! port; this module drives it.
//!
//! # The claim
//!
//! **Bitwise identity with `adapt::Forest::emit`, on every mesh** -
//! the same faces, in the same order, with the same point numbers, not a
//! different but equivalent mesh. Section 82.9 explicitly allowed a permuted
//! point set and warned that whoever wrote the emitter should decide that
//! deliberately. The decision is: no. A permuted point set would silently
//! change what `io::polymesh` writes and would break section 75.2's
//! `the_forest_emitter_reproduces_the_static_generator`, and the thing that
//! makes the numbering reproducible - section 84.2's touch rank - is cheaper
//! than the argument about whether it matters.
//!
//! # What the host still does
//!
//! Four small downloads - the gap check, the two grouping diagnostics and the
//! nine counts - because a refusal has to be raised before the kernel that
//! would read past an array, and because sizing an allocation is a host act.
//! The patch names are written here, and `HostMesh::build_cell_face_maps`
//! still runs on the host for a caller that asks for a `HostMesh`. Section
//! 84.11 measures what is left, and finds that `build_cell_face_maps` is now
//! a larger piece of a rebuild than this module is.

use cudarc::driver::{CudaFunction, PushKernelArg};

use crate::device::{cfg_for, Gpu, KernelSet, BLOCK};
use crate::error::{Error, Result};
use crate::mesh::gpugeom::FacePointCsr;
use crate::mesh::{HostMesh, PatchInfo, PatchKind};
use crate::parcels::SORT_TILE;
use crate::{DevBuf, Label, Scalar, Vec3};

/// Groups per (cell, axis) face; `EM_MAXG` in `cuda/meshemit.cu`.
const MAXG: usize = 4;

/// Slots per (cell, axis); `EM_SLOTS`.
const SLOTS: usize = 5;

/// Touch ranks per cell; `EM_PER_CELL`.
const PER_CELL: usize = 3 * SLOTS * 4;

/// Blocks in the first-index reduction; `EM_RGRID`.
const RGRID: usize = 256;

/// Threads per block in the first-index reduction; `EM_RBLOCK`.
const RBLOCK: u32 = 256;

/// The six patch names the emitter writes, in emission order.
const PATCH_NAMES: [&str; 6] = ["xmin", "xmax", "ymin", "ymax", "zmin", "zmax"];

/// The kernels of `cuda/meshemit.cu`, plus the three scan kernels of
/// `cuda/parcelsort.cu` this module reuses.
pub struct MeshEmitKernels {
    base_offsets: CudaFunction,
    leaf_boxes: CudaFunction,
    voxel_owner: CudaFunction,
    face_groups: CudaFunction,
    point_ranks: CudaFunction,
    point_flags: CudaFunction,
    points: CudaFunction,
    owned_counts: CudaFunction,
    internal_faces: CudaFunction,
    boundary_flags: CudaFunction,
    boundary_faces: CudaFunction,
    totals: CudaFunction,
    first_index: CudaFunction,
    scan: Scan,
}

impl MeshEmitKernels {
    pub fn new(gpu: &Gpu) -> Result<Self> {
        let k = KernelSet::new(gpu, crate::kernels::MESHEMIT)?;
        let s = KernelSet::new(gpu, crate::kernels::PARCELSORT)?;
        Ok(Self {
            base_offsets: k.func("emitBaseOffsets")?,
            leaf_boxes: k.func("emitLeafBoxes")?,
            voxel_owner: k.func("emitVoxelOwner")?,
            face_groups: k.func("emitFaceGroups")?,
            point_ranks: k.func("emitPointRanks")?,
            point_flags: k.func("emitPointFlags")?,
            points: k.func("emitPoints")?,
            owned_counts: k.func("emitOwnedFaceCounts")?,
            internal_faces: k.func("emitInternalFaces")?,
            boundary_flags: k.func("emitBoundaryFlags")?,
            boundary_faces: k.func("emitBoundaryFaces")?,
            totals: k.func("emitTotals")?,
            first_index: k.func("emitFirstIndex")?,
            scan: Scan {
                reduce: s.func("ofsScanReduce")?,
                block_sums: s.func("ofsScanBlockSums")?,
                downsweep: s.func("ofsScanDownsweep")?,
            },
        })
    }
}

/// The exclusive scan of `cuda/parcelsort.cu`, run at a length chosen per
/// call rather than fixed at construction.
///
/// [`crate::parcels::DeviceScan`] fixes its length because a captured graph
/// freezes the launch geometry (section 66.7) and the parcel sort runs INSIDE
/// the time step. This one runs when a mesh is rebuilt, outside any capture
/// (section 84.8), and the three lengths it is asked for - the touch rank
/// space, the cells, and six times the cells - change with every adapt. Same
/// kernels, same reduce-then-scan, same absence of atomics; only the
/// bookkeeping differs, so there is no second scan to keep correct.
struct Scan {
    reduce: CudaFunction,
    block_sums: CudaFunction,
    downsweep: CudaFunction,
}

impl Scan {
    fn run(&self, gpu: &Gpu, inp: &DevBuf<i32>, out: &mut DevBuf<i32>, n: usize) -> Result<()> {
        if n == 0 {
            return Ok(());
        }
        if n > i32::MAX as usize {
            return Err(Error::Config(format!(
                "mesh::gpuemit: a scan over {n} elements exceeds the i32 the kernels \
                 count in. Rebuild with the host emitter, adapt::Forest::build, which \
                 counts in usize (SPEC-LIT section 84.7)"
            )));
        }
        let n_tiles = n.div_ceil(SORT_TILE);
        let mut sums: DevBuf<i32> = gpu.zeros(n_tiles)?;
        let cfg_tiles = cudarc::driver::LaunchConfig {
            grid_dim: (n_tiles as u32, 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let cfg_one = cudarc::driver::LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let (ni, nt) = (n as i32, n_tiles as i32);
        unsafe {
            gpu.stream()
                .launch_builder(&self.reduce)
                .arg(inp)
                .arg(&mut sums)
                .arg(&ni)
                .launch(cfg_tiles)?;
            gpu.stream()
                .launch_builder(&self.block_sums)
                .arg(&mut sums)
                .arg(&nt)
                .launch(cfg_one)?;
            gpu.stream()
                .launch_builder(&self.downsweep)
                .arg(inp)
                .arg(out)
                .arg(&sums)
                .arg(&ni)
                .launch(cfg_tiles)?;
        }
        Ok(())
    }
}

/// The leaf set, flattened into what a kernel can read.
///
/// `leaf` holds five `i32` per leaf - base, level, oct0, oct1, oct2 - in the
/// canonical order of `adapt::Leaf::key`, which is base-cell-major. The
/// device emitter relies on that: `emitBaseOffsets` finds a base cell's
/// leaves by binary search and would find a subrange of them otherwise.
pub struct LeafGrid<'a> {
    pub n: [usize; 3],
    pub d: Vec3,
    pub lmax: u32,
    /// The largest finest-grid voxel count this call may allocate for.
    ///
    /// The CALLER owns this number, because the caller owns the memory budget
    /// and this module owns none of the policy. `adapt::Forest::emit_on_device`
    /// passes `adapt::VOXEL_LIMIT`, which is what the host emitter refuses
    /// against, so the two routes refuse the same grids in the same words -
    /// and this module does not name `crate::adapt`, which
    /// `adapt::tests::no_time_loop_reaches_the_adapt` requires of everything
    /// outside `adapt` and section 84.7 records as the reason the field is
    /// here rather than a `use`.
    pub voxel_limit: usize,
    pub leaf: &'a [i32],
}

/// The emitted mesh, resident on the device.
///
/// The nine counts are host-side because sizing an allocation is a host act;
/// everything else stays where the kernels wrote it. [`Self::download`] is
/// the only thing that brings the arrays home, and section 84.11 measures
/// what that costs against what it saves.
pub struct DeviceEmit {
    pub n_cells: usize,
    pub n_internal_faces: usize,
    pub n_boundary_faces: usize,
    pub n_points: usize,
    /// Start of each patch within the flattened boundary arrays.
    pub patch_start: [usize; 6],
    pub owner: DevBuf<Label>,
    pub neighbour: DevBuf<Label>,
    pub b_face_cells: DevBuf<Label>,
    pub points: DevBuf<Vec3>,
    /// Four point ids per face, internal faces first. The face -> point CSR's
    /// `offset` is `4 * f` by construction: every face this emitter writes is
    /// a quadrilateral.
    pub face_pt: DevBuf<Label>,
}

impl DeviceEmit {
    /// The six patches, in the order and with the names the host emitter uses.
    pub fn patches(&self) -> Vec<PatchInfo> {
        let mut out = Vec::with_capacity(6);
        for p in 0..6 {
            let end = if p == 5 {
                self.n_boundary_faces
            } else {
                self.patch_start[p + 1]
            };
            out.push(PatchInfo {
                name: PATCH_NAMES[p].to_string(),
                type_name: "patch".to_string(),
                kind: PatchKind::Generic,
                start: self.patch_start[p],
                size: end - self.patch_start[p],
                nbr_patch: None,
            });
        }
        out
    }

    /// The counts and the patch list, with every array still on the device.
    ///
    /// This is what a resident rebuild needs: `gpu_geometry_resident_csr`
    /// validates against it, and `GpuMesh` uploads nothing it did not have to.
    pub fn host_shell(&self) -> HostMesh {
        HostMesh {
            n_cells: self.n_cells,
            n_internal_faces: self.n_internal_faces,
            n_boundary_faces: self.n_boundary_faces,
            n_points: self.n_points,
            patches: self.patches(),
            ..Default::default()
        }
    }

    /// Bring the topology home: the mesh, its points, and the face -> point
    /// CSR, with the cell -> face maps built.
    ///
    /// Six downloads, no `Vec<Vec<Label>>`. The host emitter's face list is
    /// one heap allocation per face and section 82.2 measures what that costs;
    /// [`faces_from_csr`] rebuilds it for the callers that still want it, and
    /// says so at the call site rather than here.
    pub fn download(&self, gpu: &Gpu) -> Result<(HostMesh, Vec<Vec3>, FacePointCsr)> {
        let mut m = self.host_shell();
        m.owner = take(gpu, &self.owner, self.n_internal_faces)?;
        m.neighbour = take(gpu, &self.neighbour, self.n_internal_faces)?;
        m.b_face_cells = take(gpu, &self.b_face_cells, self.n_boundary_faces)?;
        let mut points = gpu.download(&self.points)?;
        points.truncate(self.n_points);

        let n_faces = self.n_internal_faces + self.n_boundary_faces;
        let csr = FacePointCsr {
            offset: (0..=n_faces).map(|f| 4 * f as Label).collect(),
            point: take(gpu, &self.face_pt, 4 * n_faces)?,
        };
        m.build_cell_face_maps();
        Ok((m, points, csr))
    }
}

fn take(gpu: &Gpu, b: &DevBuf<Label>, n: usize) -> Result<Vec<Label>> {
    let mut v = gpu.download(b)?;
    v.truncate(n);
    Ok(v)
}

/// The host emitter's `Vec<Vec<Label>>` face list, rebuilt from the CSR.
///
/// One heap allocation per face. It exists for the callers that predate the
/// CSR - `mesh::geometry::compute` and the bitwise gate - and NOT for the
/// resident route, which hands the CSR straight to the geometry sweep.
pub fn faces_from_csr(csr: &FacePointCsr) -> Vec<Vec<Label>> {
    (0..csr.offset.len().saturating_sub(1))
        .map(|f| {
            let (a, b) = (csr.offset[f] as usize, csr.offset[f + 1] as usize);
            csr.point[a..b].to_vec()
        })
        .collect()
}

/// Emit the polyMesh a leaf set is, on the device.
///
/// SPEC-LIT section 84. The refusals below are the host emitter's, word for
/// word, because a caller that switches emitters must not get a different
/// diagnosis of the same broken leaf set - the same rule
/// `mesh::gpugeom::refuse_a_cell_with_no_faces` follows for the sweep.
pub fn emit_device(gpu: &Gpu, k: &MeshEmitKernels, g: &LeafGrid<'_>) -> Result<DeviceEmit> {
    let (nx, ny, nz) = (g.n[0], g.n[1], g.n[2]);
    let n_base = nx * ny * nz;
    if g.leaf.len() % 5 != 0 {
        return Err(Error::Mesh(format!(
            "mesh::gpuemit: the packed leaf array holds {} entries, which is not \
             five per leaf",
            g.leaf.len()
        )));
    }
    let n_cells = g.leaf.len() / 5;
    if n_cells == 0 {
        return Err(Error::Mesh(
            "mesh::gpuemit: a forest with no leaves emits no mesh".to_string(),
        ));
    }

    let fac = 1usize << g.lmax;
    let vn = [nx * fac, ny * fac, nz * fac];
    let nvox = vn[0]
        .checked_mul(vn[1])
        .and_then(|a| a.checked_mul(vn[2]))
        .ok_or_else(|| Error::Mesh("the finest grid overflows a usize".to_string()))?;
    if nvox > g.voxel_limit {
        return Err(Error::Mesh(format!(
            "a {}x{}x{} base grid at level {} needs {nvox} finest-grid voxels, \
             past this module's limit of {}",
            nx, ny, nz, g.lmax, g.voxel_limit
        )));
    }

    let pn = [vn[0] + 1, vn[1] + 1, vn[2] + 1];
    let n_site = pn[0] * pn[1] * pn[2];
    let n_rank = n_cells * PER_CELL;

    // The dense touch-rank space is what makes the point numbering a scan, and
    // it is 60 ints per cell. Refuse by name rather than wrap an i32 - the
    // host emitter has no such space and is the alternative. SPEC-LIT 84.7.
    if n_rank > i32::MAX as usize || n_site > i32::MAX as usize {
        return Err(Error::Mesh(format!(
            "mesh::gpuemit: {n_cells} cells need a touch-rank space of {n_rank} and a \
             point grid of {n_site}, and the kernels count in i32. Rebuild with the \
             host emitter, adapt::Forest::build, whose point numbering is sequential \
             and needs neither (SPEC-LIT section 84.7)"
        )));
    }

    let cf = |n: usize| cfg_for(n.max(1));
    let (vnx, vny, vnz) = (vn[0] as i32, vn[1] as i32, vn[2] as i32);
    let (nlx, nly) = (nx as i32, ny as i32);
    let faci = fac as i32;
    let ncl = n_cells as i32;

    // ---- the leaf boxes and the voxel ownership map ------------------------
    let d_leaf = gpu.upload(g.leaf)?;
    let mut base_off: DevBuf<i32> = gpu.zeros(n_base + 1)?;
    let mut lo: DevBuf<i32> = gpu.zeros(3 * n_cells)?;
    let mut hi: DevBuf<i32> = gpu.zeros(3 * n_cells)?;
    let mut owner_of: DevBuf<i32> = gpu.zeros(nvox)?;

    let nbase_i = n_base as i32;
    let nvox_i = nvox as i32;
    unsafe {
        gpu.stream()
            .launch_builder(&k.base_offsets)
            .arg(&d_leaf)
            .arg(&ncl)
            .arg(&nbase_i)
            .arg(&mut base_off)
            .launch(cf(n_base + 1))?;
        gpu.stream()
            .launch_builder(&k.leaf_boxes)
            .arg(&d_leaf)
            .arg(&ncl)
            .arg(&nlx)
            .arg(&nly)
            .arg(&faci)
            .arg(&mut lo)
            .arg(&mut hi)
            .launch(cf(n_cells))?;
    }
    unsafe {
        gpu.stream()
            .launch_builder(&k.voxel_owner)
            .arg(&base_off)
            .arg(&lo)
            .arg(&hi)
            .arg(&vnx)
            .arg(&vny)
            .arg(&faci)
            .arg(&nlx)
            .arg(&nly)
            .arg(&nvox_i)
            .arg(&mut owner_of)
            .launch(cf(nvox))?;
    }

    // A voxel no leaf claimed means the leaf set has a gap, and the grouping
    // kernel would index the leaf arrays with -1. Asked and answered before
    // anything reads it, in the host emitter's words.
    if let Some(v) = first_index(gpu, k, &owner_of, nvox, 0)? {
        return Err(Error::Mesh(format!(
            "voxel {v} of the finest grid belongs to no leaf; the leaf set \
             overlaps itself somewhere and leaves a gap here"
        )));
    }

    // ---- the face grouping ------------------------------------------------
    let mut slot_info: DevBuf<i32> = gpu.zeros(3 * n_cells)?;
    let mut grp_nb: DevBuf<i32> = gpu.zeros(3 * n_cells * MAXG)?;
    let mut bad_nb: DevBuf<i32> = gpu.zeros(3 * n_cells)?;
    let mut bad_many: DevBuf<i32> = gpu.zeros(3 * n_cells)?;
    unsafe {
        gpu.stream()
            .launch_builder(&k.face_groups)
            .arg(&owner_of)
            .arg(&lo)
            .arg(&hi)
            .arg(&vnx)
            .arg(&vny)
            .arg(&vnz)
            .arg(&ncl)
            .arg(&mut slot_info)
            .arg(&mut grp_nb)
            .arg(&mut bad_nb)
            .arg(&mut bad_many)
            .launch(cf(3 * n_cells))?;
    }

    if let Some(id) = first_index(gpu, k, &bad_many, 3 * n_cells, 2)? {
        let (c, axis) = (id / 3, id % 3);
        return Err(Error::Mesh(format!(
            "mesh::gpuemit: cell {c}'s +{axis} face is shared with more than four \
             leaves, which 2:1 balance forbids and this emitter holds four of in \
             registers. Rebuild with the host emitter, adapt::Forest::build, which \
             groups an unbounded number of neighbours and will name the pair it \
             refuses (SPEC-LIT section 84.7)"
        )));
    }
    if let Some(id) = first_index(gpu, k, &bad_nb, 3 * n_cells, 1)? {
        let (c, axis) = (id / 3, id % 3);
        let nb = gpu.download(&bad_nb)?[id];
        return Err(Error::Mesh(format!(
            "cells {c} and {nb} share a non-rectangular region on axis \
             {axis}; the leaf set is not 2:1 balanced"
        )));
    }

    // ---- the point numbering ----------------------------------------------
    let mut min_rank: DevBuf<u32> = gpu.zeros(n_site)?;
    let nsite_i = n_site as i32;
    unsafe {
        gpu.stream()
            .launch_builder(&k.point_ranks)
            .arg(&owner_of)
            .arg(&lo)
            .arg(&hi)
            .arg(&slot_info)
            .arg(&grp_nb)
            .arg(&vnx)
            .arg(&vny)
            .arg(&vnz)
            .arg(&nsite_i)
            .arg(&mut min_rank)
            .launch(cf(n_site))?;
    }

    let mut pflag: DevBuf<i32> = gpu.zeros(n_rank)?;
    let mut pid: DevBuf<i32> = gpu.zeros(n_rank)?;
    let nrank_i = n_rank as i32;
    unsafe {
        gpu.stream()
            .launch_builder(&k.point_flags)
            .arg(&lo)
            .arg(&hi)
            .arg(&slot_info)
            .arg(&grp_nb)
            .arg(&min_rank)
            .arg(&vnx)
            .arg(&vny)
            .arg(&nrank_i)
            .arg(&mut pflag)
            .launch(cf(n_rank))?;
    }
    k.scan.run(gpu, &pflag, &mut pid, n_rank)?;

    // ---- the internal faces, and the boundary faces -----------------------
    let mut own_cnt: DevBuf<i32> = gpu.zeros(n_cells)?;
    let mut own_off: DevBuf<i32> = gpu.zeros(n_cells)?;
    unsafe {
        gpu.stream()
            .launch_builder(&k.owned_counts)
            .arg(&owner_of)
            .arg(&lo)
            .arg(&hi)
            .arg(&slot_info)
            .arg(&grp_nb)
            .arg(&vnx)
            .arg(&vny)
            .arg(&ncl)
            .arg(&mut own_cnt)
            .launch(cf(n_cells))?;
    }
    k.scan.run(gpu, &own_cnt, &mut own_off, n_cells)?;

    let mut bflag: DevBuf<i32> = gpu.zeros(6 * n_cells)?;
    let mut boff: DevBuf<i32> = gpu.zeros(6 * n_cells)?;
    unsafe {
        gpu.stream()
            .launch_builder(&k.boundary_flags)
            .arg(&lo)
            .arg(&hi)
            .arg(&vnx)
            .arg(&vny)
            .arg(&vnz)
            .arg(&ncl)
            .arg(&mut bflag)
            .launch(cf(6 * n_cells))?;
    }
    k.scan.run(gpu, &bflag, &mut boff, 6 * n_cells)?;

    let mut totals: DevBuf<i32> = gpu.zeros(9)?;
    unsafe {
        gpu.stream()
            .launch_builder(&k.totals)
            .arg(&pid)
            .arg(&pflag)
            .arg(&nrank_i)
            .arg(&own_off)
            .arg(&own_cnt)
            .arg(&ncl)
            .arg(&boff)
            .arg(&bflag)
            .arg(&mut totals)
            .launch(cfg_for(1))?;
    }
    let t = gpu.download(&totals)?;
    let n_points = t[0].max(0) as usize;
    let n_if = t[1].max(0) as usize;
    let n_bf = t[2].max(0) as usize;
    let mut patch_start = [0usize; 6];
    for (p, s) in patch_start.iter_mut().enumerate() {
        *s = t[3 + p].max(0) as usize;
    }

    let n_faces = n_if + n_bf;
    let mut out = DeviceEmit {
        n_cells,
        n_internal_faces: n_if,
        n_boundary_faces: n_bf,
        n_points,
        patch_start,
        owner: gpu.zeros(n_if.max(1))?,
        neighbour: gpu.zeros(n_if.max(1))?,
        b_face_cells: gpu.zeros(n_bf.max(1))?,
        points: gpu.zeros(n_points.max(1))?,
        face_pt: gpu.zeros((4 * n_faces).max(1))?,
    };

    let (hx, hy, hz) = (
        g.d.x / fac as Scalar,
        g.d.y / fac as Scalar,
        g.d.z / fac as Scalar,
    );
    let nif_i = n_if as i32;
    unsafe {
        gpu.stream()
            .launch_builder(&k.points)
            .arg(&min_rank)
            .arg(&pid)
            .arg(&nsite_i)
            .arg(&vnx)
            .arg(&vny)
            .arg(&hx)
            .arg(&hy)
            .arg(&hz)
            .arg(&mut out.points)
            .launch(cf(n_site))?;
        gpu.stream()
            .launch_builder(&k.internal_faces)
            .arg(&owner_of)
            .arg(&lo)
            .arg(&hi)
            .arg(&slot_info)
            .arg(&grp_nb)
            .arg(&own_off)
            .arg(&min_rank)
            .arg(&pid)
            .arg(&vnx)
            .arg(&vny)
            .arg(&ncl)
            .arg(&mut out.owner)
            .arg(&mut out.neighbour)
            .arg(&mut out.face_pt)
            .launch(cf(n_cells))?;
    }
    unsafe {
        gpu.stream()
            .launch_builder(&k.boundary_faces)
            .arg(&lo)
            .arg(&hi)
            .arg(&bflag)
            .arg(&boff)
            .arg(&min_rank)
            .arg(&pid)
            .arg(&vnx)
            .arg(&vny)
            .arg(&ncl)
            .arg(&nif_i)
            .arg(&mut out.b_face_cells)
            .arg(&mut out.face_pt)
            .launch(cf(6 * n_cells))?;
    }

    Ok(out)
}

/// The smallest index at which `a` satisfies the test, or `None`.
///
/// `mode` 0 is `a[i] < 0`, 1 is `a[i] >= 0`, 2 is `a[i] != 0`. A fixed grid,
/// a fixed reduction tree and a host-side min over the partials: the answer
/// does not depend on how the blocks were scheduled, which is why a refusal
/// can name the same cell the host emitter would have named.
fn first_index(
    gpu: &Gpu,
    k: &MeshEmitKernels,
    a: &DevBuf<i32>,
    n: usize,
    mode: i32,
) -> Result<Option<usize>> {
    if n == 0 {
        return Ok(None);
    }
    let mut partial: DevBuf<i32> = gpu.zeros(RGRID)?;
    let ni = n as i32;
    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: (RGRID as u32, 1, 1),
        block_dim: (RBLOCK, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        gpu.stream()
            .launch_builder(&k.first_index)
            .arg(a)
            .arg(&ni)
            .arg(&mode)
            .arg(&mut partial)
            .launch(cfg)?;
    }
    let p = gpu.download(&partial)?;
    let best = p.into_iter().min().unwrap_or(i32::MAX);
    Ok(if best == i32::MAX {
        None
    } else {
        Some(best as usize)
    })
}

#[cfg(test)]
mod tests;
