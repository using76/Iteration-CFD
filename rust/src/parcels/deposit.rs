// meteor-cfd - Copyright (c) 2026 주식회사 메테오시뮬레이션 (Meteo Simulation Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

//! The parcel sort, the per-cell CSR, and gather-shaped deposition -
//! SPEC-LIT `SPEC-LIT.md` section 67.
//!
//! [`Parcels`](super::Parcels) (S66) moves parcels and never touches a cell
//! field. Every Lagrangian source term that will ever reach the gas has the
//! shape `phi[cell[p]] += n_p w_p`, which written one thread per parcel is a
//! **scatter** and needs `atomicAdd(double*)`. Double-precision atomic
//! addition is not associative, so the summation order - the hardware's
//! scheduling order - would change the last bits of every coupled source,
//! hence the matrix, hence the Krylov iteration count, hence the answer.
//!
//! This module is the transpose. It sorts the pool on the 64+`cellBits`-bit
//! **total order** `(cell, uid)`, reads a per-cell CSR of parcel indices
//! straight out of the sorted keys, and deposits by walking each cell's own
//! segment in one thread - structurally identical to the cell -> face gather
//! S1 already uses for matrix assembly. There is no f64 atomic anywhere in
//! it, and there must never be one.
//!
//! Written from:
//!   N. Satish, M. Harris, M. Garland, *Designing efficient sorting
//!     algorithms for manycore GPUs*, IEEE IPDPS 2009,
//!     DOI `10.1109/IPDPS.2009.5161005` - the three-phase radix pass: a
//!     per-block digit histogram, one global exclusive scan over the
//!     block-by-digit counters, and a stable scatter. The paper was read; no
//!     implementation of it was opened
//!   D. Merrill, A. Grimshaw, *Parallel scan for stream architectures*,
//!     University of Virginia Technical Report CS2009-14 - the
//!     reduce-then-scan decomposition of [`DeviceScan`]
//!   G. E. Blelloch, *Prefix sums and their applications*, CMU-CS-90-190
//!     (1990) - the exclusive scan and its work-efficiency argument
//!   W. D. Hillis, G. L. Steele Jr., *Commun. ACM* 29(12) (1986) 1170,
//!     DOI `10.1145/7902.7903` - the in-block scan network, chosen because
//!     its shape depends on `blockDim` and nothing else
//!   C. T. Crowe, M. P. Sharma, D. E. Stock, *The particle-source-in-cell
//!     (PSI-CELL) model for gas-droplet flows*, J. Fluids Eng. 99 (1977) 325,
//!     DOI `10.1115/1.3448756` - the deposition itself
//!   ofgpu `SPEC-LIT.md` S67 - the section this module implements; S66 for
//!     the pool and the identity, S1 for the CSR shape it copies
//!
//! No GPL-licensed source was consulted, and in particular OpenFOAM's
//! `src/lagrangian` tree - which contains the obvious reference
//! implementation of a per-cell parcel grouping, and is GPL-3.0 - was not
//! opened.
//!
//! # Why the sort is written as kernels rather than called from CUB
//!
//! `build.rs` compiles each `.cu` with `nvcc --cubin` and loads the CUBIN
//! through `cudarc`. There is **no host-side translation unit in this
//! project**, so `cub::DeviceRadixSort` and `cub::DeviceScan` - which are
//! host functions that launch kernels - are not callable. Reaching them would
//! mean adding a host `.cu`, linking it into the Rust binary, and mixing the
//! CUDA *runtime* API into a process whose context is owned by the *driver*
//! API through `cudarc`. That is a build change with a context-ownership
//! hazard at the end of it, in exchange for a sort. S67.9 states what the
//! choice costs instead of hiding it: nine to twelve radix passes, eleven on
//! a million-cell mesh, where a tuned library would use six or seven - and
//! measured, 0.46 ms of sort for a million-slot pool.

use cudarc::driver::{CudaFunction, LaunchConfig, PushKernelArg};

use crate::device::{cfg_for, DevBuf, Gpu, KernelSet, BLOCK};
use crate::error::{Error, Result};
use crate::mesh::GpuMesh;
use crate::{Label, Scalar};

use super::Parcels;

#[cfg(test)]
mod tests;

/// Items per tile in the scan and the radix sort, mirroring
/// `OFS_TILE = OFS_BLOCK*OFS_ITEMS` in `cuda/parcelsort.cu`. The parcel pool
/// is padded up to a multiple of this so that every radix launch covers whole
/// tiles and no kernel needs a bounds test on the hot path.
pub const SORT_TILE: usize = 256 * 4;

/// Radix digit width in bits, mirroring `OFS_RADIX_BITS`.
pub const RADIX_BITS: u32 = 8;

/// Buckets per pass, `2^RADIX_BITS`.
pub const RADIX_DIGITS: usize = 1 << RADIX_BITS;

/// Passes needed for the 64-bit identity (S66.9). The identity is a bijection
/// on the full 64 bits, so none of them can be skipped.
pub const UID_PASSES: u32 = 64 / RADIX_BITS;

// ==========================================================================
//  SPEC-LIT (67.2): the device exclusive scan
// ==========================================================================

/// A deterministic device-wide exclusive scan over `i32`.
///
/// The crate did not have one: `cf_offset` is built on the host in
/// `HostMesh`, and everything else that needed a prefix sum had it at setup.
/// The radix sort needs one *inside the time step*, so here it is, sized once
/// and reused - S67.12 names compaction and injection slot assignment as the
/// next two callers.
///
/// Three kernels, Merrill & Grimshaw's reduce-then-scan: sum each tile, scan
/// the tile sums in one block, then re-scan each tile on top of its offset.
/// It reads the input twice where a decoupled-look-back scan reads it once -
/// but look-back needs an atomically assigned tile order, and a tile order
/// handed out by an atomic is exactly the scheduling dependence this whole
/// module exists to keep out. Integer addition is associative, the network's
/// shape is fixed by `BLOCK`, and there is no atomic: the result is a pure
/// function of the input.
pub struct DeviceScan {
    reduce: CudaFunction,
    block_sums: CudaFunction,
    downsweep: CudaFunction,
    /// Tile sums, one per tile. Allocated once, at setup.
    sums: DevBuf<i32>,
    n: usize,
    n_tiles: usize,
    cfg_tiles: LaunchConfig,
    cfg_one: LaunchConfig,
}

impl DeviceScan {
    /// Size the scan for exactly `n` elements. `n` is fixed for the object's
    /// life, because a captured graph freezes the launch geometry and there
    /// is no update path (S66.7).
    pub fn new(gpu: &Gpu, n: usize) -> Result<Self> {
        if n == 0 {
            return Err(Error::Config(
                "parcels/scan: a scan over zero elements is the absence of a scan, \
                 not a mode of it (SPEC-LIT S67.2)"
                    .to_string(),
            ));
        }
        if n > i32::MAX as usize {
            return Err(Error::Config(format!(
                "parcels/scan: {n} elements exceeds the i32 the kernels count in"
            )));
        }
        let k = KernelSet::new(gpu, crate::kernels::PARCELSORT)?;
        let n_tiles = n.div_ceil(SORT_TILE);
        Ok(Self {
            reduce: k.func("ofsScanReduce")?,
            block_sums: k.func("ofsScanBlockSums")?,
            downsweep: k.func("ofsScanDownsweep")?,
            sums: gpu.zeros(n_tiles)?,
            n,
            n_tiles,
            cfg_tiles: LaunchConfig {
                grid_dim: (n_tiles as u32, 1, 1),
                block_dim: (BLOCK, 1, 1),
                shared_mem_bytes: 0,
            },
            cfg_one: LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (BLOCK, 1, 1),
                shared_mem_bytes: 0,
            },
        })
    }

    pub fn len(&self) -> usize {
        self.n
    }

    pub fn is_empty(&self) -> bool {
        false
    }

    /// `out[i] = sum_{j < i} inp[j]`. `inp` and `out` must both hold at least
    /// `self.len()` elements and must not alias.
    pub fn run(&mut self, gpu: &Gpu, inp: &DevBuf<i32>, out: &mut DevBuf<i32>) -> Result<()> {
        if inp.len() < self.n || out.len() < self.n {
            return Err(Error::Config(format!(
                "parcels/scan: sized for {} elements but was given {} in and {} out",
                self.n,
                inp.len(),
                out.len()
            )));
        }
        let n = self.n as i32;
        let n_tiles = self.n_tiles as i32;
        let cfg_tiles = self.cfg_tiles;
        let cfg_one = self.cfg_one;

        let f = self.reduce.clone();
        let sums = &mut self.sums;
        unsafe {
            gpu.stream()
                .launch_builder(&f)
                .arg(inp)
                .arg(&mut *sums)
                .arg(&n)
                .launch(cfg_tiles)?;
        }

        let f = self.block_sums.clone();
        let sums = &mut self.sums;
        unsafe {
            gpu.stream()
                .launch_builder(&f)
                .arg(&mut *sums)
                .arg(&n_tiles)
                .launch(cfg_one)?;
        }

        let f = self.downsweep.clone();
        let sums = &self.sums;
        unsafe {
            gpu.stream()
                .launch_builder(&f)
                .arg(inp)
                .arg(out)
                .arg(sums)
                .arg(&n)
                .launch(cfg_tiles)?;
        }
        Ok(())
    }
}

// ==========================================================================
//  Kernels
// ==========================================================================

struct DepositKernels {
    init_uid: CudaFunction,
    load_cell: CudaFunction,
    histogram: CudaFunction,
    scatter: CudaFunction,
    csr_offsets: CudaFunction,
    deposit: CudaFunction,
}

impl DepositKernels {
    fn new(gpu: &Gpu) -> Result<Self> {
        let k = KernelSet::new(gpu, crate::kernels::PARCELSORT)?;
        Ok(Self {
            init_uid: k.func("parcelSortInitUid")?,
            load_cell: k.func("parcelSortLoadCell")?,
            histogram: k.func("parcelRadixHistogram")?,
            scatter: k.func("parcelRadixScatter")?,
            csr_offsets: k.func("parcelCsrOffsets")?,
            deposit: k.func("parcelDeposit")?,
        })
    }
}

// ==========================================================================
//  What comes back to the host
// ==========================================================================

/// A host copy of the per-cell CSR - what the S67.10 gates read.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ParcelCsrSnapshot {
    /// `[n_cells + 1]`, the exclusive prefix of parcels per cell.
    /// `offset[n_cells]` is the number of live parcels.
    pub offset: Vec<Label>,
    /// The parcel slot indices, grouped by cell and ordered by identity
    /// within each cell. Entries at or past `offset[n_cells]` are the dead
    /// and padding slots and mean nothing.
    pub index: Vec<Label>,
    /// The sorted cell keys, one per entry of `index`. A dead, free or
    /// padding slot carries `n_cells`.
    pub cell_key: Vec<u64>,
    pub n_cells: usize,
    /// `offset[n_cells]`.
    pub n_live: usize,
}

impl ParcelCsrSnapshot {
    /// The live slot indices in canonical `(cell, uid)` order.
    #[must_use]
    pub fn live_order(&self) -> &[Label] {
        &self.index[..self.n_live]
    }

    /// The cell each entry of [`Self::live_order`] belongs to.
    #[must_use]
    pub fn live_cells(&self) -> Vec<Label> {
        self.cell_key[..self.n_live].iter().map(|&k| k as Label).collect()
    }
}

/// A host copy of the deposited per-cell fields - SPEC-LIT (67.6).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct DepositSnapshot {
    /// Parcels in the cell. Sums to the live parcel count *exactly*: it is an
    /// integer count and every live parcel is in exactly one segment.
    pub count: Vec<i32>,
    /// `sum n_p` - how many physical droplets the cell holds.
    pub weight: Vec<Scalar>,
    /// `(1/V_P) sum n_p (pi/6) d^3` - the dispersed-phase volume fraction.
    pub volume_fraction: Vec<Scalar>,
    /// `sum n_p rho_l (pi/6) d^3`, kg of liquid in the cell.
    pub mass: Vec<Scalar>,
}

impl DepositSnapshot {
    /// Total parcels deposited. Exact: an integer sum.
    #[must_use]
    pub fn total_count(&self) -> i64 {
        self.count.iter().map(|&c| i64::from(c)).sum()
    }

    /// `sum_P weight[P]`, summed in ascending cell order so that the answer
    /// is a stated, reproducible one rather than whatever order a `sum()`
    /// happened to use.
    #[must_use]
    pub fn total_weight(&self) -> Scalar {
        let mut s = 0.0 as Scalar;
        for &w in &self.weight {
            s += w;
        }
        s
    }

    /// `sum_P mass[P]`, kg, in ascending cell order.
    #[must_use]
    pub fn total_mass(&self) -> Scalar {
        let mut s = 0.0 as Scalar;
        for &m in &self.mass {
            s += m;
        }
        s
    }
}

// ==========================================================================
//  SPEC-LIT (67.3)-(67.6): the sort, the CSR and the gather
// ==========================================================================

/// The scratch, the CSR and the deposited fields for one parcel pool.
///
/// Held apart from [`Parcels`](super::Parcels) rather than inside it, because
/// its scratch is `~26` bytes per slot on top of the pool's own and a case
/// that only tracks parcels should not pay for it. Everything it allocates is
/// allocated **once**, at construction: nothing in [`Self::update`] allocates,
/// synchronises or reads back, so the whole sequence captures into a CUDA
/// graph exactly as it stands.
pub struct ParcelDeposition<'m> {
    m: &'m GpuMesh,
    k: DepositKernels,
    scan: DeviceScan,

    capacity: usize,
    /// `capacity` rounded up to a whole number of [`SORT_TILE`]s.
    n_pad: usize,
    /// Radix blocks, `n_pad / SORT_TILE`.
    nb: usize,
    /// `RADIX_DIGITS * nb`.
    n_counters: usize,
    /// Passes over the cell key. The sentinel is `n_cells`, so this is
    /// `ceil(bits(n_cells) / RADIX_BITS)` and never more than four.
    n_cell_passes: u32,
    /// Whether the sorted result lands in the `a` buffers. A setup constant,
    /// because `UID_PASSES` is even: it is `n_cell_passes` that decides. It
    /// must be a constant - a ping-pong that rotated between steps would move
    /// the answer out from under a captured graph.
    final_is_a: bool,

    key_a: DevBuf<u64>,
    key_b: DevBuf<u64>,
    idx_a: DevBuf<Label>,
    idx_b: DevBuf<Label>,
    counters: DevBuf<i32>,
    counters_scan: DevBuf<i32>,

    pc_offset: DevBuf<Label>,

    count: DevBuf<i32>,
    weight: DevBuf<Scalar>,
    volume_fraction: DevBuf<Scalar>,
    mass: DevBuf<Scalar>,

    cfg_pad: LaunchConfig,
    cfg_radix: LaunchConfig,
    cfg_offsets: LaunchConfig,
    cfg_cells: LaunchConfig,
}

impl<'m> ParcelDeposition<'m> {
    /// Size everything against `p`'s pool and mesh.
    pub fn new(gpu: &Gpu, p: &Parcels<'m>) -> Result<Self> {
        let m = p.m;
        let capacity = p.ctrl.capacity;
        let n_cells = m.n_cells;

        if n_cells == 0 {
            return Err(Error::Config(
                "parcels/deposition: the mesh has no cells".to_string(),
            ));
        }
        // The dead/padding sentinel is `n_cells` itself, so `n_cells` has to
        // be a legal key AND `n_cells + 1` a legal offset array length.
        if n_cells >= i32::MAX as usize {
            return Err(Error::Config(format!(
                "parcels/deposition: {n_cells} cells, and the (67.5) sentinel key is \
                 n_cells itself, which must fit the i32 the CSR indexes with"
            )));
        }

        let n_pad = capacity.div_ceil(SORT_TILE) * SORT_TILE;
        if n_pad > i32::MAX as usize {
            return Err(Error::Config(format!(
                "parcels/deposition: capacity {capacity} pads to {n_pad}, above the i32 \
                 the sort indexes with"
            )));
        }
        let nb = n_pad / SORT_TILE;
        let n_counters = RADIX_DIGITS * nb;

        // (67.3): passes over the cell key. The largest key is the sentinel
        // `n_cells`, so this covers every value the key can take and not one
        // pass more - a 4096-cell mesh costs two passes, not four.
        let bits = (usize::BITS - n_cells.leading_zeros()).max(1);
        let n_cell_passes = bits.div_ceil(RADIX_BITS);

        Ok(Self {
            m,
            k: DepositKernels::new(gpu)?,
            scan: DeviceScan::new(gpu, n_counters)?,
            capacity,
            n_pad,
            nb,
            n_counters,
            n_cell_passes,
            final_is_a: n_cell_passes % 2 == 0,

            key_a: gpu.zeros(n_pad)?,
            key_b: gpu.zeros(n_pad)?,
            idx_a: gpu.zeros(n_pad)?,
            idx_b: gpu.zeros(n_pad)?,
            counters: gpu.zeros(n_counters)?,
            counters_scan: gpu.zeros(n_counters)?,

            pc_offset: gpu.zeros(n_cells + 1)?,

            count: gpu.zeros(n_cells)?,
            weight: gpu.zeros(n_cells)?,
            volume_fraction: gpu.zeros(n_cells)?,
            mass: gpu.zeros(n_cells)?,

            cfg_pad: cfg_for(n_pad),
            cfg_radix: LaunchConfig {
                grid_dim: (nb as u32, 1, 1),
                block_dim: (BLOCK, 1, 1),
                shared_mem_bytes: 0,
            },
            cfg_offsets: cfg_for(n_cells + 1),
            cfg_cells: cfg_for(n_cells),
        })
    }

    /// Radix passes the sort runs per rebuild: eight over the identity plus
    /// [`Self::cell_passes`] over the cell key.
    pub fn passes(&self) -> u32 {
        UID_PASSES + self.n_cell_passes
    }

    pub fn cell_passes(&self) -> u32 {
        self.n_cell_passes
    }

    /// Slots the sort actually covers - `capacity` rounded up to a tile.
    pub fn padded_capacity(&self) -> usize {
        self.n_pad
    }

    /// Device bytes this object holds, so a case can be told before it runs
    /// out of them.
    pub fn device_bytes(&self) -> usize {
        let n_cells = self.m.n_cells;
        2 * self.n_pad * std::mem::size_of::<u64>()
            + 2 * self.n_pad * std::mem::size_of::<Label>()
            + 2 * self.n_counters * std::mem::size_of::<i32>()
            + (n_cells + 1) * std::mem::size_of::<Label>()
            + n_cells * std::mem::size_of::<i32>()
            + 3 * n_cells * std::mem::size_of::<Scalar>()
            + self.n_counters.div_ceil(SORT_TILE) * std::mem::size_of::<i32>()
    }

    // ---- the device-side arrays, for whatever consumes them next ------

    /// `[n_cells + 1]`, the CSR row pointer.
    pub fn offsets(&self) -> &DevBuf<Label> {
        &self.pc_offset
    }

    /// The CSR column array: parcel slot indices in `(cell, uid)` order.
    pub fn index(&self) -> &DevBuf<Label> {
        if self.final_is_a {
            &self.idx_a
        } else {
            &self.idx_b
        }
    }

    /// The sorted cell keys, parallel to [`Self::index`].
    pub fn sorted_keys(&self) -> &DevBuf<u64> {
        if self.final_is_a {
            &self.key_a
        } else {
            &self.key_b
        }
    }

    pub fn count(&self) -> &DevBuf<i32> {
        &self.count
    }

    pub fn weight(&self) -> &DevBuf<Scalar> {
        &self.weight
    }

    pub fn volume_fraction(&self) -> &DevBuf<Scalar> {
        &self.volume_fraction
    }

    pub fn mass(&self) -> &DevBuf<Scalar> {
        &self.mass
    }

    // ---- the step -----------------------------------------------------

    /// Sort, build the CSR and deposit. Capturable as it stands.
    pub fn update(&mut self, gpu: &Gpu, p: &Parcels<'m>) -> Result<()> {
        self.build(gpu, p)?;
        self.deposit(gpu, p)
    }

    /// SPEC-LIT (67.3)-(67.5): sort the pool on `(cell, uid)` and read the
    /// per-cell CSR out of the sorted keys.
    pub fn build(&mut self, gpu: &Gpu, p: &Parcels<'m>) -> Result<()> {
        self.check(p)?;

        let capacity = self.capacity as i32;
        let n_pad = self.n_pad as i32;
        let n_cells = self.m.n_cells as i32;
        let cfg_pad = self.cfg_pad;

        // (67.3) phase A: key = uid, value = slot.
        let f = self.k.init_uid.clone();
        {
            let Self { key_a, idx_a, .. } = self;
            unsafe {
                gpu.stream()
                    .launch_builder(&f)
                    .arg(&p.uid)
                    .arg(&mut *key_a)
                    .arg(&mut *idx_a)
                    .arg(&capacity)
                    .arg(&n_pad)
                    .launch(cfg_pad)?;
            }
        }
        for pass in 0..UID_PASSES {
            let shift = (pass * RADIX_BITS) as i32;
            self.radix_pass(gpu, shift, pass % 2 == 0)?;
        }

        // (67.3) phase B: re-key by cell, keeping the identity order, then
        // stably sort. LSD radix is stable, so the result is (cell, uid).
        let f = self.k.load_cell.clone();
        {
            let Self { key_a, idx_a, .. } = self;
            unsafe {
                gpu.stream()
                    .launch_builder(&f)
                    .arg(&p.cell)
                    .arg(&*idx_a)
                    .arg(&mut *key_a)
                    .arg(&n_cells)
                    .arg(&n_pad)
                    .launch(cfg_pad)?;
            }
        }
        for pass in 0..self.n_cell_passes {
            let shift = (pass * RADIX_BITS) as i32;
            self.radix_pass(gpu, shift, pass % 2 == 0)?;
        }

        // (67.5): the row pointer, by lower bound over the sorted keys.
        let f = self.k.csr_offsets.clone();
        let cfg_offsets = self.cfg_offsets;
        let final_is_a = self.final_is_a;
        {
            let Self { key_a, key_b, pc_offset, .. } = self;
            let key: &DevBuf<u64> = if final_is_a { &*key_a } else { &*key_b };
            unsafe {
                gpu.stream()
                    .launch_builder(&f)
                    .arg(key)
                    .arg(&mut *pc_offset)
                    .arg(&n_cells)
                    .arg(&n_pad)
                    .launch(cfg_offsets)?;
            }
        }
        Ok(())
    }

    /// SPEC-LIT (67.6): one thread per cell, walking its own CSR segment.
    pub fn deposit(&mut self, gpu: &Gpu, p: &Parcels<'m>) -> Result<()> {
        self.check(p)?;

        let n_cells = self.m.n_cells as i32;
        let rho_l = p.ctrl.rho_liquid;
        let cfg_cells = self.cfg_cells;
        let final_is_a = self.final_is_a;
        let f = self.k.deposit.clone();
        let m = self.m;
        let Self {
            idx_a,
            idx_b,
            pc_offset,
            count,
            weight,
            volume_fraction,
            mass,
            ..
        } = self;
        let index: &DevBuf<Label> = if final_is_a { &*idx_a } else { &*idx_b };
        unsafe {
            gpu.stream()
                .launch_builder(&f)
                .arg(&*pc_offset)
                .arg(index)
                .arg(&p.np)
                .arg(&p.d)
                .arg(&m.v)
                .arg(&mut *count)
                .arg(&mut *weight)
                .arg(&mut *volume_fraction)
                .arg(&mut *mass)
                .arg(&rho_l)
                .arg(&n_cells)
                .launch(cfg_cells)?;
        }
        Ok(())
    }

    /// SPEC-LIT (67.4): one LSD radix pass over the digit at `shift`.
    ///
    /// Three launches - histogram, scan, scatter - all of fixed geometry.
    /// `from_a` is derived from the pass number and never from data, so the
    /// pointer sequence is identical on every rebuild, which is what lets a
    /// captured graph replay it.
    fn radix_pass(&mut self, gpu: &Gpu, shift: i32, from_a: bool) -> Result<()> {
        let nb = self.nb as i32;
        let cfg_radix = self.cfg_radix;

        let f = self.k.histogram.clone();
        {
            let Self { key_a, key_b, counters, .. } = self;
            let key: &DevBuf<u64> = if from_a { &*key_a } else { &*key_b };
            unsafe {
                gpu.stream()
                    .launch_builder(&f)
                    .arg(key)
                    .arg(&mut *counters)
                    .arg(&shift)
                    .arg(&nb)
                    .launch(cfg_radix)?;
            }
        }

        self.scan.run(gpu, &self.counters, &mut self.counters_scan)?;

        let f = self.k.scatter.clone();
        {
            let Self { key_a, key_b, idx_a, idx_b, counters_scan, .. } = self;
            unsafe {
                if from_a {
                    gpu.stream()
                        .launch_builder(&f)
                        .arg(&*key_a)
                        .arg(&*idx_a)
                        .arg(&mut *key_b)
                        .arg(&mut *idx_b)
                        .arg(&*counters_scan)
                        .arg(&shift)
                        .arg(&nb)
                        .launch(cfg_radix)?;
                } else {
                    gpu.stream()
                        .launch_builder(&f)
                        .arg(&*key_b)
                        .arg(&*idx_b)
                        .arg(&mut *key_a)
                        .arg(&mut *idx_a)
                        .arg(&*counters_scan)
                        .arg(&shift)
                        .arg(&nb)
                        .launch(cfg_radix)?;
                }
            }
        }
        Ok(())
    }

    fn check(&self, p: &Parcels<'m>) -> Result<()> {
        if !std::ptr::eq(self.m, p.m) {
            return Err(Error::Config(
                "parcels/deposition: this deposition was built against a different \
                 GpuMesh from the pool it was handed (SPEC-LIT S67.5)"
                    .to_string(),
            ));
        }
        if self.capacity != p.ctrl.capacity {
            return Err(Error::Config(format!(
                "parcels/deposition: sized for capacity {} but the pool has {}; the sort \
                 scratch is allocated once at setup and a captured graph has frozen its \
                 geometry (SPEC-LIT S67.7)",
                self.capacity, p.ctrl.capacity
            )));
        }
        Ok(())
    }

    // ---- read-back, for output and for tests --------------------------

    /// The whole CSR, on the host. A device read-back: call it when a driver
    /// reports, never inside the step.
    pub fn csr_snapshot(&self, gpu: &Gpu) -> Result<ParcelCsrSnapshot> {
        let offset = gpu.download(self.offsets())?;
        let index = gpu.download(self.index())?;
        let cell_key = gpu.download(self.sorted_keys())?;
        let n_cells = self.m.n_cells;
        let n_live = offset[n_cells].max(0) as usize;
        Ok(ParcelCsrSnapshot { offset, index, cell_key, n_cells, n_live })
    }

    /// The deposited fields, on the host.
    pub fn snapshot(&self, gpu: &Gpu) -> Result<DepositSnapshot> {
        Ok(DepositSnapshot {
            count: gpu.download(&self.count)?,
            weight: gpu.download(&self.weight)?,
            volume_fraction: gpu.download(&self.volume_fraction)?,
            mass: gpu.download(&self.mass)?,
        })
    }

    /// `pc_offset[n_cells]` - how many parcels the CSR holds.
    pub fn live_count(&self, gpu: &Gpu) -> Result<usize> {
        Ok(gpu.download(&self.pc_offset)?[self.m.n_cells].max(0) as usize)
    }
}
