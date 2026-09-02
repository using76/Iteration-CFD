// meteor-cfd - Copyright (c) 2026 주식회사 메테오시뮬레이션 (Meteo Simulation Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

//! The adapt: what marks a cell, and what the mesh becomes when it does.
//!
//! Written from:
//!   ofgpu SPEC-LIT.md sections 1, 2, 3, 74 and 75
//!   R. Loehner, "An adaptive finite element scheme for transient problems in
//!     CFD", Comput. Methods Appl. Mech. Engrg. 61 (1987) 323-338 - the
//!     second-derivative error ratio of SPEC-LIT section 75.3(a). The form
//!     here is a cell-centred finite-volume RESTATEMENT of Loehner's nodal
//!     ratio and is marked DESIGN where it departs.
//!   T. J. Barth and D. C. Jespersen, "The design and application of upwind
//!     schemes on unstructured meshes", AIAA paper 89-0366 (1989) - the
//!     monotone reconstruction limiter of SPEC-LIT section 75.6, which is what
//!     stops prolongation inventing a new extremum.
//!   T. Isaac, C. Burstedde and O. Ghattas, "Low-cost parallel algorithms for
//!     2:1 octree balance", IPDPS 2012, 426-437 - the balance condition. Its
//!     O(1)-ripple algorithm is NAMED AND NOT IMPLEMENTED; the monotone
//!     fixed-point sweep of SPEC-LIT section 75.4 is used instead, and that
//!     section says why it is enough here.
//!   S. Muzaferija and D. Gosman, J. Comput. Phys. 138 (1997) 766-787 - a
//!     locally refined arbitrary-topology FV mesh is an ordinary polyhedral
//!     mesh, which is why an adapt here produces a plain `HostMesh` and needs
//!     no new patch kind and no flux register.
//!   H. Jasak and A. D. Gosman, Numer. Heat Transfer B 38 (2000) 237-256 and
//!     257-271 - the a-posteriori residual estimator and the refine/coarsen
//!     driver built on it. NAMED AND NOT IMPLEMENTED; refused by name in
//!     SPEC-LIT section 75.9 with the Loehner indicator as the alternative.
//!   K. McGrattan, S. Hostikka, R. McDermott, J. Floyd, C. Weinschenk and
//!     K. Overholt, Fire Dynamics Simulator User's Guide, NIST Special
//!     Publication 1019, "Mesh Resolution" - US Government work, public
//!     domain. The characteristic fire diameter D* and the D*/dx resolution
//!     measure of SPEC-LIT section 75.3(b).
//! No GPL-licensed source was consulted.
//!
//! # What this module does, and what it does not
//!
//! It **changes a mesh**. `mesh::refined` (SPEC-LIT section 74) builds a mesh
//! that is *born* with 2:1 interfaces; this module takes such a mesh, a field
//! and a marking rule, and produces the mesh and the fields that come out the
//! other side of a refine or a coarsen.
//!
//! It is **not wired into any time loop.** No case file can reach it, no
//! solver calls it, and SPEC-LIT section 75.9 records that as the honest
//! statement of where the feature stands. What is delivered and measured here
//! is the adapt itself - the criterion, the plan, the rebuild and the transfer
//! - each with its own gate.
//!
//! # The state
//!
//! A [`Forest`] is a base grid of hexahedra with an octree over each, stored
//! as its **leaves** and nothing else - a linear octree. A leaf is
//! `(base cell, level, octant coordinates at that level)`. There is no tree to
//! walk: every question this module asks of the forest is answered by an
//! integer index or by the cell -> face CSR of the mesh the forest emits.
//!
//! # The four things an adapt has to get right
//!
//! 1. **2:1 balance**, or the emitter cannot produce a rectangle where two
//!    cells meet. A monotone fixed-point sweep over the leaf face-adjacency
//!    graph, order-independent because integer `max` is associative and levels
//!    only rise.
//! 2. **A complete family**, or a coarsen is not a coarsen. All eight siblings
//!    must be leaves, at the same level, all marked. Detected here with a
//!    `BTreeMap` on the parent key - ordered, therefore deterministic, and
//!    never a hash order.
//! 3. **Conservation**, which is [`transfer`]'s job and is exact by
//!    construction rather than by rescale. See the module note there: the
//!    design note of record prescribed a multiplicative rescale that is
//!    singular for any field with a zero volume-weighted mean, and recentring
//!    the reconstruction removes the need for one entirely.
//! 4. **The rebuild**, which is [`rebuild`]'s job: the LDU order and the
//!    cell -> face CSR, from one stable sort and two binary searches, with no
//!    atomic anywhere and no prefix scan either.

pub mod rebuild;
pub mod transfer;

use std::collections::BTreeMap;

use cudarc::driver::PushKernelArg;

use crate::device::{cfg_for, Gpu, KernelSet};
use crate::error::{Error, Result};
use crate::mesh::refined::RefinedBox;
use crate::mesh::{GpuMesh, HostMesh, PatchInfo, PatchKind};
use crate::{DevBuf, Label, Scalar, Vec3};

/// Where the geometry sweep of a rebuild runs.
///
/// SPEC-LIT section 82. An adapt renumbers every cell, so the mesh and its
/// geometry have to be built again from nothing; section 75.8 measured that
/// this, and not the CUDA graph, is what makes an adapt expensive, and section
/// 82.2 measures which part of it.
///
/// **The two arms compute the same bits.** Not to a tolerance:
/// `mesh::gpugeom` is gated on bitwise identity with `mesh::geometry`, so
/// [`Rebuild::Device`] cannot change any answer, only how long it takes to
/// reach it. [`Rebuild::Host`] is the default everywhere and is literally the
/// code that was here before the device arm existed, so a caller that does not
/// ask for the device cannot be affected by it - BY CONSTRUCTION, section
/// 13.4.1.
#[derive(Clone, Copy)]
pub enum Rebuild<'a> {
    /// `mesh::geometry::compute`, on the host. The default.
    Host,
    /// `mesh::gpugeom::gpu_compute_geometry`, on the device.
    ///
    /// Faster only if you then keep the mesh: section 82.2 measures the
    /// download at two thirds of what the whole device path costs. This arm
    /// must return a `HostMesh`, so it cannot avoid that download; the
    /// device-resident route is [`plan_resident`], which does not build one -
    /// section 83.
    Device(&'a Gpu, &'a crate::mesh::gpugeom::MeshGeomKernels),
}

/// The deepest level a [`Leaf`] may carry. The canonical ordering key packs
/// octant coordinates shifted to this level, so it also bounds the key.
pub const LEVEL_MAX: u32 = 6;

/// The largest number of finest-grid voxels [`Forest::build`] will allocate
/// the leaf-lookup array for. A base grid of `n` cells whose deepest leaf is
/// at level `L` needs `n * 8^L` of them, and that is the one place in this
/// module where an innocent-looking level costs gigabytes. Refused by name
/// rather than discovered by the allocator.
pub const VOXEL_LIMIT: usize = 64 << 20;

// ===========================================================================
//  The forest
// ===========================================================================

/// One leaf of the forest: a hexahedron.
///
/// `oct` is the leaf's integer position inside its base cell **at its own
/// level**, so `oct[a] < 2^level` on every axis. A leaf at level 0 is the
/// whole base cell and has `oct = [0, 0, 0]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Leaf {
    pub base: u32,
    pub level: u32,
    pub oct: [u32; 3],
}

impl Leaf {
    /// The leaf's lower corner on the common grid of level [`LEVEL_MAX`].
    ///
    /// Comparing two leaves on this grid is the canonical order, and it does
    /// not depend on the forest's current deepest level: shifting both sides
    /// by the same amount cannot reorder them.
    #[inline]
    pub fn corner(&self) -> [u32; 3] {
        let s = LEVEL_MAX - self.level;
        [self.oct[0] << s, self.oct[1] << s, self.oct[2] << s]
    }

    /// The ordering key: base cell first, then `(z, y, x)` of the corner.
    ///
    /// Base-cell-major with the leaves of a base cell in `(z, y, x)` order is
    /// exactly the numbering `mesh::refined` documents, so a forest in which
    /// every base cell is uniformly refined emits **the same mesh, bit for
    /// bit**, as the static generator does.
    /// `tests::the_forest_emitter_reproduces_the_static_generator` is that
    /// statement, and it is the cross-check that this module's emitter is not
    /// quietly a different mesh generator.
    #[inline]
    pub fn key(&self) -> (u32, u32, u32, u32) {
        let c = self.corner();
        (self.base, c[2], c[1], c[0])
    }

    /// The leaf this one would coarsen into. `None` at level 0.
    #[inline]
    pub fn parent(&self) -> Option<Leaf> {
        if self.level == 0 {
            return None;
        }
        Some(Leaf {
            base: self.base,
            level: self.level - 1,
            oct: [self.oct[0] >> 1, self.oct[1] >> 1, self.oct[2] >> 1],
        })
    }

    /// The eight leaves this one would refine into, in `(z, y, x)` order -
    /// the same order the canonical key sorts them into.
    pub fn children(&self) -> [Leaf; 8] {
        let mut out = [*self; 8];
        let mut i = 0;
        for c in 0..2u32 {
            for b in 0..2u32 {
                for a in 0..2u32 {
                    out[i] = Leaf {
                        base: self.base,
                        level: self.level + 1,
                        oct: [2 * self.oct[0] + a, 2 * self.oct[1] + b, 2 * self.oct[2] + c],
                    };
                    i += 1;
                }
            }
        }
        out
    }
}

/// A forest of octrees over a structured base grid: the adaptable mesh state.
///
/// Leaves are kept sorted by [`Leaf::key`] and every operation that produces a
/// new forest re-sorts, so the cell numbering is a function of the leaf SET
/// and never of the order the adapt happened to visit things in. That is what
/// makes an adapt reproducible run to run, and it is the reason this module
/// keeps no free list (SPEC-LIT section 75.5).
#[derive(Debug, Clone)]
pub struct Forest {
    n: [usize; 3],
    d: Vec3,
    leaf: Vec<Leaf>,
}

impl Forest {
    /// The unrefined base grid: one leaf per base cell, all at level 0.
    pub fn uniform(n: [usize; 3], d: Vec3) -> Result<Self> {
        Self::from_base_levels(n, d, &vec![0u32; n[0] * n[1] * n[2]])
    }

    /// A forest in which base cell `i` is uniformly refined `level[i]` times.
    ///
    /// The levels are 2:1 balanced across face-adjacent BASE cells first, by
    /// the same sweep `mesh::refined::balance_2to1` runs, so this constructor
    /// and `mesh::refined::build` are handed identical level fields.
    pub fn from_base_levels(n: [usize; 3], d: Vec3, level: &[u32]) -> Result<Self> {
        let nb = n[0] * n[1] * n[2];
        if n[0] == 0 || n[1] == 0 || n[2] == 0 {
            return Err(Error::Mesh(
                "a forest needs at least one base cell on every axis".to_string(),
            ));
        }
        if level.len() != nb {
            return Err(Error::Mesh(format!(
                "the level field has {} entries, but the base grid has {nb} cells",
                level.len()
            )));
        }
        let mut lev = level.to_vec();
        crate::mesh::refined::balance_2to1(n, &mut lev);
        let lmax = lev.iter().copied().max().unwrap_or(0);
        if lmax > LEVEL_MAX {
            return Err(Error::Mesh(format!(
                "refinement level {lmax} is past this crate's limit of {LEVEL_MAX}"
            )));
        }

        let mut leaf = Vec::new();
        for k in 0..n[2] {
            for j in 0..n[1] {
                for i in 0..n[0] {
                    let b = i + n[0] * (j + n[1] * k);
                    let l = lev[b];
                    let s = 1u32 << l;
                    for c in 0..s {
                        for bb in 0..s {
                            for a in 0..s {
                                leaf.push(Leaf { base: b as u32, level: l, oct: [a, bb, c] });
                            }
                        }
                    }
                }
            }
        }
        Ok(Self { n, d, leaf })
    }

    /// Build a forest from an explicit leaf set. The set is sorted into the
    /// canonical order and checked for overlaps and gaps, because a leaf set
    /// that does not tile the base grid is not a mesh.
    pub fn from_leaves(n: [usize; 3], d: Vec3, mut leaf: Vec<Leaf>) -> Result<Self> {
        if n[0] == 0 || n[1] == 0 || n[2] == 0 {
            return Err(Error::Mesh(
                "a forest needs at least one base cell on every axis".to_string(),
            ));
        }
        if leaf.is_empty() {
            return Err(Error::Mesh("a forest with no leaves is not a mesh".to_string()));
        }
        let nb = n[0] * n[1] * n[2];
        for l in &leaf {
            if l.level > LEVEL_MAX {
                return Err(Error::Mesh(format!(
                    "leaf level {} is past this crate's limit of {LEVEL_MAX}",
                    l.level
                )));
            }
            if (l.base as usize) >= nb {
                return Err(Error::Mesh(format!("leaf names base cell {} of {nb}", l.base)));
            }
            let s = 1u32 << l.level;
            if l.oct.iter().any(|&o| o >= s) {
                return Err(Error::Mesh(format!(
                    "leaf octant {:?} is outside a level-{} base cell",
                    l.oct, l.level
                )));
            }
        }
        leaf.sort_by_key(|l| l.key());

        // The leaves must tile every base cell exactly. Measured as a volume
        // sum in exact INTEGER arithmetic on the level-LEVEL_MAX grid, so it
        // cannot be fooled by round-off; it catches overlap (too much) and gap
        // (too little) only in their sum, which is why the corners are checked
        // for duplicates separately.
        let unit = 1u64 << (3 * LEVEL_MAX);
        let mut vol = vec![0u64; nb];
        for l in &leaf {
            vol[l.base as usize] += 1u64 << (3 * (LEVEL_MAX - l.level));
        }
        for (b, v) in vol.iter().enumerate() {
            if *v != unit {
                return Err(Error::Mesh(format!(
                    "the leaves of base cell {b} cover {} of it, not all of it - \
                     the leaf set is not a partition",
                    *v as f64 / unit as f64
                )));
            }
        }
        for w in leaf.windows(2) {
            if w[0].key() == w[1].key() {
                return Err(Error::Mesh(format!(
                    "two leaves share the corner {:?} in base cell {}",
                    w[0].corner(),
                    w[0].base
                )));
            }
        }

        Ok(Self { n, d, leaf })
    }

    pub fn base_n(&self) -> [usize; 3] {
        self.n
    }
    pub fn base_d(&self) -> Vec3 {
        self.d
    }
    pub fn leaves(&self) -> &[Leaf] {
        &self.leaf
    }
    pub fn len(&self) -> usize {
        self.leaf.len()
    }
    pub fn is_empty(&self) -> bool {
        self.leaf.is_empty()
    }
    pub fn level(&self, c: usize) -> u32 {
        self.leaf[c].level
    }
    /// The refinement level of every leaf, in cell order.
    pub fn levels(&self) -> Vec<u32> {
        self.leaf.iter().map(|l| l.level).collect()
    }
    pub fn lmax(&self) -> u32 {
        self.leaf.iter().map(|l| l.level).max().unwrap_or(0)
    }

    /// The leaf's edge lengths.
    pub fn size(&self, c: usize) -> Vec3 {
        let s = (1u32 << self.leaf[c].level) as Scalar;
        Vec3::new(self.d.x / s, self.d.y / s, self.d.z / s)
    }

    /// The leaf's centre, from integer arithmetic on the base grid rather than
    /// from emitted geometry, so it is available before any mesh exists.
    pub fn centre(&self, c: usize) -> Vec3 {
        let l = &self.leaf[c];
        let b = l.base as usize;
        let (i, j, k) =
            (b % self.n[0], (b / self.n[0]) % self.n[1], b / (self.n[0] * self.n[1]));
        let s = (1u32 << l.level) as Scalar;
        let h = Vec3::new(self.d.x / s, self.d.y / s, self.d.z / s);
        Vec3::new(
            i as Scalar * self.d.x + (l.oct[0] as Scalar + 0.5) * h.x,
            j as Scalar * self.d.y + (l.oct[1] as Scalar + 0.5) * h.y,
            k as Scalar * self.d.z + (l.oct[2] as Scalar + 0.5) * h.z,
        )
    }

    // -----------------------------------------------------------------------
    //  Emission
    // -----------------------------------------------------------------------

    /// Emit the polyMesh this leaf set is.
    ///
    /// The traversal is deliberately the same one `mesh::refined::build`
    /// runs: cell-major, minus side then plus side, the same corner winding,
    /// the same `BTreeMap` grouping. On a forest the static generator can also
    /// express, the two therefore produce identical bits.
    ///
    /// The ONE difference is the branch marked below. An adapted forest can
    /// number a leaf's `+axis` neighbour EARLIER than the leaf itself, which
    /// cannot happen in the static generator's ordering, so the face is
    /// emitted with the smaller index as owner and its polygon reversed, and
    /// the sort at the end puts the whole list into upper-triangular order.
    pub fn build(&self) -> Result<RefinedBox> {
        self.build_with(Rebuild::Host)
    }

    /// [`Forest::build`], with the geometry sweep run where `how` says.
    ///
    /// The emission - the voxel map, the face grouping, the point numbering
    /// and the sort - is the same code in both arms and is still on the host;
    /// SPEC-LIT section 82.9 records that it is now the larger half of a
    /// rebuild and what a device version of it would have to reproduce.
    pub fn build_with(&self, how: Rebuild<'_>) -> Result<RefinedBox> {
        let (mut mesh, points, faces) = self.emit()?;
        match how {
            Rebuild::Host => mesh.compute_geometry(&points, &faces)?,
            Rebuild::Device(gpu, k) => {
                crate::mesh::gpugeom::gpu_compute_geometry(gpu, k, &mut mesh, &points, &faces)?
            }
        }
        Ok(RefinedBox {
            mesh,
            points,
            faces,
            level: self.levels(),
            base_n: self.n,
            base_d: self.d,
        })
    }

    /// [`Forest::build`], with the geometry computed on the device and **left
    /// there**: the mesh comes back already uploaded.
    ///
    /// SPEC-LIT section 83. [`Rebuild::Device`] runs the same kernels and then
    /// spends the sixteen-array download to fill a `HostMesh`, which a caller
    /// then spends a sixteen-array upload to put back; section 82.2 measured
    /// the first of those at half of what the device drop-in costs and section
    /// 82.5 measured the second separately. This route pays neither. It is not
    /// a faster `build_with` - it is a different destination.
    ///
    /// **What comes back on the host.** The `RefinedBox`'s mesh carries its
    /// points, its faces, its topology, its addressing, the four boundary
    /// bookkeeping arrays - and of the sixteen geometric arrays, only `v`.
    /// `v` is not a print: [`plan_resident`] builds the conservative
    /// prolongation weights `w_qp = V_q / sum V` of section 75.6 out of it on
    /// the host, and `GpuMesh::total_volume` is its fold. Section 83.3 prices
    /// that one array and section 83.9 says what a device weight kernel would
    /// have to reproduce to remove it. Every other geometric array on that
    /// mesh is EMPTY, deliberately: an empty `c` is a length mismatch at the
    /// first line that reads it, and a stale or zeroed one would not be.
    pub fn build_resident(
        &self,
        gpu: &Gpu,
        k: &crate::mesh::gpugeom::MeshGeomKernels,
    ) -> Result<(RefinedBox, GpuMesh)> {
        let (mut mesh, points, faces) = self.emit()?;
        let csr = crate::mesh::gpugeom::flatten_faces(&faces);
        let geom =
            crate::mesh::gpugeom::gpu_geometry_resident(gpu, k, &mut mesh, &points, &faces, &csr)?;
        mesh.v = geom.download_volumes(gpu)?;
        let gm = GpuMesh::from_device_geometry(gpu, &mesh, geom)?;
        Ok((
            RefinedBox {
                mesh,
                points,
                faces,
                level: self.levels(),
                base_n: self.n,
                base_d: self.d,
            },
            gm,
        ))
    }

    /// The emitter proper: everything but the geometry sweep.
    ///
    /// Returns the mesh with its topology and its cell -> face CSR filled and
    /// every geometric array still empty, together with the points and the
    /// face point lists the sweep needs. `build_cell_face_maps` runs here
    /// because the device sweep GATHERS over that CSR and cannot be called
    /// without it - see `mesh::gpugeom`'s precondition.
    fn emit(&self) -> Result<(HostMesh, Vec<Vec3>, Vec<Vec<Label>>)> {
        let (nx, ny, nz) = (self.n[0], self.n[1], self.n[2]);
        let lmax = self.lmax();
        let fac = 1usize << lmax;
        let vn = [nx * fac, ny * fac, nz * fac];
        let nvox = vn[0]
            .checked_mul(vn[1])
            .and_then(|a| a.checked_mul(vn[2]))
            .ok_or_else(|| Error::Mesh("the finest grid overflows a usize".to_string()))?;
        if nvox > VOXEL_LIMIT {
            return Err(Error::Mesh(format!(
                "a {}x{}x{} base grid at level {lmax} needs {nvox} finest-grid voxels, \
                 past this module's limit of {VOXEL_LIMIT}",
                nx, ny, nz
            )));
        }
        let vox = |i: usize, j: usize, k: usize| i + vn[0] * (j + vn[1] * k);

        // Every leaf's voxel box [lo, hi) on the finest grid.
        let n_cells = self.leaf.len();
        let mut lo: Vec<[usize; 3]> = Vec::with_capacity(n_cells);
        let mut hi: Vec<[usize; 3]> = Vec::with_capacity(n_cells);
        for l in &self.leaf {
            let b = l.base as usize;
            let (i, j, k) = (b % nx, (b / nx) % ny, b / (nx * ny));
            let step = fac >> l.level;
            let p = [
                i * fac + l.oct[0] as usize * step,
                j * fac + l.oct[1] as usize * step,
                k * fac + l.oct[2] as usize * step,
            ];
            lo.push(p);
            hi.push([p[0] + step, p[1] + step, p[2] + step]);
        }

        // Which leaf owns each finest-grid voxel. This is the whole neighbour
        // search: one O(1) lookup per voxel, no tree walk and no hash table.
        let mut owner_of = vec![-1 as Label; nvox];
        for (c, (a, b)) in lo.iter().zip(hi.iter()).enumerate() {
            for k in a[2]..b[2] {
                for j in a[1]..b[1] {
                    for i in a[0]..b[0] {
                        owner_of[vox(i, j, k)] = c as Label;
                    }
                }
            }
        }
        // A voxel no leaf claimed means the leaf set has a gap - and, since
        // the volumes summed to the base cell in `from_leaves`, an overlap
        // somewhere else paying for it. `from_leaves` catches that pair only
        // when the two leaves share a corner; this catches the rest, here,
        // rather than letting the face grouping below read a `-1` and index
        // the leaf arrays out of range.
        if let Some(v) = owner_of.iter().position(|&c| c < 0) {
            return Err(Error::Mesh(format!(
                "voxel {v} of the finest grid belongs to no leaf; the leaf set \
                 overlaps itself somewhere and leaves a gap here"
            )));
        }

        let h = Vec3::new(
            self.d.x / fac as Scalar,
            self.d.y / fac as Scalar,
            self.d.z / fac as Scalar,
        );
        let pn = [vn[0] + 1, vn[1] + 1, vn[2] + 1];
        let mut point_id = vec![-1 as Label; pn[0] * pn[1] * pn[2]];
        let mut points: Vec<Vec3> = Vec::new();

        let corners = |axis: usize, q: usize, a0: usize, a1: usize, b0: usize, b1: usize| {
            let mk = |a: usize, b: usize| -> [usize; 3] {
                let mut p = [0usize; 3];
                p[axis] = q;
                p[(axis + 1) % 3] = a;
                p[(axis + 2) % 3] = b;
                p
            };
            [mk(a0, b0), mk(a1, b0), mk(a1, b1), mk(a0, b1)]
        };

        let mut internal: Vec<(Label, Label, Vec<Label>)> = Vec::new();
        let mut patch_faces: Vec<Vec<(Label, Vec<Label>)>> = vec![Vec::new(); 6];

        for c in 0..n_cells {
            let (a, b) = (lo[c], hi[c]);
            for axis in 0..3 {
                let (t1, t2) = ((axis + 1) % 3, (axis + 2) % 3);

                let mut emit = |cs: [[usize; 3]; 4], points: &mut Vec<Vec3>| -> Vec<Label> {
                    cs.iter()
                        .map(|p| {
                            let s = p[0] + pn[0] * (p[1] + pn[1] * p[2]);
                            if point_id[s] < 0 {
                                point_id[s] = points.len() as Label;
                                points.push(Vec3::new(
                                    p[0] as Scalar * h.x,
                                    p[1] as Scalar * h.y,
                                    p[2] as Scalar * h.z,
                                ));
                            }
                            point_id[s]
                        })
                        .collect()
                };

                if a[axis] == 0 {
                    let cs = corners(axis, a[axis], a[t1], b[t1], a[t2], b[t2]);
                    let mut ps = emit(cs, &mut points);
                    ps.reverse();
                    patch_faces[2 * axis].push((c as Label, ps));
                }

                if b[axis] == vn[axis] {
                    let cs = corners(axis, b[axis], a[t1], b[t1], a[t2], b[t2]);
                    let ps = emit(cs, &mut points);
                    patch_faces[2 * axis + 1].push((c as Label, ps));
                    continue;
                }

                // Group the voxels of this face by the leaf on the far side.
                // A BTreeMap, so the emitted order is the neighbour's cell
                // order and never a hash order.
                let mut groups: BTreeMap<Label, [usize; 4]> = BTreeMap::new();
                for u in a[t1]..b[t1] {
                    for v in a[t2]..b[t2] {
                        let mut p = [0usize; 3];
                        p[axis] = b[axis];
                        p[t1] = u;
                        p[t2] = v;
                        let nb = owner_of[vox(p[0], p[1], p[2])];
                        let e = groups.entry(nb).or_insert([u, u, v, v]);
                        e[0] = e[0].min(u);
                        e[1] = e[1].max(u);
                        e[2] = e[2].min(v);
                        e[3] = e[3].max(v);
                    }
                }

                for (nb, r) in groups {
                    let nb_u = nb as usize;
                    // Under 2:1 balance the shared region is a full rectangle
                    // - one whole face of whichever cell is finer. If it is
                    // not, the mesh will not close and everything downstream
                    // is invalid, so say so here rather than emit it.
                    let want = (r[1] + 1 - r[0]) * (r[3] + 1 - r[2]);
                    let got = (a[t1].max(lo[nb_u][t1])..b[t1].min(hi[nb_u][t1])).len()
                        * (a[t2].max(lo[nb_u][t2])..b[t2].min(hi[nb_u][t2])).len();
                    if want != got {
                        return Err(Error::Mesh(format!(
                            "cells {c} and {nb_u} share a non-rectangular region on axis \
                             {axis}; the leaf set is not 2:1 balanced"
                        )));
                    }
                    let cs = corners(axis, b[axis], r[0], r[1] + 1, r[2], r[3] + 1);
                    let mut ps = emit(cs, &mut points);
                    // THE ONE DIFFERENCE from the static generator. `Sf` must
                    // point owner -> neighbour, and `emit` wound the polygon
                    // so it points along +axis, i.e. from `c` to `nb`. When
                    // `nb` is the smaller index it is the owner, so the
                    // winding is reversed and the pair swapped.
                    if nb_u < c {
                        ps.reverse();
                        internal.push((nb, c as Label, ps));
                    } else {
                        internal.push((c as Label, nb, ps));
                    }
                }
            }
        }

        internal.sort_by_key(|&(o, nb, _)| (o, nb));

        let names = ["xmin", "xmax", "ymin", "ymax", "zmin", "zmax"];
        let mut faces: Vec<Vec<Label>> = Vec::with_capacity(internal.len());
        let mut owner = Vec::with_capacity(internal.len());
        let mut neighbour = Vec::with_capacity(internal.len());
        for (o, nb, fp) in internal {
            owner.push(o);
            neighbour.push(nb);
            faces.push(fp);
        }

        let mut b_face_cells = Vec::new();
        let mut patches = Vec::new();
        for (p, mut pf) in patch_faces.into_iter().enumerate() {
            pf.sort_by_key(|(c, _)| *c);
            let start = b_face_cells.len();
            let size = pf.len();
            for (c, fp) in pf {
                b_face_cells.push(c);
                faces.push(fp);
            }
            patches.push(PatchInfo {
                name: names[p].to_string(),
                type_name: "patch".to_string(),
                kind: PatchKind::Generic,
                start,
                size,
                nbr_patch: None,
            });
        }

        let mut mesh = HostMesh {
            n_cells,
            n_internal_faces: owner.len(),
            n_boundary_faces: b_face_cells.len(),
            n_points: points.len(),
            owner,
            neighbour,
            b_face_cells,
            patches,
            ..Default::default()
        };
        mesh.build_cell_face_maps();

        Ok((mesh, points, faces))
    }
}

// ===========================================================================
//  Marking
// ===========================================================================

/// What the criterion asked for a cell. `Keep` is the answer to every question
/// the criterion did not decide.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i8)]
pub enum Mark {
    Coarsen = -1,
    Keep = 0,
    Refine = 1,
}

/// Turn an error indicator into marks, with the hysteresis band that stops a
/// mesh sitting on the threshold from refining and coarsening on alternate
/// adapts - SPEC-LIT section 75.3(c).
///
/// `tau_c <= tau_r / 4` is required rather than recommended: a band narrower
/// than that is what the fixed-point gate
/// `tests::hysteresis_reaches_a_fixed_point_on_a_steady_field` exists to
/// forbid, and a caller that wants no band can pass `tau_c = 0`, which never
/// coarsens anything with a non-zero indicator.
pub fn mark_with_hysteresis(
    e: &[Scalar],
    level: &[u32],
    tau_r: Scalar,
    tau_c: Scalar,
    l_max: u32,
) -> Result<Vec<Mark>> {
    if e.len() != level.len() {
        return Err(Error::Config(format!(
            "the indicator has {} entries and the level field {}",
            e.len(),
            level.len()
        )));
    }
    let band = tau_r / 4.0;
    if tau_c > band || tau_c.is_nan() || band.is_nan() {
        return Err(Error::Config(format!(
            "the hysteresis band is too narrow: tau_c = {tau_c} must be at most \
             tau_r/4 = {}, or a mesh on the threshold refines and coarsens on \
             alternate adapts and every adapt costs a full rebuild",
            tau_r / 4.0
        )));
    }
    Ok(e.iter()
        .zip(level.iter())
        .map(|(&ei, &li)| {
            if ei > tau_r && li < l_max {
                Mark::Refine
            } else if ei < tau_c && li > 0 {
                Mark::Coarsen
            } else {
                Mark::Keep
            }
        })
        .collect())
}

/// Combine several indicators by weighted maximum - SPEC-LIT section 75.3(c).
///
/// A maximum and not a sum: two indicators that each say "this is smooth"
/// should not add up to "refine", and one indicator that says "refine" must
/// not be outvoted.
pub fn combine_max(out: &mut Vec<Scalar>, e: &[Scalar], w: Scalar) {
    if out.len() != e.len() {
        out.clear();
        out.resize(e.len(), 0.0);
    }
    for (o, &ei) in out.iter_mut().zip(e.iter()) {
        *o = o.max(w * ei);
    }
}

// ===========================================================================
//  The plan
// ===========================================================================

/// How a coarsening family is grouped: the parent's LEVEL and its canonical
/// key. The level is not decoration - see the note in [`plan`].
type FamilyKey = (u32, (u32, u32, u32, u32));

/// The gather map from the new cell numbering to the old one.
///
/// Both directions are stored because both are needed and both are gathers:
/// the transfer reads `src_*` (one thread per NEW cell), and the parent
/// quantities of SPEC-LIT section 75.6 read `own_*` (one thread per OLD cell).
/// Neither is a scatter and neither needs an atomic.
#[derive(Debug, Clone, Default)]
pub struct Map {
    pub n_old: usize,
    pub n_new: usize,
    /// `[n_new + 1]` offsets into [`Self::src_cell`].
    pub src_offset: Vec<Label>,
    /// The OLD cells a new cell draws from: one for a kept or refined cell,
    /// eight for a coarsened one.
    pub src_cell: Vec<Label>,
    /// The conservative weight of each source, `w_qp` of SPEC-LIT section
    /// 75.6. Exactly `1` for a kept or coarsened cell; `V_q / sum_c V_c` for a
    /// refined one, so that `sum_q w_qp = 1` for every old cell.
    pub src_w: Vec<Scalar>,
    /// `[n_old + 1]` offsets into [`Self::own_child`] - the transpose.
    pub own_offset: Vec<Label>,
    /// The NEW cells an old cell feeds: one for a kept or coarsened cell,
    /// eight for a refined one.
    pub own_child: Vec<Label>,
    /// The same weights, aligned with [`Self::own_child`].
    pub own_w: Vec<Scalar>,
}

/// What an adapt did, and the mesh it produced.
pub struct Plan {
    /// The forest after the adapt.
    pub after: Forest,
    /// The mesh the new forest emits, with geometry already computed.
    pub mesh: RefinedBox,
    /// The gather map between the two numberings.
    pub map: Map,
    /// The target level of every OLD cell, after balance and family checks.
    pub target: Vec<u32>,
    pub n_refined: usize,
    pub n_coarsened: usize,
    pub n_kept: usize,
    /// Balance sweeps run before the fixed point was reached.
    pub balance_sweeps: usize,
    /// Cells promoted by 2:1 balance that the criterion had not marked.
    pub promoted: usize,
    /// Families the criterion marked for coarsening that balance then refused.
    pub cancelled_coarsen: usize,
}

impl Plan {
    /// True when the adapt left the mesh alone. The cheapest useful question
    /// to ask, because it is what decides whether anything downstream has to
    /// be rebuilt at all.
    pub fn is_identity(&self) -> bool {
        self.n_refined == 0 && self.n_coarsened == 0
    }
}

/// One monotone 2:1 balance sweep over the leaf face-adjacency graph.
///
/// `target[P] <- max(target[P], max_N target[N] - 1)`. Levels only rise and
/// integer `max` is associative and exact, so the fixed point does not depend
/// on the order the cells are visited - which is what lets the device version
/// (`adaptBalanceSweep`) run one thread per cell with no atomic and get the
/// same answer.
///
/// Returns the number of cells it raised.
pub fn balance_sweep(target: &mut [u32], owner: &[Label], neighbour: &[Label]) -> usize {
    let n = target.len();
    let mut want = vec![0u32; n];
    for (o, nb) in owner.iter().zip(neighbour.iter()) {
        let (o, nb) = (*o as usize, *nb as usize);
        if o >= n || nb >= n {
            continue;
        }
        want[o] = want[o].max(target[nb]);
        want[nb] = want[nb].max(target[o]);
    }
    let mut raised = 0;
    for c in 0..n {
        let floor = want[c].saturating_sub(1);
        if target[c] < floor {
            target[c] = floor;
            raised += 1;
        }
    }
    raised
}

/// Plan an adapt: turn marks into a target level per cell, a new forest and
/// the gather map between the two numberings.
///
/// `mesh` must be the mesh `before` emits - it supplies the face adjacency,
/// and nothing else. It is passed rather than re-derived because the caller
/// already has it and building it is the expensive half.
pub fn plan(before: &Forest, mesh: &HostMesh, mark: &[Mark], l_max: u32) -> Result<Plan> {
    plan_with(before, mesh, mark, l_max, Rebuild::Host)
}

/// [`plan`], with the geometry sweep of the rebuild run where `how` says.
///
/// The two arms produce the same `Plan`, bit for bit - see [`Rebuild`].
pub fn plan_with(
    before: &Forest,
    mesh: &HostMesh,
    mark: &[Mark],
    l_max: u32,
    how: Rebuild<'_>,
) -> Result<Plan> {
    plan_by(before, mesh, mark, l_max, &mut |f: &Forest| f.build_with(how))
}

/// [`plan`], with the new mesh built **on the device and left there**.
///
/// SPEC-LIT section 83. Returns the same `Plan` the host route returns, except
/// that its `mesh.mesh` carries only `v` of the sixteen geometric arrays - see
/// [`Forest::build_resident`] - together with the [`GpuMesh`] that carries all
/// sixteen and needed no upload to get them.
///
/// **The `Map` is bit-for-bit the host route's.** Its weights are folds over
/// `v_new` in ascending cell id and `v` is downloaded, so the arithmetic that
/// produces them is literally the same code on the same bits; nothing about
/// the transfer moved. That is asserted rather than argued, in
/// `ofgpu-validate`'s adapt section.
///
/// This is the entry point section 82.5 said was missing: `plan_with` "must
/// return a `HostMesh`" and so could not use `GpuGeometry::compute`. It still
/// must, and it still does - the emitter is on the host (section 82.9) - but
/// the geometry no longer makes the round trip with it.
pub fn plan_resident(
    before: &Forest,
    mesh: &HostMesh,
    mark: &[Mark],
    l_max: u32,
    gpu: &Gpu,
    k: &crate::mesh::gpugeom::MeshGeomKernels,
) -> Result<(Plan, GpuMesh)> {
    let mut built: Option<GpuMesh> = None;
    let plan = plan_by(before, mesh, mark, l_max, &mut |f: &Forest| {
        let (rb, gm) = f.build_resident(gpu, k)?;
        built = Some(gm);
        Ok(rb)
    })?;
    let gm = built.ok_or_else(|| {
        Error::Mesh(
            "adapt::plan_resident: the rebuild did not run, so there is no device mesh to \
             return. This cannot happen unless plan_by stopped calling `build`"
                .to_string(),
        )
    })?;
    Ok((plan, gm))
}

/// The body of [`plan_with`] and [`plan_resident`], with the one line that
/// differs between them - how the new mesh is built - passed in.
///
/// The two entry points must not be two copies of the balance fixed point and
/// the gather map. SPEC-LIT section 83.5: the `Map` a resident adapt hands to
/// `transfer` is gated as bitwise identical to the host route's, and that gate
/// is only interesting because it is testing one implementation reached two
/// ways rather than two implementations that agree today.
fn plan_by(
    before: &Forest,
    mesh: &HostMesh,
    mark: &[Mark],
    l_max: u32,
    build: &mut dyn FnMut(&Forest) -> Result<RefinedBox>,
) -> Result<Plan> {
    let n_old = before.len();
    if mesh.n_cells != n_old {
        return Err(Error::Mesh(format!(
            "the mesh has {} cells and the forest {n_old}",
            mesh.n_cells
        )));
    }
    if mark.len() != n_old {
        return Err(Error::Config(format!(
            "the mark list has {} entries and the forest {n_old} cells",
            mark.len()
        )));
    }
    if l_max > LEVEL_MAX {
        return Err(Error::Config(format!(
            "l_max = {l_max} is past this crate's limit of {LEVEL_MAX}"
        )));
    }

    let level = before.levels();

    // The input must already be balanced, or the fixed point below is being
    // asked to repair a mesh rather than to preserve a property. Refuse by
    // name: an unbalanced input is a bug upstream, not something to paper over.
    for f in 0..mesh.n_internal_faces {
        let (o, nb) = (mesh.owner[f] as usize, mesh.neighbour[f] as usize);
        if level[o].abs_diff(level[nb]) > 1 {
            return Err(Error::Mesh(format!(
                "the mesh handed to plan() is not 2:1 balanced: cells {o} and {nb} \
                 are at levels {} and {}",
                level[o], level[nb]
            )));
        }
    }

    // ---- tentative targets -------------------------------------------------
    let mut target: Vec<u32> = level.clone();
    for c in 0..n_old {
        if mark[c] == Mark::Refine && level[c] < l_max {
            target[c] = level[c] + 1;
        }
    }

    // ---- candidate families ------------------------------------------------
    //
    // A coarsen needs a COMPLETE family: all eight siblings present as leaves,
    // all at the same level, all marked. Grouped by the parent key in a
    // BTreeMap - ordered, so the iteration order is a property of the keys and
    // not of a hash seed.
    //
    // The key carries the parent's LEVEL as well as its corner, and it has to.
    // `Leaf::key` shifts the octant up to a common grid, so a level-1 parent
    // at octant (0,0,0) and a level-0 parent at octant (0,0,0) have the SAME
    // corner. Without the level in the key those two buckets merge, a
    // legitimate family lands in a bucket of fifteen, and the coarsen is
    // silently refused - conservative, but wrong, and only on a mesh with
    // three levels in one base cell, which is exactly where it would not be
    // noticed.
    let mut fam: BTreeMap<FamilyKey, Vec<usize>> = BTreeMap::new();
    for c in 0..n_old {
        if mark[c] != Mark::Coarsen || level[c] == 0 {
            continue;
        }
        if let Some(p) = before.leaves()[c].parent() {
            fam.entry((p.level, p.key())).or_default().push(c);
        }
    }
    let mut families: Vec<Vec<usize>> = Vec::new();
    for (_, members) in fam {
        if members.len() == 8 && members.iter().all(|&c| level[c] == level[members[0]]) {
            families.push(members);
        }
    }
    let n_candidate_families = families.len();
    let mut alive = vec![true; families.len()];
    for members in &families {
        for &c in members {
            target[c] = level[c] - 1;
        }
    }

    // ---- the fixed point ---------------------------------------------------
    //
    // Balance raises; a raise on a family member cancels that family, which is
    // itself a raise. Every step after the first assignment only raises, so the
    // iteration is monotone and terminates, and the result does not depend on
    // the order.
    let mut sweeps = 0usize;
    loop {
        let raised = balance_sweep(&mut target, &mesh.owner, &mesh.neighbour);
        sweeps += 1;
        let mut cancelled = 0usize;
        for (fi, members) in families.iter().enumerate() {
            if !alive[fi] {
                continue;
            }
            if members.iter().any(|&c| target[c] > level[c] - 1) {
                alive[fi] = false;
                cancelled += 1;
                for &c in members {
                    target[c] = target[c].max(level[c]);
                }
            }
        }
        if raised == 0 && cancelled == 0 {
            break;
        }
        if sweeps > 4 * (LEVEL_MAX as usize + 1) {
            return Err(Error::Mesh(
                "the 2:1 balance fixed point did not settle; the level field cannot \
                 be monotone, which means this sweep has a bug"
                    .to_string(),
            ));
        }
    }
    let cancelled_coarsen = n_candidate_families - alive.iter().filter(|a| **a).count();
    let promoted = (0..n_old)
        .filter(|&c| target[c] > level[c] && mark[c] != Mark::Refine)
        .count();

    // ---- the new leaf set --------------------------------------------------
    let mut leaves: Vec<Leaf> = Vec::with_capacity(n_old);
    let mut n_refined = 0usize;
    let mut n_coarsened = 0usize;
    let mut n_kept = 0usize;
    for c in 0..n_old {
        let l = &before.leaves()[c];
        match target[c].cmp(&level[c]) {
            std::cmp::Ordering::Greater => {
                leaves.extend_from_slice(&l.children());
                n_refined += 1;
            }
            std::cmp::Ordering::Equal => {
                leaves.push(*l);
                n_kept += 1;
            }
            std::cmp::Ordering::Less => {
                n_coarsened += 1;
                // Emitted once per family, by the member the canonical order
                // reaches first - which is the one whose octant is even on
                // every axis. Any other rule would emit eight copies.
                if l.oct.iter().all(|o| o % 2 == 0) {
                    leaves.push(l.parent().expect("a coarsened leaf is above level 0"));
                }
            }
        }
    }
    let after = Forest::from_leaves(before.n, before.d, leaves)?;

    // ---- the gather map ----------------------------------------------------
    let n_new = after.len();
    let index: BTreeMap<(u32, u32, u32, u32), usize> =
        after.leaves().iter().enumerate().map(|(i, l)| (l.key(), i)).collect();

    // The new cell each old cell feeds, and the child slot within it.
    let mut own_offset = vec![0 as Label; n_old + 1];
    let mut own_child: Vec<Label> = Vec::with_capacity(n_old);
    for c in 0..n_old {
        let l = &before.leaves()[c];
        match target[c].cmp(&level[c]) {
            std::cmp::Ordering::Greater => {
                for ch in l.children() {
                    own_child.push(*index.get(&ch.key()).expect("child in the new forest") as Label);
                }
                own_offset[c + 1] = 8;
            }
            std::cmp::Ordering::Equal => {
                own_child.push(*index.get(&l.key()).expect("kept leaf in the new forest") as Label);
                own_offset[c + 1] = 1;
            }
            std::cmp::Ordering::Less => {
                let p = l.parent().expect("a coarsened leaf is above level 0");
                own_child.push(*index.get(&p.key()).expect("parent in the new forest") as Label);
                own_offset[c + 1] = 1;
            }
        }
    }
    for c in 0..n_old {
        own_offset[c + 1] += own_offset[c];
    }

    // The transpose, built by counting then filling - one pass each, and the
    // fill walks old cells in ascending order so a new cell's source list is
    // ascending in the OLD index. That fixed order is what makes the gathered
    // sum in `transfer` bitwise reproducible.
    let mut src_offset = vec![0 as Label; n_new + 1];
    for &q in &own_child {
        src_offset[q as usize + 1] += 1;
    }
    for q in 0..n_new {
        src_offset[q + 1] += src_offset[q];
    }
    let total = src_offset[n_new] as usize;
    let mut src_cell = vec![-1 as Label; total];
    let mut cursor: Vec<Label> = src_offset[..n_new].to_vec();
    for c in 0..n_old {
        for i in own_offset[c]..own_offset[c + 1] {
            let q = own_child[i as usize] as usize;
            src_cell[cursor[q] as usize] = c as Label;
            cursor[q] += 1;
        }
    }

    let mesh_new = build(&after)?;
    let v_new = &mesh_new.mesh.v;

    // The conservative weights. `w_qp = V_q / sum_{q' in C(p)} V_q'` makes
    // `sum_q w_qp = 1` for every old cell, which is the whole conservation
    // argument of SPEC-LIT section 75.6 - and it is exactly 1 for a kept or
    // coarsened cell because the sum then has one term.
    let mut own_w = vec![0.0 as Scalar; own_child.len()];
    for c in 0..n_old {
        let (a, b) = (own_offset[c] as usize, own_offset[c + 1] as usize);
        if b - a == 1 {
            own_w[a] = 1.0;
            continue;
        }
        let mut s = 0.0;
        for i in a..b {
            s += v_new[own_child[i] as usize];
        }
        for i in a..b {
            own_w[i] = v_new[own_child[i] as usize] / s;
        }
    }
    let mut src_w = vec![0.0 as Scalar; total];
    let mut cursor: Vec<Label> = src_offset[..n_new].to_vec();
    for c in 0..n_old {
        for i in own_offset[c]..own_offset[c + 1] {
            let q = own_child[i as usize] as usize;
            src_w[cursor[q] as usize] = own_w[i as usize];
            cursor[q] += 1;
        }
    }

    Ok(Plan {
        after,
        mesh: mesh_new,
        map: Map {
            n_old,
            n_new,
            src_offset,
            src_cell,
            src_w,
            own_offset,
            own_child,
            own_w,
        },
        target,
        n_refined,
        n_coarsened,
        n_kept,
        balance_sweeps: sweeps,
        promoted,
        cancelled_coarsen,
    })
}

// ===========================================================================
//  The criteria
// ===========================================================================

/// The noise filter of the Loehner ratio. Loehner's own value for the
/// equivalent term is 0.01; below it the indicator chases round-off in a
/// smooth region and the mesh refines everywhere.
pub const LOEHNER_EPS: Scalar = 0.01;

/// The Loehner second-derivative indicator, restated for a cell-centred
/// finite-volume mesh - SPEC-LIT section 75.3(a). **DESIGN**: the ratio is
/// Loehner's; the surface-integral form of the numerator and denominator is
/// this project's, because Loehner wrote his for nodal shape functions.
///
/// ```text
/// N_P = | sum_f Sf . [ (grad phi)_nbr(f) - (grad phi)_own(f) ] |
/// D_P = sum_f |Sf| ( |nf.(grad phi)_N| + |nf.(grad phi)_P| )
///     + eps sum_f |Sf| Delta_f ( |phi_N| + |phi_P| )
/// E_P = N_P / max(D_P, tiny)
/// ```
///
/// The numerator carries no owner sign, and that is the point rather than an
/// omission: written with the OUTWARD area vector the term is
/// `s_f (s_f Sf) . [grad_M - grad_P]`, and the two sign flips cancel, so both
/// cells of a face receive `+Sf.(grad_nbr - grad_own)`. Subtracting on the
/// neighbour side instead - which is what a careless transcription does -
/// turns the numerator into a third-derivative measure that is blind to a
/// parabola.
///
/// A boundary face reads the cell's own gradient on both sides, so it adds
/// nothing to the numerator and its full share to the denominator: the
/// indicator is damped at a wall rather than excited by it.
///
/// `E_P` lies in `[0, 1]` by construction - the numerator's every term is
/// bounded by the matching pair in the denominator - which is what makes one
/// threshold mean the same thing for every field.
/// `tests::the_loehner_indicator_is_bounded_and_blind_to_a_linear_field` is
/// that statement.
#[allow(clippy::too_many_arguments)]
pub fn loehner_indicator(
    out: &mut Vec<Scalar>,
    phi: &[Scalar],
    bphi: &[Scalar],
    grad: &[Vec3],
    m: &HostMesh,
    eps: Scalar,
) {
    out.clear();
    out.resize(m.n_cells, 0.0);
    let mut num = vec![0.0 as Scalar; m.n_cells];
    let mut den = vec![0.0 as Scalar; m.n_cells];

    // Scatter over faces - the reference shape of `reference.rs`, deliberately
    // the opposite of the kernel's gather.
    for f in 0..m.n_internal_faces {
        let p = m.owner[f] as usize;
        let n = m.neighbour[f] as usize;
        let sf = m.sf[f];
        let mag = m.mag_sf[f];
        let nf = if mag > 0.0 { sf / mag } else { Vec3::ZERO };
        // Both cells get the SAME contribution, and that is not a slip. In
        // outward-normal form the term is `s_f (s_f Sf) . [grad_M - grad_P]`
        // and the two sign flips cancel: on the owner side `s_f = +1` and
        // `M = n`, on the neighbour side `s_f = -1` and `M = p`, and both
        // reduce to `+Sf . (grad_n - grad_p)`. A version that subtracts on
        // the neighbour side measures a THIRD derivative, not a second.
        let jump = sf.dot(grad[n] - grad[p]);
        num[p] += jump;
        num[n] += jump;
        let d = mag * (nf.dot(grad[n]).abs() + nf.dot(grad[p]).abs())
            + eps * mag * m.delta_coeffs[f] * (phi[n].abs() + phi[p].abs());
        den[p] += d;
        den[n] += d;
    }
    // `bf` indexes six arrays and the patch lookup, not one, so an
    // `enumerate` over any single array would be a step backwards.
    #[allow(clippy::needless_range_loop)]
    for bf in 0..m.n_boundary_faces {
        if crate::reference::is_empty_face(m, bf) {
            continue;
        }
        let p = m.b_face_cells[bf] as usize;
        let mag = m.b_mag_sf[bf];
        let nf = if mag > 0.0 { m.b_sf[bf] / mag } else { Vec3::ZERO };
        den[p] += mag * 2.0 * nf.dot(grad[p]).abs()
            + eps * mag * m.b_delta_coeffs[bf] * (phi[p].abs() + bphi[bf].abs());
    }

    for c in 0..m.n_cells {
        let d = den[c];
        out[c] = if d > Scalar::MIN_POSITIVE { num[c].abs() / d } else { 0.0 };
    }
}

/// The characteristic fire diameter, FDS User's Guide, "Mesh Resolution":
///
/// ```text
/// D* = ( Qdot / (rho_inf cp_inf T_inf sqrt(g)) )^(2/5)
/// ```
///
/// `q_dot` is the total heat release rate in watts - `sum_P V_P qdot'''_P`.
/// Returns zero for a non-positive heat release, which is the honest answer:
/// with no fire there is no fire length scale.
pub fn d_star(q_dot: Scalar, rho_inf: Scalar, cp_inf: Scalar, t_inf: Scalar, g: Scalar) -> Scalar {
    let den = rho_inf * cp_inf * t_inf * g.max(0.0).sqrt();
    if q_dot <= 0.0 || den <= 0.0 {
        return 0.0;
    }
    (q_dot / den).powf(0.4)
}

/// The fire resolution indicator: `1` where the cell is inside the reacting
/// region and too coarse for `D*`, `0` elsewhere - SPEC-LIT section 75.3(b).
///
/// `n_star` is the number of cells wanted across `D*`; the FDS User's Guide
/// puts a well-resolved fire at 16 and calls 4 the coarse end of usable, so
/// `n_star = 16` is this crate's default and is not a free parameter dressed
/// up as one.
///
/// The cell size is `V_P^(1/3)`, which is the edge length for a cube and the
/// equivalent edge length for anything else.
pub fn fire_resolution_indicator(
    out: &mut Vec<Scalar>,
    v: &[Scalar],
    burning: &[Scalar],
    d_star: Scalar,
    n_star: Scalar,
) {
    out.clear();
    out.resize(v.len(), 0.0);
    if d_star <= 0.0 || n_star <= 0.0 {
        return;
    }
    let want = d_star / n_star;
    for c in 0..v.len() {
        if burning[c] > 0.0 && v[c].cbrt() > want {
            out[c] = 1.0;
        }
    }
}

// ===========================================================================
//  The device side
// ===========================================================================

/// The adapt kernels: the indicator, the balance sweep, the transfer and the
/// addressing rebuild.
///
/// Every one of them is a gather - one thread per cell, reading a CSR - and
/// none of them uses an atomic of any width. That is not an accident of this
/// implementation: SPEC-LIT section 75.5 is the argument that the whole adapt
/// can be written that way, and this struct is what it is arguing about.
pub struct AdaptKernels {
    pub loehner: cudarc::driver::CudaFunction,
    pub balance_sweep: cudarc::driver::CudaFunction,
    pub fire_resolution: cudarc::driver::CudaFunction,
    pub parent_targets: cudarc::driver::CudaFunction,
    pub limiter: cudarc::driver::CudaFunction,
    pub transfer_density: cudarc::driver::CudaFunction,
    pub transfer_scalar: cudarc::driver::CudaFunction,
    pub cell_face_csr: cudarc::driver::CudaFunction,
    pub boundary_csr: cudarc::driver::CudaFunction,
}

impl AdaptKernels {
    pub fn new(gpu: &Gpu) -> Result<Self> {
        let ks = KernelSet::new(gpu, crate::kernels::ADAPT)?;
        Ok(Self {
            loehner: ks.func("adaptLoehner")?,
            balance_sweep: ks.func("adaptBalanceSweep")?,
            fire_resolution: ks.func("adaptFireResolution")?,
            parent_targets: ks.func("adaptParentTargets")?,
            limiter: ks.func("adaptLimiter")?,
            transfer_density: ks.func("adaptTransferDensity")?,
            transfer_scalar: ks.func("adaptTransferScalar")?,
            cell_face_csr: ks.func("adaptCellFaceCsr")?,
            boundary_csr: ks.func("adaptBoundaryCsr")?,
        })
    }
}

/// The Loehner indicator on the device: one thread per cell, gathering over
/// the cell -> face CSR. Same arithmetic as [`loehner_indicator`], opposite
/// loop shape.
#[allow(clippy::too_many_arguments)]
pub fn gpu_loehner_indicator(
    gpu: &Gpu,
    k: &AdaptKernels,
    out: &mut DevBuf<Scalar>,
    phi: &DevBuf<Scalar>,
    bphi: &DevBuf<Scalar>,
    grad: &DevBuf<Vec3>,
    m: &GpuMesh,
    eps: Scalar,
) -> Result<()> {
    if m.n_cells == 0 {
        return Ok(());
    }
    let nl = m.n_cells as Label;
    unsafe {
        gpu.stream()
            .launch_builder(&k.loehner)
            .arg(&mut *out)
            .arg(phi)
            .arg(bphi)
            .arg(grad)
            .arg(&m.sf)
            .arg(&m.mag_sf)
            .arg(&m.delta_coeffs)
            .arg(&m.b_sf)
            .arg(&m.b_mag_sf)
            .arg(&m.b_delta_coeffs)
            .arg(&m.b_kind)
            .arg(&m.owner)
            .arg(&m.neighbour)
            .arg(&m.cf_offset)
            .arg(&m.cf_face)
            .arg(&m.cf_own)
            .arg(&m.bcf_offset)
            .arg(&m.bcf_face)
            .arg(&eps)
            .arg(&nl)
            .launch(cfg_for(m.n_cells))?;
    }
    Ok(())
}

/// One 2:1 balance sweep on the device: reads `target`, writes `out`, and the
/// caller swaps. An in-place sweep would race, and although the FIXED POINT of
/// that race would still be the same - levels only rise and integer `max` is
/// associative - the number of sweeps to reach it would not be reproducible.
///
/// `changed` is a single `Label` the caller reads back to decide whether to
/// sweep again: one four-byte copy per sweep, and at most `LEVEL_MAX` sweeps
/// per adapt.
pub fn gpu_balance_sweep(
    gpu: &Gpu,
    k: &AdaptKernels,
    target: &DevBuf<Label>,
    out: &mut DevBuf<Label>,
    changed: &mut DevBuf<Label>,
    m: &GpuMesh,
) -> Result<()> {
    if m.n_cells == 0 {
        return Ok(());
    }
    let nl = m.n_cells as Label;
    unsafe {
        gpu.stream()
            .launch_builder(&k.balance_sweep)
            .arg(target)
            .arg(&mut *out)
            .arg(&mut *changed)
            .arg(&m.owner)
            .arg(&m.neighbour)
            .arg(&m.cf_offset)
            .arg(&m.cf_face)
            .arg(&m.cf_own)
            .arg(&nl)
            .launch(cfg_for(m.n_cells))?;
    }
    Ok(())
}

/// The fire resolution indicator on the device: elementwise, and the one
/// criterion here that needs a global reduction first (`q_dot`), which the
/// caller supplies as a scalar.
#[allow(clippy::too_many_arguments)]
pub fn gpu_fire_resolution_indicator(
    gpu: &Gpu,
    k: &AdaptKernels,
    out: &mut DevBuf<Scalar>,
    v: &DevBuf<Scalar>,
    burning: &DevBuf<Scalar>,
    d_star: Scalar,
    n_star: Scalar,
    n_cells: usize,
) -> Result<()> {
    if n_cells == 0 {
        return Ok(());
    }
    let nl = n_cells as Label;
    unsafe {
        gpu.stream()
            .launch_builder(&k.fire_resolution)
            .arg(&mut *out)
            .arg(v)
            .arg(burning)
            .arg(&d_star)
            .arg(&n_star)
            .arg(&nl)
            .launch(cfg_for(n_cells))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const CUBE: Vec3 = Vec3::new(0.25, 0.25, 0.25);

    fn box_forest(n: [usize; 3]) -> Forest {
        Forest::uniform(n, CUBE).unwrap()
    }

    /// The emitter in this module and the static generator of SPEC-LIT
    /// section 74 must agree BIT FOR BIT on every mesh the static generator
    /// can express - otherwise "the adapt produces the mesh the generator
    /// would have produced" is a claim about two different mesh generators.
    #[test]
    fn the_forest_emitter_reproduces_the_static_generator() {
        for (n, lev) in [
            ([4usize, 4, 4], vec![0u32; 64]),
            ([4, 4, 4], {
                let mut l = vec![0u32; 64];
                for k in 0..4 {
                    for j in 0..4 {
                        for i in 2..4 {
                            l[i + 4 * (j + 4 * k)] = 1;
                        }
                    }
                }
                l
            }),
            ([6, 6, 6], {
                let mut l = vec![0u32; 216];
                l[2 + 6 * (2 + 6 * 2)] = 2;
                l
            }),
        ] {
            let a = Forest::from_base_levels(n, CUBE, &lev).unwrap().build().unwrap();
            let b = crate::mesh::refined::build(n, CUBE, &lev).unwrap();
            assert_eq!(a.mesh.n_cells, b.mesh.n_cells);
            assert_eq!(a.mesh.n_internal_faces, b.mesh.n_internal_faces);
            assert_eq!(a.mesh.n_boundary_faces, b.mesh.n_boundary_faces);
            assert_eq!(a.mesh.n_points, b.mesh.n_points);
            assert_eq!(a.points, b.points, "points differ for {n:?}");
            assert_eq!(a.faces, b.faces, "face point lists differ for {n:?}");
            assert_eq!(a.mesh.owner, b.mesh.owner);
            assert_eq!(a.mesh.neighbour, b.mesh.neighbour);
            assert_eq!(a.mesh.v, b.mesh.v, "volumes differ for {n:?}");
            assert_eq!(a.mesh.c, b.mesh.c);
            assert_eq!(a.mesh.sf, b.mesh.sf);
            assert_eq!(a.mesh.weights, b.mesh.weights);
            assert_eq!(a.mesh.delta_coeffs, b.mesh.delta_coeffs);
            assert_eq!(a.mesh.non_orth_corr, b.mesh.non_orth_corr);
            assert_eq!(a.mesh.skew_corr, b.mesh.skew_corr);
            assert_eq!(a.mesh.b_face_cells, b.mesh.b_face_cells);
            assert_eq!(a.mesh.cf_offset, b.mesh.cf_offset);
            assert_eq!(a.mesh.cf_face, b.mesh.cf_face);
            assert_eq!(a.level, b.level);
        }
    }

    /// The centre a `Forest` computes from integer arithmetic and the centre
    /// the geometry sweep computes from the emitted polygons are the same
    /// point. They are computed by completely different routes, so this is a
    /// real cross-check of the emitter's placement of every leaf.
    #[test]
    fn the_forest_and_the_geometry_agree_on_where_every_cell_is() {
        let at = |i: usize, j: usize, k: usize| i + 4 * (j + 4 * k);
        let mut lev = vec![0u32; 64];
        lev[at(1, 1, 1)] = 2;
        let f = Forest::from_base_levels([4, 4, 4], CUBE, &lev).unwrap();
        let r = f.build().unwrap();
        for c in 0..f.len() {
            let a = f.centre(c);
            let b = r.mesh.c[c];
            assert!(
                (a - b).mag() < 1e-14,
                "cell {c}: forest says {a}, geometry says {b}"
            );
            let s = f.size(c);
            assert!(
                (s.x * s.y * s.z - r.mesh.v[c]).abs() < 1e-14 * r.mesh.v[c],
                "cell {c} volume"
            );
        }
    }

    /// An adapt that refines a block and one that refines nothing, and what
    /// the plan says about each.
    #[test]
    fn a_refine_makes_eight_cells_where_there_was_one() {
        let f = box_forest([4, 4, 4]);
        let m = f.build().unwrap();
        let mut mark = vec![Mark::Keep; f.len()];
        mark[21] = Mark::Refine;
        let p = plan(&f, &m.mesh, &mark, 2).unwrap();
        assert_eq!(p.n_refined, 1);
        assert_eq!(p.n_coarsened, 0);
        assert_eq!(p.after.len(), f.len() + 7);
        assert_eq!(p.mesh.max_level_jump(), 1, "the adapt must stay 2:1 balanced");
        assert_eq!(p.map.own_offset[22] - p.map.own_offset[21], 8);

        let none = plan(&f, &m.mesh, &vec![Mark::Keep; f.len()], 2).unwrap();
        assert!(none.is_identity());
        assert_eq!(none.after.len(), f.len());
    }

    /// A coarsen only happens for a COMPLETE family. Seven of eight siblings
    /// marked must leave the mesh alone.
    #[test]
    fn a_coarsen_needs_all_eight_siblings() {
        let f = Forest::from_base_levels([2, 2, 2], CUBE, &[1, 1, 1, 1, 1, 1, 1, 1]).unwrap();
        let m = f.build().unwrap();
        assert_eq!(f.len(), 64);

        // Seven of the first family.
        let mut mark = vec![Mark::Keep; f.len()];
        for m in mark.iter_mut().take(7) {
            *m = Mark::Coarsen;
        }
        let p = plan(&f, &m.mesh, &mark, 2).unwrap();
        assert!(p.is_identity(), "seven siblings are not a family");

        // All eight.
        let mut mark = vec![Mark::Keep; f.len()];
        for m in mark.iter_mut().take(8) {
            *m = Mark::Coarsen;
        }
        let p = plan(&f, &m.mesh, &mark, 2).unwrap();
        assert_eq!(p.n_coarsened, 8);
        assert_eq!(p.after.len(), f.len() - 7);
        assert_eq!(p.mesh.max_level_jump(), 1);
        // The eight old cells all point at one new cell.
        let q = p.map.own_child[0];
        for c in 0..8 {
            assert_eq!(p.map.own_child[p.map.own_offset[c] as usize], q);
        }
        assert_eq!(p.map.src_offset[q as usize + 1] - p.map.src_offset[q as usize], 8);
    }

    /// A family is grouped by its PARENT, and a parent is a level as well as a
    /// corner.
    ///
    /// `Leaf::key` shifts the octant onto a common grid, so a level-1 parent
    /// at octant `(0,0,0)` and a level-0 parent at octant `(0,0,0)` have the
    /// same corner. Keying the family map on the corner alone merges those two
    /// buckets, and a legitimate eight-member family lands in a bucket of
    /// fifteen and is refused. The refusal is conservative rather than wrong,
    /// which is exactly why it needs a test: it shows up only where three
    /// levels meet inside one base cell.
    #[test]
    fn a_family_is_not_confused_with_its_grandparent() {
        let f = Forest::from_base_levels([2, 2, 2], CUBE, &[1; 8]).unwrap();
        let m = f.build().unwrap();
        assert_eq!(f.len(), 64);

        // Refine one level-1 leaf, so base cell 0 holds seven level-1 leaves
        // and eight level-2 ones.
        let mut mark = vec![Mark::Keep; f.len()];
        mark[0] = Mark::Refine;
        let p = plan(&f, &m.mesh, &mark, 2).unwrap();
        assert_eq!(p.after.len(), 71);
        let mid = p.after;
        let mm = p.mesh;

        // Now coarsen everything in base cell 0: the eight level-2 leaves are
        // a complete family and must go back to level 1; the seven level-1
        // leaves are not a family and must not move.
        let mark: Vec<Mark> = mid
            .leaves()
            .iter()
            .map(|l| if l.base == 0 { Mark::Coarsen } else { Mark::Keep })
            .collect();
        assert_eq!(mark.iter().filter(|m| **m == Mark::Coarsen).count(), 15);
        let p2 = plan(&mid, &mm.mesh, &mark, 2).unwrap();
        assert_eq!(
            p2.n_coarsened, 8,
            "the level-2 family must coarsen; a corner-only family key refuses it"
        );
        assert_eq!(p2.after.len(), 64);
        assert!(p2.after.leaves().iter().all(|l| l.level == 1));
    }

    /// 2:1 balance promotes cells the criterion never marked, and refuses a
    /// coarsen that would break the condition. Both are measured here, on a
    /// mesh built so that both must happen.
    #[test]
    fn balance_promotes_and_cancels() {
        // A level-1 core inside a level-0 box; coarsening only the core's
        // outermost family would leave a 2:1 jump nowhere - so instead refine
        // one cell two levels away and watch the promotion ripple.
        let mut lev = vec![0u32; 216];
        lev[3 + 6 * (3 + 6 * 3)] = 1;
        let f = Forest::from_base_levels([6, 6, 6], CUBE, &lev).unwrap();
        let m = f.build().unwrap();

        // Refine every level-1 leaf. Its level-0 face neighbours must be
        // promoted to level 1 by balance, though nothing marked them.
        let mark: Vec<Mark> = (0..f.len())
            .map(|c| if f.level(c) == 1 { Mark::Refine } else { Mark::Keep })
            .collect();
        let p = plan(&f, &m.mesh, &mark, 3).unwrap();
        assert!(p.promoted > 0, "balance must promote the ring of neighbours");
        assert_eq!(p.mesh.max_level_jump(), 1);

        // And the other half: a coarsen that balance must refuse.
        //
        // A level-2 island forces a ring of level-1 base cells around it, by
        // 2:1 balance on the base grid. Coarsening one of those rings'
        // families down to level 0 would leave a jump of TWO onto the island,
        // so the fixed point must raise it back and cancel the family.
        let mut lev = vec![0u32; 216];
        lev[3 + 6 * (3 + 6 * 3)] = 2;
        let f = Forest::from_base_levels([6, 6, 6], CUBE, &lev).unwrap();
        let m = f.build().unwrap();
        let victim = f.leaves().iter().find(|l| l.level == 1).unwrap().base;
        let mut mark = vec![Mark::Keep; f.len()];
        for (c, mk) in mark.iter_mut().enumerate() {
            if f.leaves()[c].base == victim {
                *mk = Mark::Coarsen;
            }
        }
        assert_eq!(mark.iter().filter(|m| **m == Mark::Coarsen).count(), 8);
        let p = plan(&f, &m.mesh, &mark, 3).unwrap();
        assert_eq!(p.cancelled_coarsen, 1, "the coarsen must be cancelled");
        assert!(p.is_identity(), "a cancelled coarsen must leave the mesh alone");
        assert_eq!(p.mesh.max_level_jump(), 1);
    }

    /// The balance sweep's fixed point does not depend on the order the faces
    /// are visited. Run it on the mesh, and on the mesh with its face list
    /// reversed, and require the same levels.
    #[test]
    fn the_balance_fixed_point_is_order_independent() {
        let mut lev = vec![0u32; 216];
        lev[3 + 6 * (3 + 6 * 3)] = 2;
        let f = Forest::from_base_levels([6, 6, 6], CUBE, &lev).unwrap();
        let m = f.build().unwrap();
        let mut a: Vec<u32> = f.levels();
        a[0] = 2;
        let mut b = a.clone();

        let mut ro: Vec<Label> = m.mesh.owner.clone();
        let mut rn: Vec<Label> = m.mesh.neighbour.clone();
        ro.reverse();
        rn.reverse();

        while balance_sweep(&mut a, &m.mesh.owner, &m.mesh.neighbour) > 0 {}
        while balance_sweep(&mut b, &ro, &rn) > 0 {}
        assert_eq!(a, b);
    }

    /// The indicator is bounded by one, and is blind to a linear field - a
    /// second-derivative indicator that fires on a constant gradient would
    /// refine a uniform shear flow everywhere.
    #[test]
    fn the_loehner_indicator_is_bounded_and_blind_to_a_linear_field() {
        let f = box_forest([6, 6, 6]);
        let r = f.build().unwrap();
        let m = &r.mesh;

        let lin = |p: Vec3| 2.0 * p.x - 3.0 * p.y + 0.5 * p.z;
        let phi: Vec<Scalar> = m.c.iter().map(|&p| lin(p)).collect();
        let bphi: Vec<Scalar> = m.b_cf.iter().map(|&p| lin(p)).collect();
        let grad = vec![Vec3::new(2.0, -3.0, 0.5); m.n_cells];
        let mut e = Vec::new();
        loehner_indicator(&mut e, &phi, &bphi, &grad, m, LOEHNER_EPS);
        let worst = e.iter().cloned().fold(0.0 as Scalar, Scalar::max);
        assert!(worst < 1e-12, "a linear field must not be marked; got {worst:e}");

        // A step, and a real gradient field for it.
        let step: Vec<Scalar> = m.c.iter().map(|&p| if p.x > 0.75 { 1.0 } else { 0.0 }).collect();
        let bstep: Vec<Scalar> =
            m.b_cf.iter().map(|&p| if p.x > 0.75 { 1.0 } else { 0.0 }).collect();
        let mut g = Vec::new();
        crate::reference::fvc_grad_scalar(&mut g, &step, &bstep, m);
        let mut e = Vec::new();
        loehner_indicator(&mut e, &step, &bstep, &g, m, LOEHNER_EPS);
        for (c, &ec) in e.iter().enumerate() {
            assert!(
                (0.0..=1.0 + 1e-12).contains(&ec),
                "cell {c}: indicator {ec} is outside [0,1]"
            );
        }
        assert!(
            e.iter().cloned().fold(0.0 as Scalar, Scalar::max) > 0.2,
            "a step must be marked"
        );
    }

    /// The hysteresis band is required, not advised - a caller that asks for a
    /// narrower one is refused by name.
    #[test]
    fn a_narrow_hysteresis_band_is_refused_by_name() {
        let e = vec![0.5 as Scalar; 4];
        let l = vec![1u32; 4];
        let err = mark_with_hysteresis(&e, &l, 0.4, 0.2, 3).unwrap_err().to_string();
        assert!(
            err.contains("hysteresis band is too narrow") && err.contains("alternate adapts"),
            "{err}"
        );
        assert!(mark_with_hysteresis(&e, &l, 0.4, 0.1, 3).is_ok());
    }

    /// The fixed-point requirement of SPEC-LIT section 75.3(c): on a field
    /// that does not change, repeated adapts must stop changing the mesh.
    #[test]
    fn hysteresis_reaches_a_fixed_point_on_a_steady_field() {
        let mut f = box_forest([6, 6, 6]);
        let blob = |p: Vec3| {
            let r = ((p.x - 0.75).powi(2) + (p.y - 0.75).powi(2) + (p.z - 0.75).powi(2)).sqrt();
            (-((r / 0.25).powi(2))).exp()
        };
        let mut counts = Vec::new();
        for _ in 0..6 {
            let r = f.build().unwrap();
            let m = &r.mesh;
            let phi: Vec<Scalar> = m.c.iter().map(|&p| blob(p)).collect();
            let bphi: Vec<Scalar> = m.b_cf.iter().map(|&p| blob(p)).collect();
            let mut g = Vec::new();
            crate::reference::fvc_grad_scalar(&mut g, &phi, &bphi, m);
            let mut e = Vec::new();
            loehner_indicator(&mut e, &phi, &bphi, &g, m, LOEHNER_EPS);
            let mark = mark_with_hysteresis(&e, &f.levels(), 0.20, 0.05, 2).unwrap();
            let p = plan(&f, m, &mark, 2).unwrap();
            counts.push(p.after.len());
            f = p.after;
        }
        assert_eq!(
            counts[counts.len() - 1],
            counts[counts.len() - 2],
            "the cell count must stop moving on a steady field: {counts:?}"
        );
    }

    fn gpu() -> Option<Gpu> {
        Gpu::new(0).ok()
    }

    /// SPEC-LIT section 75.9's bit-identical-defaults claim, asserted rather
    /// than promised.
    ///
    /// This section adds no case-file setting, so the section 13.4.1 pair test
    /// has nothing to pair. What it asserts instead is the stronger statement
    /// the pair test would only sample: **no shipped case can produce a
    /// different answer because no code path reaches this module.** The
    /// crate's own sources are walked and required not to name it, outside
    /// this module, the module declaration and the validation binary.
    #[test]
    fn no_time_loop_reaches_the_adapt() {
        use std::path::{Path, PathBuf};
        fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
            let Ok(rd) = std::fs::read_dir(dir) else { return };
            for e in rd.flatten() {
                let p = e.path();
                if p.is_dir() {
                    walk(&p, out);
                } else if p.extension().and_then(|s| s.to_str()) == Some("rs") {
                    out.push(p);
                }
            }
        }
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
        let mut files = Vec::new();
        walk(&root, &mut files);
        files.sort();

        // Matched on the PATH and not on the file name. A bare-name list would
        // exempt a future `src/parcels/transfer.rs` from the claim without
        // anyone noticing, which is the failure mode this whole test exists to
        // rule out.
        let allowed = [
            "adapt.rs",
            "adapt/rebuild.rs",
            "adapt/transfer.rs",
            "bin/validate.rs",
            "lib.rs",
        ];
        let mut offenders = Vec::new();
        for p in &files {
            let rel = p
                .strip_prefix(&root)
                .unwrap_or(p)
                .display()
                .to_string()
                .replace(std::path::MAIN_SEPARATOR, "/");
            if allowed.contains(&rel.as_str()) {
                continue;
            }
            // Non-comment lines only. A doc comment pointing here is prose,
            // not a code path, and `mesh::refined` carries one deliberately.
            // Same shape as `validate.rs`'s own verdict-registry test.
            let t = std::fs::read_to_string(p).unwrap_or_default();
            if t.lines().any(|l| {
                let l = l.trim_start();
                !l.starts_with("//") && (l.contains("crate::adapt") || l.contains("ofgpu::adapt"))
            }) {
                offenders.push(rel);
            }
        }
        assert!(
            offenders.is_empty(),
            "SPEC-LIT section 75.9 claims nothing in a time loop reaches the adapt, \
             but {} file(s) name it:\n  {}",
            offenders.len(),
            offenders.join("\n  ")
        );
    }

    /// Every kernel in `adapt.cu`, against the host reference of the same
    /// arithmetic, on a mesh with 2:1 interfaces. The two loops are different
    /// shapes - the reference scatters over faces and over old cells, the
    /// kernels gather one thread per cell - so agreement is evidence and not
    /// a tautology.
    #[test]
    fn the_device_agrees_with_the_host_on_every_adapt_kernel() {
        let Some(g) = gpu() else { return };
        let k = AdaptKernels::new(&g).expect("the adapt kernels must load");
        use crate::adapt::rebuild;
        use crate::adapt::transfer::{self, Prolongation};

        let mut lev = vec![0u32; 216];
        lev[3 + 6 * (3 + 6 * 3)] = 1;
        let f = Forest::from_base_levels([6, 6, 6], CUBE, &lev).unwrap();
        let r = f.build().unwrap();
        let m = &r.mesh;
        let gm = GpuMesh::upload(&g, m).unwrap();

        let blob = |p: Vec3| {
            let d = ((p.x - 0.6).powi(2) + (p.y - 0.5).powi(2) + (p.z - 0.4).powi(2)).sqrt();
            0.3 + (-((d / 0.3).powi(2))).exp()
        };
        let phi: Vec<Scalar> = m.c.iter().map(|&p| blob(p)).collect();
        let bphi: Vec<Scalar> = m.b_cf.iter().map(|&p| blob(p)).collect();
        let rho: Vec<Scalar> = m.c.iter().map(|&p| 1.0 + 0.2 * p.x).collect();
        let mut grad = Vec::new();
        crate::reference::fvc_grad_scalar(&mut grad, &phi, &bphi, m);

        let d_phi = g.upload(&phi).unwrap();
        let d_bphi = g.upload(&bphi).unwrap();
        let d_rho = g.upload(&rho).unwrap();
        let d_grad = g.upload(&grad).unwrap();

        let worst = |a: &[Scalar], b: &[Scalar]| -> Scalar {
            a.iter()
                .zip(b.iter())
                .map(|(x, y)| (x - y).abs() / (1.0 as Scalar).max(x.abs()))
                .fold(0.0 as Scalar, Scalar::max)
        };

        // ---- the indicator -------------------------------------------------
        let mut e = Vec::new();
        loehner_indicator(&mut e, &phi, &bphi, &grad, m, LOEHNER_EPS);
        let mut d_e: DevBuf<Scalar> = g.zeros(m.n_cells).unwrap();
        gpu_loehner_indicator(&g, &k, &mut d_e, &d_phi, &d_bphi, &d_grad, &gm, LOEHNER_EPS)
            .unwrap();
        let got = g.download(&d_e).unwrap();
        assert!(worst(&e, &got) < 1e-12, "Loehner: {:e}", worst(&e, &got));

        // ---- the fire resolution indicator ---------------------------------
        let burn: Vec<Scalar> =
            (0..m.n_cells).map(|c| if c % 3 == 0 { 1.0 } else { 0.0 }).collect();
        let ds = d_star(5.0e5, 1.2, 1005.0, 293.0, 9.81);
        let mut fr = Vec::new();
        fire_resolution_indicator(&mut fr, &m.v, &burn, ds, 4.0);
        let d_v = g.upload(&m.v).unwrap();
        let d_burn = g.upload(&burn).unwrap();
        let mut d_fr: DevBuf<Scalar> = g.zeros(m.n_cells).unwrap();
        gpu_fire_resolution_indicator(&g, &k, &mut d_fr, &d_v, &d_burn, ds, 4.0, m.n_cells)
            .unwrap();
        assert_eq!(fr, g.download(&d_fr).unwrap());

        // ---- the balance sweep ---------------------------------------------
        let mut want: Vec<u32> = f.levels();
        want[0] = 3;
        let mut host = want.clone();
        while balance_sweep(&mut host, &m.owner, &m.neighbour) > 0 {}
        let mut a: DevBuf<Label> =
            g.upload(&want.iter().map(|&l| l as Label).collect::<Vec<_>>()).unwrap();
        let mut b: DevBuf<Label> = g.zeros(m.n_cells).unwrap();
        let mut changed: DevBuf<Label> = g.zeros(1).unwrap();
        for _ in 0..(LEVEL_MAX as usize + 2) {
            g.fill_zero(&mut changed).unwrap();
            gpu_balance_sweep(&g, &k, &a, &mut b, &mut changed, &gm).unwrap();
            std::mem::swap(&mut a, &mut b);
            if g.download(&changed).unwrap()[0] == 0 {
                break;
            }
        }
        let dev: Vec<u32> = g.download(&a).unwrap().iter().map(|&x| x as u32).collect();
        assert_eq!(dev, host, "the device balance sweep must reach the host fixed point");

        // ---- the addressing rebuild ----------------------------------------
        let nbr = rebuild::neighbour_order(&m.neighbour, m.n_cells).unwrap();
        let bo = rebuild::boundary_order(&m.b_face_cells, m.n_cells).unwrap();
        let (d_np, d_nk) = (g.upload(&nbr.perm).unwrap(), g.upload(&nbr.key).unwrap());
        let (d_bp, d_bk) = (g.upload(&bo.perm).unwrap(), g.upload(&bo.key).unwrap());
        let csr =
            rebuild::gpu_rebuild_addressing(&g, &k, &gm, &d_np, &d_nk, &d_bp, &d_bk).unwrap();
        assert_eq!(g.download(&csr.cf_offset).unwrap(), m.cf_offset);
        assert_eq!(g.download(&csr.cf_face).unwrap(), m.cf_face);
        assert_eq!(g.download(&csr.cf_own).unwrap(), m.cf_own);
        assert_eq!(g.download(&csr.bcf_offset).unwrap(), m.bcf_offset);
        assert_eq!(g.download(&csr.bcf_face).unwrap(), m.bcf_face);

        // ---- the transfer ---------------------------------------------------
        let mark: Vec<Mark> = (0..f.len())
            .map(|c| if e[c] > 0.05 { Mark::Refine } else { Mark::Keep })
            .collect();
        let p = plan(&f, m, &mark, 2).unwrap();
        assert!(!p.is_identity(), "the device transfer test needs a real adapt");
        let nm = &p.mesh.mesh;
        let gmap = transfer::GpuMap::upload(&g, &p.map).unwrap();
        let d_cnew = g.upload(&nm.c).unwrap();
        let d_vold = g.upload(&m.v).unwrap();
        let d_vnew = g.upload(&nm.v).unwrap();

        let t = transfer::parent_targets(&p.map, &nm.c).unwrap();
        let mut d_xbar: DevBuf<Vec3> = g.zeros(p.map.n_old).unwrap();
        let mut d_wsum: DevBuf<Scalar> = g.zeros(p.map.n_old).unwrap();
        transfer::gpu_parent_targets(&g, &k, &mut d_xbar, &mut d_wsum, &gmap, &d_cnew)
            .unwrap();
        let gx = g.download(&d_xbar).unwrap();
        for (c, x) in gx.iter().enumerate() {
            assert!((*x - t.xbar[c]).mag() < 1e-14, "xbar cell {c}");
        }
        assert!(worst(&t.wsum, &g.download(&d_wsum).unwrap()) < 1e-15);

        let mut psi = Vec::new();
        transfer::barth_jespersen(&mut psi, &phi, &bphi, &grad, &t, &p.map, &nm.c, m);
        let mut d_psi: DevBuf<Scalar> = g.zeros(p.map.n_old).unwrap();
        transfer::gpu_barth_jespersen(
            &g, &k, &mut d_psi, &d_phi, &d_bphi, &d_grad, &d_xbar, &gmap, &d_cnew, &gm,
        )
        .unwrap();
        let gp = g.download(&d_psi).unwrap();
        assert!(worst(&psi, &gp) < 1e-13, "limiter: {:e}", worst(&psi, &gp));

        let mut rho_new = Vec::new();
        transfer::transfer_density(&mut rho_new, &rho, &m.v, &nm.v, &p.map).unwrap();
        let mut d_rn: DevBuf<Scalar> = g.zeros(p.map.n_new).unwrap();
        transfer::gpu_transfer_density(&g, &k, &mut d_rn, &d_rho, &d_vold, &d_vnew, &gmap)
            .unwrap();
        assert!(worst(&rho_new, &g.download(&d_rn).unwrap()) < 1e-14);

        for mode in [Prolongation::Constant, Prolongation::LimitedLinear] {
            let mut phi_new = Vec::new();
            transfer::transfer_scalar(
                &mut phi_new, &phi, &rho, &grad, &psi, &t, &m.v, &nm.v, &nm.c, &p.map, mode,
            )
            .unwrap();
            let mut d_pn: DevBuf<Scalar> = g.zeros(p.map.n_new).unwrap();
            transfer::gpu_transfer_scalar(
                &g, &k, &mut d_pn, &d_phi, &d_rho, &d_grad, &d_psi, &d_xbar, &d_vold,
                &d_vnew, &d_cnew, &gmap, mode,
            )
            .unwrap();
            let got = g.download(&d_pn).unwrap();
            assert!(worst(&phi_new, &got) < 1e-13, "{mode:?}: {:e}", worst(&phi_new, &got));
        }
    }

    /// D* against the FDS User's Guide's own worked arithmetic. A 1000 kW fire
    /// in air at 293 K has D* close to 0.96 m, and the resolution measure is
    /// the ratio of that to the cell size.
    #[test]
    fn the_fire_diameter_matches_the_published_formula() {
        let d = d_star(1.0e6, 1.2, 1005.0, 293.0, 9.81);
        assert!((d - 0.9612).abs() < 5e-3, "D* = {d}");
        assert_eq!(d_star(0.0, 1.2, 1005.0, 293.0, 9.81), 0.0);

        let v = vec![0.001 as Scalar, 0.001, 1.0e-6];
        let burn = vec![1.0 as Scalar, 0.0, 1.0];
        let mut out = Vec::new();
        fire_resolution_indicator(&mut out, &v, &burn, d, 16.0);
        assert_eq!(out, vec![1.0, 0.0, 0.0]);
    }
}
