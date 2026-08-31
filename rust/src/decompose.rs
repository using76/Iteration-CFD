// meteor-cfd - Copyright (c) 2026 주식회사 메테오시뮬레이션 (Meteo Simulation Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

//! Cutting a mesh into pieces, and the ghost layer that joins them back up.
//!
//! SPEC-LIT §71. Host side only: this module runs once, at set-up, and
//! produces `P` ordinary [`HostMesh`]es plus the plan for moving one cell's
//! worth of data across each cut. `src/halo.rs` executes that plan on the
//! device; nothing here touches a GPU.
//!
//! # What a decomposition is here
//!
//! Given a partition map `pi : cell -> part`, part `p` owns
//! `C_p = { c : pi(c) = p }` and needs a one-cell-deep **halo** `H_p` - every
//! cell on the other side of a cut face. Local cells are numbered owned-first:
//!
//! ```text
//! 0 .. n_p              owned, ascending in GLOBAL cell id
//! n_p .. n_p + h_p      halo,  ascending in (owning part, global cell id)
//! ```
//!
//! A cut internal face becomes one **boundary** face on each side, on a patch
//! whose `type_name` is `"processor"`, carrying the neighbouring cell's halo
//! index in `b_nbr_cell` - which is exactly the shape a cyclic couple already
//! has, and exactly what `lduAmul` already reads. A cyclic couple whose two
//! halves land on different parts needs no new patch at all: it stays on its
//! own patch with its own metrics, and only its `b_nbr_cell` moves into the
//! halo.
//!
//! # The one property everything else rests on
//!
//! The metrics of a cut face are **copied from the whole mesh**, never
//! recomputed. §71.3 is the argument: two parts computing a shared face's
//! `Sf` from their own point lists traverse the face in opposite windings and
//! agree to round-off but not to the bit, and one ulp in `Sf` is one ulp in
//! `upper[f]`, and then the decomposed matrix is not the serial matrix. This
//! module derives every part from an already-computed [`HostMesh`], so the
//! authority question has one answer - the whole mesh - and the copy is exact
//! by construction rather than by care.
//!
//! Provenance: ORIGINAL. The partition is a space-filling-curve sort, written
//! from Skilling (2004) and Butz (1971) - papers, and this project's
//! `LICENSING.md` permits papers; the halo build, the local renumbering and
//! the exchange plan are this project's own design. **METIS 5.2.x is
//! Apache-2.0** (verified 2026-08-31 from `LICENSE` at
//! `github.com/KarypisLab/METIS`) and could be linked, and is deliberately not
//! - §71.2 gives the reasons. No GPL-licensed source was consulted.

use crate::error::{Error, Result};
use crate::ldu::HostLduMatrix;
use crate::mesh::{HostMesh, PatchInfo, PatchKind};
use crate::{Label, Scalar, Vec3};

// ==========================================================================
//  The partition map
// ==========================================================================

/// How the cells are handed out to the parts.
///
/// Every method here is a **pure function of the mesh**, so the same mesh cut
/// into the same number of parts always gives the same map, on every machine
/// and in every build. That is not a nicety: a decomposed run is only
/// reproducible run to run if the decomposition is, and a partitioner that
/// consults a random seed or a hash-map iteration order would take that away
/// (§71.2).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum PartitionMethod {
    /// Cut the cell list into `P` runs of as near equal length as integer
    /// division allows: cell `c` goes to part `floor(c P / n)`.
    ///
    /// Crude, and on a mesh whose cells are numbered in a sensible sweep -
    /// everything [`crate::blockgen`] writes, and every polyMesh this crate
    /// reads - it is a slab decomposition along the slowest-varying index,
    /// which is a perfectly respectable cut. It exists mostly because it is
    /// the one map a reader can verify by eye.
    Linear,

    /// Sort the cells along a Hilbert space-filling curve through their
    /// centres, then cut the sorted list into `P` equal runs.
    ///
    /// The default. See [`hilbert_index`].
    #[default]
    Hilbert,

    /// A map supplied by the caller, `[n_cells]` long with every entry in
    /// `[0, P)`. Used by the tests to force partitions no sane partitioner
    /// would produce - which is the point, because a bad partition is a
    /// *harder* test of bitwise invariance than a good one.
    Explicit(Vec<Label>),
}

impl PartitionMethod {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Linear => "linear",
            Self::Hilbert => "hilbert",
            Self::Explicit(_) => "explicit",
        }
    }
}

/// Bits per axis in the Hilbert grid. `3 * 21 = 63`, so the index fits a
/// `u64` with a bit to spare; at 21 bits the grid is 2 M cells across each
/// axis, which is finer than any mesh this crate can hold in memory.
const HILBERT_BITS: u32 = 21;

/// The Hilbert index of an integer lattice point, `bits` per axis.
///
/// Written from Skilling, *Programming the Hilbert Curve*, AIP Conf. Proc.
/// **707**, 381-387 (2004), which gives the `n`-dimensional Gray-code
/// construction, and Butz, "Alternative Algorithm for Hilbert's Space-Filling
/// Curve", *IEEE Trans. Computers* **C-20**(4), 424-426 (1971), which is the
/// original. Both are papers; this project's `LICENSING.md` permits papers and
/// forbids reading GPL source, and no source was read for this.
///
/// The construction in three steps, which is why it is written out rather than
/// pulled from a crate:
///
/// 1. **Undo the reflections.** Walking the levels from coarse to fine, each
///    axis either inverts the leading axis or exchanges its low bits with it.
///    That converts the axis-aligned coordinates into the curve's own
///    "transpose" representation.
/// 2. **Gray-encode.** `x[i] ^= x[i-1]` for `i > 0`, then fold the last axis's
///    accumulated parity back into every axis. This is the step that makes
///    consecutive indices differ in exactly one bit, which is what makes
///    consecutive points *adjacent*.
/// 3. **Interleave.** Read the transpose most-significant-bit first, one bit
///    from each axis per level.
///
/// The adjacency property is what the sort is for: cells with nearby indices
/// are nearby in space, so cutting the sorted list into runs gives compact
/// parts with small surfaces, with no graph and no dependency.
/// `the_hilbert_curve_visits_every_point_and_never_jumps` checks both halves
/// of that claim - bijection and unit steps - rather than assuming them.
pub fn hilbert_index(coord: [u32; 3], bits: u32) -> u64 {
    debug_assert!((1..=21).contains(&bits));
    let mut x = coord;
    let m = 1u32 << (bits - 1);

    // 1. inverse undo
    let mut q = m;
    while q > 1 {
        let p = q - 1;
        for i in 0..3 {
            if x[i] & q != 0 {
                x[0] ^= p;
            } else {
                let t = (x[0] ^ x[i]) & p;
                x[0] ^= t;
                x[i] ^= t;
            }
        }
        q >>= 1;
    }

    // 2. Gray encode
    for i in 1..3 {
        x[i] ^= x[i - 1];
    }
    let mut t = 0u32;
    let mut q = m;
    while q > 1 {
        if x[2] & q != 0 {
            t ^= q - 1;
        }
        q >>= 1;
    }
    for xi in x.iter_mut() {
        *xi ^= t;
    }

    // 3. interleave, most significant level first
    let mut idx = 0u64;
    for b in (0..bits).rev() {
        for xi in x.iter() {
            idx = (idx << 1) | ((xi >> b) & 1) as u64;
        }
    }
    idx
}

/// Map a cell centre onto the Hilbert lattice.
///
/// The bounding box is taken over the cell centres, and a degenerate axis - a
/// 2-D case's empty direction, where every centre has the same coordinate -
/// collapses to lattice coordinate 0 rather than dividing by zero. A 2-D mesh
/// therefore gets a 2-D Hilbert cut, which is the right answer.
fn lattice(c: Vec3, lo: Vec3, span: Vec3, scale: Scalar) -> [u32; 3] {
    let one = |v: Scalar, l: Scalar, s: Scalar| -> u32 {
        // A degenerate axis - a 2-D case's empty direction - collapses to 0
        // rather than dividing by zero, and a NaN centre goes there too: a
        // partition has to be a total function of the mesh, so there is no
        // input for which this may fail to answer.
        if s.is_nan() || s <= 0.0 {
            return 0;
        }
        let u = ((v - l) / s * scale).floor();
        if u.is_nan() || u < 0.0 {
            0
        } else if u >= scale {
            (scale as u32) - 1
        } else {
            u as u32
        }
    };
    [
        one(c.x, lo.x, span.x),
        one(c.y, lo.y, span.y),
        one(c.z, lo.z, span.z),
    ]
}

/// Compute the partition map `pi : cell -> part`.
///
/// Returns `[n_cells]` with every entry in `[0, n_parts)`.
pub fn partition(m: &HostMesh, n_parts: usize, method: &PartitionMethod) -> Result<Vec<Label>> {
    let n = m.n_cells;
    if n_parts == 0 {
        return Err(Error::Config(
            "decompose: a mesh cannot be cut into 0 parts".to_string(),
        ));
    }
    if n_parts > n {
        return Err(Error::Config(format!(
            "decompose: {n_parts} parts asked for but the mesh has only {n} \
             cells; every part must own at least one cell"
        )));
    }

    match method {
        PartitionMethod::Explicit(v) => {
            if v.len() != n {
                return Err(Error::Config(format!(
                    "decompose: the explicit partition map has {} entries, but \
                     the mesh has {n} cells",
                    v.len()
                )));
            }
            let mut seen = vec![false; n_parts];
            for (c, &p) in v.iter().enumerate() {
                if p < 0 || p as usize >= n_parts {
                    return Err(Error::Config(format!(
                        "decompose: the explicit partition map sends cell {c} \
                         to part {p}, which is outside [0, {n_parts})"
                    )));
                }
                seen[p as usize] = true;
            }
            if let Some(p) = seen.iter().position(|&s| !s) {
                return Err(Error::Config(format!(
                    "decompose: the explicit partition map leaves part {p} \
                     empty; every part must own at least one cell"
                )));
            }
            Ok(v.clone())
        }

        PartitionMethod::Linear => Ok((0..n).map(|c| ((c * n_parts) / n) as Label).collect()),

        PartitionMethod::Hilbert => {
            if m.c.len() < n {
                return Err(Error::Config(format!(
                    "decompose: the Hilbert cut needs {n} cell centres, the \
                     mesh carries {}",
                    m.c.len()
                )));
            }
            let mut lo = m.c[0];
            let mut hi = m.c[0];
            for c in &m.c[..n] {
                lo.x = lo.x.min(c.x);
                lo.y = lo.y.min(c.y);
                lo.z = lo.z.min(c.z);
                hi.x = hi.x.max(c.x);
                hi.y = hi.y.max(c.y);
                hi.z = hi.z.max(c.z);
            }
            let span = hi - lo;
            let scale = (1u64 << HILBERT_BITS) as Scalar;

            // Ties are broken by the global cell id, so the sort is a total
            // order and the map is a function of the mesh alone. Two cells CAN
            // share a lattice point - a mesh with a wildly non-uniform spacing
            // - and without the tie-break the partition would depend on the
            // sort's implementation.
            let mut order: Vec<(u64, u32)> = (0..n)
                .map(|c| {
                    (
                        hilbert_index(lattice(m.c[c], lo, span, scale), HILBERT_BITS),
                        c as u32,
                    )
                })
                .collect();
            order.sort_unstable();

            let mut part = vec![0 as Label; n];
            for (rank, &(_, c)) in order.iter().enumerate() {
                part[c as usize] = ((rank * n_parts) / n) as Label;
            }
            Ok(part)
        }
    }
}

// ==========================================================================
//  The decomposition itself
// ==========================================================================

/// One part of a decomposed mesh, plus its half of the exchange plan.
///
/// [`Self::mesh`] is an ordinary [`HostMesh`]: it uploads with
/// [`crate::mesh::GpuMesh::upload`] and is applied by the same kernels. What
/// marks it as a part is that its `global_face` is not the identity - so the
/// merged row map of SPEC-LIT §70 puts every row back in whole-mesh order -
/// and that some of its `b_nbr_cell` entries point past `n_cells`, into the
/// halo.
#[derive(Debug, Clone)]
pub struct PartMesh {
    /// Which part this is, `0 .. n_parts`.
    pub part: usize,

    /// The part's own mesh. `mesh.n_cells` counts **owned** cells only.
    pub mesh: HostMesh,

    /// Halo depth: `mesh.n_cells .. mesh.n_cells + n_halo` are the ghost
    /// cells. Every field buffer this part solves on must be
    /// `n_cells + n_halo` long; every kernel still launches `n_cells` threads
    /// and still guards `if (c >= nCells) return;`, so no kernel that writes
    /// only owned cells needs to change at all.
    pub n_halo: usize,

    /// `[n_cells + n_halo]` the whole mesh's cell id of every local cell.
    /// The canonical key for every ordering decision in this module, and the
    /// map the gate uses to put a decomposed answer back in serial order.
    pub global_cell: Vec<Label>,

    /// The parts this part exchanges with, ascending. Fixed for the run.
    pub nbr_parts: Vec<Label>,

    /// `[nbr_parts.len() + 1]` CSR over [`Self::send_index`].
    pub send_offset: Vec<Label>,

    /// LOCAL owned-cell index of every cell this part sends. Slice `i` goes to
    /// `nbr_parts[i]`, ascending in global cell id.
    pub send_index: Vec<Label>,

    /// `[nbr_parts.len() + 1]` CSR over the halo, HALO-relative: the cells
    /// from `nbr_parts[i]` land at
    /// `n_cells + recv_offset[i] .. n_cells + recv_offset[i+1]`.
    ///
    /// They land there **directly**, which is the whole reason the halo is
    /// ordered by (owning part, global cell id): the receive is contiguous, so
    /// there is no unpack kernel and therefore no scatter anywhere in the
    /// exchange path.
    pub recv_offset: Vec<Label>,

    /// `(patch index in mesh.patches, neighbouring part)` for every processor
    /// patch, ascending in the neighbouring part - the same order as
    /// [`Self::nbr_parts`], though not necessarily the same length: a part can
    /// be a neighbour through a cut CYCLIC couple alone, which adds no
    /// processor patch.
    pub proc_patches: Vec<(usize, Label)>,
}

impl PartMesh {
    /// Total length every field buffer on this part must have.
    #[inline]
    pub fn n_local(&self) -> usize {
        self.mesh.n_cells + self.n_halo
    }
}

/// A whole mesh cut into `n_parts` pieces.
#[derive(Debug, Clone)]
pub struct Decomposition {
    pub n_parts: usize,
    /// `[n_global_cells]` the partition map itself.
    pub cell_part: Vec<Label>,
    pub parts: Vec<PartMesh>,

    pub n_global_cells: usize,
    pub n_global_internal_faces: usize,
    pub n_global_boundary_faces: usize,
    /// Internal faces whose two cells landed on different parts. Each becomes
    /// two boundary faces, so the decomposition holds
    /// `n_global_boundary_faces + 2 n_cut_faces` boundary faces in total.
    pub n_cut_faces: usize,
    /// Coupled boundary faces (cyclic, conjugate interface) whose partner cell
    /// landed on another part. These need no new patch - they keep their own -
    /// but they do put a cell in the halo.
    pub n_cut_couples: usize,
}

/// What [`Decomposition::report`] found: the numbers a reader wants before
/// believing a cut.
#[derive(Debug, Clone)]
pub struct DecompositionReport {
    pub n_parts: usize,
    pub cells: Vec<usize>,
    pub halo: Vec<usize>,
    pub nbrs: Vec<usize>,
    pub n_cut_faces: usize,
    pub n_cut_couples: usize,
    /// `max_p n_p / (n / P) - 1`, the fractional load imbalance.
    pub imbalance: Scalar,
    /// Halo cells summed over parts, as a fraction of the whole mesh. This is
    /// the communication volume of one exchange, in cells.
    pub halo_fraction: Scalar,
}

/// The per-part face and cell lists, bucketed once for the whole mesh.
struct PartLists {
    /// `[n_parts]` owned cells, ascending in global id.
    owned: Vec<Vec<Label>>,
    /// `[n_parts]` halo cells, ascending in (owning part, global id).
    need: Vec<Vec<Label>>,
    /// `[n_parts]` interior internal faces, ascending in global face id.
    interior: Vec<Vec<Label>>,
    /// `[n_parts]` `(neighbouring part, global internal face id)` for cut
    /// faces, ascending in (neighbouring part, face id).
    cut: Vec<Vec<(Label, Label)>>,
    /// `[n_parts]` inherited boundary faces, ascending in global boundary id.
    boundary: Vec<Vec<Label>>,
}

/// The shared, read-only state a single part's build needs.
struct PartBuild<'a> {
    m: &'a HostMesh,
    cell_part: &'a [Label],
    /// `[n_global_cells]` index of a cell within its own part's owned list.
    local_of: &'a [Label],
    /// `[n_global_cells]` index of a cell within THIS part's halo, or `-1`.
    /// Rewritten between parts; only the current part's entries are set.
    halo_of: &'a [Label],
}

impl Decomposition {
    /// Cut `m` into `n_parts` and build every part's mesh and exchange plan.
    pub fn build(m: &HostMesh, n_parts: usize, method: &PartitionMethod) -> Result<Self> {
        let cell_part = partition(m, n_parts, method)?;
        Self::from_map(m, n_parts, cell_part)
    }

    /// The same, from a partition map already computed.
    pub fn from_map(m: &HostMesh, n_parts: usize, cell_part: Vec<Label>) -> Result<Self> {
        let n = m.n_cells;
        let nif = m.n_internal_faces;
        let nbf = m.n_boundary_faces;

        if n_parts == 0 {
            return Err(Error::Config(
                "decompose: a mesh cannot be cut into 0 parts".to_string(),
            ));
        }
        if cell_part.len() != n {
            return Err(Error::Config(format!(
                "decompose: the partition map has {} entries, the mesh has {n} \
                 cells",
                cell_part.len()
            )));
        }
        check_lengths(m)?;
        for (c, &p) in cell_part.iter().enumerate() {
            if p < 0 || p as usize >= n_parts {
                return Err(Error::Config(format!(
                    "decompose: cell {c} is assigned to part {p}, outside \
                     [0, {n_parts})"
                )));
            }
        }

        // ---- owned cells, ascending in global id --------------------------
        let mut owned: Vec<Vec<Label>> = vec![Vec::new(); n_parts];
        let mut local_of = vec![-1 as Label; n];
        for (c, &p) in cell_part.iter().enumerate() {
            let p = p as usize;
            local_of[c] = owned[p].len() as Label;
            owned[p].push(c as Label);
        }
        if let Some(p) = owned.iter().position(|o| o.is_empty()) {
            return Err(Error::Config(format!(
                "decompose: part {p} owns no cells; every part must own at \
                 least one cell, or its kernels launch a zero-block grid"
            )));
        }

        // ---- the halo: every cell across a cut ----------------------------
        // Collected for EVERY part before any send list is built, because part
        // p's send list is read off part q's halo rather than recomputed -
        // which is what makes the plan symmetric with no handshake (§71.4).
        let mut need: Vec<Vec<Label>> = vec![Vec::new(); n_parts];
        let mut n_cut_faces = 0usize;
        for f in 0..nif {
            let o = m.owner[f];
            let nb = m.neighbour[f];
            let (po, pn) = (cell_part[o as usize], cell_part[nb as usize]);
            if po != pn {
                n_cut_faces += 1;
                need[po as usize].push(nb);
                need[pn as usize].push(o);
            }
        }
        let mut n_cut_couples = 0usize;
        for bf in 0..nbf {
            let nc = m.b_nbr_cell[bf];
            if nc < 0 {
                continue;
            }
            let pc = cell_part[m.b_face_cells[bf] as usize];
            if cell_part[nc as usize] != pc {
                n_cut_couples += 1;
                need[pc as usize].push(nc);
            }
        }
        for h in need.iter_mut() {
            h.sort_unstable_by_key(|&g| (cell_part[g as usize], g));
            h.dedup();
        }

        // ---- faces, bucketed by part in one pass over each list -----------
        let mut interior: Vec<Vec<Label>> = vec![Vec::new(); n_parts];
        let mut cut: Vec<Vec<(Label, Label)>> = vec![Vec::new(); n_parts];
        for f in 0..nif {
            let o = m.owner[f];
            let nb = m.neighbour[f];
            let (po, pn) = (cell_part[o as usize], cell_part[nb as usize]);
            if po == pn {
                interior[po as usize].push(f as Label);
            } else {
                cut[po as usize].push((pn, f as Label));
                cut[pn as usize].push((po, f as Label));
            }
        }
        // Stable, so within a neighbouring part the faces stay ascending in f.
        for c in cut.iter_mut() {
            c.sort_by_key(|&(q, _)| q);
        }

        let mut boundary: Vec<Vec<Label>> = vec![Vec::new(); n_parts];
        for bf in 0..nbf {
            let c = m.b_face_cells[bf];
            boundary[cell_part[c as usize] as usize].push(bf as Label);
        }

        let lists = PartLists {
            owned,
            need,
            interior,
            cut,
            boundary,
        };

        // ---- build each part ----------------------------------------------
        // Two scratch arrays, sized once and cleared entry by entry after each
        // part, so the peak cost is O(n + nbf) rather than O(P (n + nbf)).
        let mut halo_of = vec![-1 as Label; n];
        let mut local_bf_of = vec![-1 as Label; nbf];
        let mut parts = Vec::with_capacity(n_parts);

        for p in 0..n_parts {
            for (i, &g) in lists.need[p].iter().enumerate() {
                halo_of[g as usize] = (lists.owned[p].len() + i) as Label;
            }
            let b = PartBuild {
                m,
                cell_part: &cell_part,
                local_of: &local_of,
                halo_of: &halo_of,
            };
            let part = b.build(p, &lists, &mut local_bf_of)?;
            for &g in &lists.need[p] {
                halo_of[g as usize] = -1;
            }
            for &bf in &lists.boundary[p] {
                local_bf_of[bf as usize] = -1;
            }
            parts.push(part);
        }

        // ---- the send lists, read off the neighbours' halos ---------------
        for p in 0..n_parts {
            let mut send_index = Vec::new();
            let mut send_offset = vec![0 as Label];
            for &q in &parts[p].nbr_parts {
                for &g in &lists.need[q as usize] {
                    if cell_part[g as usize] == p as Label {
                        send_index.push(local_of[g as usize]);
                    }
                }
                send_offset.push(send_index.len() as Label);
            }
            // The plan is symmetric or it is wrong: if p hears from q then q
            // must hear from p, and the two slices must be the same length. A
            // one-sided coupling in the mesh is the only way to reach here, and
            // it would DEADLOCK a real exchange rather than give a wrong
            // answer, so it is named now, on the host, where it is cheap.
            for (i, &q) in parts[p].nbr_parts.iter().enumerate() {
                let qi = q as usize;
                let back = parts[qi]
                    .nbr_parts
                    .iter()
                    .position(|&r| r == p as Label)
                    .ok_or_else(|| {
                        Error::Mesh(format!(
                            "decompose: part {p} expects halo cells from part \
                             {qi}, but part {qi} does not list {p} as a \
                             neighbour - the mesh has a one-sided coupling"
                        ))
                    })?;
                let sent = (send_offset[i + 1] - send_offset[i]) as usize;
                let recvd =
                    (parts[qi].recv_offset[back + 1] - parts[qi].recv_offset[back]) as usize;
                if sent != recvd {
                    return Err(Error::Mesh(format!(
                        "decompose: part {p} would send {sent} cells to part \
                         {qi}, which expects {recvd}"
                    )));
                }
            }
            parts[p].send_index = send_index;
            parts[p].send_offset = send_offset;
        }

        Ok(Self {
            n_parts,
            cell_part,
            parts,
            n_global_cells: n,
            n_global_internal_faces: nif,
            n_global_boundary_faces: nbf,
            n_cut_faces,
            n_cut_couples,
        })
    }

    pub fn report(&self) -> DecompositionReport {
        let cells: Vec<usize> = self.parts.iter().map(|p| p.mesh.n_cells).collect();
        let halo: Vec<usize> = self.parts.iter().map(|p| p.n_halo).collect();
        let nbrs: Vec<usize> = self.parts.iter().map(|p| p.nbr_parts.len()).collect();
        let ideal = self.n_global_cells as Scalar / self.n_parts as Scalar;
        let imbalance = if ideal > 0.0 {
            cells.iter().copied().max().unwrap_or(0) as Scalar / ideal - 1.0
        } else {
            0.0
        };
        let halo_fraction = if self.n_global_cells > 0 {
            halo.iter().sum::<usize>() as Scalar / self.n_global_cells as Scalar
        } else {
            0.0
        };
        DecompositionReport {
            n_parts: self.n_parts,
            cells,
            halo,
            nbrs,
            n_cut_faces: self.n_cut_faces,
            n_cut_couples: self.n_cut_couples,
            imbalance,
            halo_fraction,
        }
    }
}

impl PartBuild<'_> {
    /// Assemble one part's [`HostMesh`] and the receive half of its plan.
    fn build(&self, p: usize, lists: &PartLists, local_bf_of: &mut [Label]) -> Result<PartMesh> {
        let m = self.m;
        let owned = &lists.owned[p];
        let need = &lists.need[p];
        let interior = &lists.interior[p];
        let cut = &lists.cut[p];
        let inherited = &lists.boundary[p];

        let np = owned.len();
        let nh = need.len();
        let n_if = interior.len();
        let n_bf = inherited.len() + cut.len();
        let nif_global = m.n_internal_faces;

        let mut pm = HostMesh {
            n_cells: np,
            n_internal_faces: n_if,
            n_boundary_faces: n_bf,
            n_points: 0,
            ..Default::default()
        };

        // ---- cells ---------------------------------------------------------
        pm.v = owned.iter().map(|&g| m.v[g as usize]).collect();
        pm.c = owned.iter().map(|&g| m.c[g as usize]).collect();

        // ---- interior faces: a straight copy, renumbered -------------------
        // Ascending in the global face id, and the global mesh is in
        // upper-triangular (owner, neighbour) order, so the local mesh is too:
        // `local_of` is monotonic within a part, and a subsequence of a sorted
        // sequence is sorted.
        pm.owner = interior
            .iter()
            .map(|&f| self.local_of[m.owner[f as usize] as usize])
            .collect();
        pm.neighbour = interior
            .iter()
            .map(|&f| self.local_of[m.neighbour[f as usize] as usize])
            .collect();
        pm.sf = interior.iter().map(|&f| m.sf[f as usize]).collect();
        pm.mag_sf = interior.iter().map(|&f| m.mag_sf[f as usize]).collect();
        pm.cf = interior.iter().map(|&f| m.cf[f as usize]).collect();
        pm.weights = interior.iter().map(|&f| m.weights[f as usize]).collect();
        pm.delta_coeffs = interior
            .iter()
            .map(|&f| m.delta_coeffs[f as usize])
            .collect();
        pm.non_orth_corr = interior
            .iter()
            .map(|&f| m.non_orth_corr[f as usize])
            .collect();

        // ---- boundary faces: inherited patches, then processor patches -----
        let mut b_face_cells = Vec::with_capacity(n_bf);
        let mut b_sf = Vec::with_capacity(n_bf);
        let mut b_mag_sf = Vec::with_capacity(n_bf);
        let mut b_cf = Vec::with_capacity(n_bf);
        let mut b_delta_coeffs = Vec::with_capacity(n_bf);
        let mut b_non_orth_corr = Vec::with_capacity(n_bf);
        let mut b_y = Vec::with_capacity(n_bf);
        let mut b_nbr_cell = Vec::with_capacity(n_bf);
        let mut b_nbr_face = Vec::with_capacity(n_bf);
        let mut b_weights = Vec::with_capacity(n_bf);
        let mut b_kind = Vec::with_capacity(n_bf);
        let mut b_patch = Vec::with_capacity(n_bf);
        let mut global_face: Vec<Label> = Vec::with_capacity(n_if + n_bf);
        global_face.extend(interior.iter().copied());
        let mut patches: Vec<PatchInfo> = Vec::new();

        // Inherited faces, grouped by their original patch. EVERY original
        // patch is kept, even where this part owns none of it: patch indices
        // then mean the same thing on every part, which is what a
        // patch-averaging collective will need (§71.7), and an empty patch
        // costs nothing.
        let mut by_patch: Vec<Vec<Label>> = vec![Vec::new(); m.patches.len()];
        let mut orphan: Vec<Label> = Vec::new();
        for &bf in inherited {
            let k = if m.b_patch.len() == m.n_boundary_faces {
                m.b_patch[bf as usize]
            } else {
                -1
            };
            if k >= 0 && (k as usize) < by_patch.len() {
                by_patch[k as usize].push(bf);
            } else {
                orphan.push(bf);
            }
        }
        if !orphan.is_empty() {
            return Err(Error::Mesh(format!(
                "decompose: {} boundary face(s) of part {p} belong to no \
                 patch; b_patch must name a patch for every boundary face \
                 before a mesh can be cut",
                orphan.len()
            )));
        }

        // The local index a face will get, needed before the fill so that
        // `b_nbr_face` can name a partner that is also on this part.
        let mut next = 0usize;
        for faces in by_patch.iter() {
            for &bf in faces {
                local_bf_of[bf as usize] = next as Label;
                next += 1;
            }
        }

        for (k, faces) in by_patch.iter().enumerate() {
            let pi = &m.patches[k];
            patches.push(PatchInfo {
                name: pi.name.clone(),
                type_name: pi.type_name.clone(),
                kind: pi.kind,
                start: b_face_cells.len(),
                size: faces.len(),
                nbr_patch: pi.nbr_patch,
            });
            for &bfl in faces {
                let bf = bfl as usize;
                b_face_cells.push(self.local_of[m.b_face_cells[bf] as usize]);
                b_sf.push(pick(&m.b_sf, bf, Vec3::ZERO));
                b_mag_sf.push(pick(&m.b_mag_sf, bf, 0.0));
                b_cf.push(pick(&m.b_cf, bf, Vec3::ZERO));
                b_delta_coeffs.push(pick(&m.b_delta_coeffs, bf, 0.0));
                b_non_orth_corr.push(pick(&m.b_non_orth_corr, bf, Vec3::ZERO));
                b_y.push(pick(&m.b_y, bf, 0.0));
                b_weights.push(pick(&m.b_weights, bf, 1.0));
                b_kind.push(pick(&m.b_kind, bf, pi.kind as Label));
                b_patch.push(k as Label);

                // The coupled neighbour: still owned here, or now in the halo.
                let nc = m.b_nbr_cell[bf];
                b_nbr_cell.push(if nc < 0 {
                    -1
                } else if self.cell_part[nc as usize] == p as Label {
                    self.local_of[nc as usize]
                } else {
                    self.halo_of[nc as usize]
                });

                // §48.3's face pairing survives only where BOTH halves are on
                // this part. Across a cut the partner face is a different
                // part's array and there is no local index for it, so the
                // pairing is dropped rather than faked - every consumer
                // already reads -1 as "no couple to compare against".
                let nf = if m.b_nbr_face.len() == m.n_boundary_faces {
                    m.b_nbr_face[bf]
                } else {
                    -1
                };
                b_nbr_face.push(if nf >= 0 { local_bf_of[nf as usize] } else { -1 });

                global_face.push((nif_global + bf) as Label);
            }
        }

        // Processor patches: one per neighbouring part that a CUT INTERNAL
        // FACE reaches. A cut cyclic couple gets no patch here - it stays on
        // its own patch above, with its own metrics, and only its
        // `b_nbr_cell` moved into the halo.
        let mut proc_patches = Vec::new();
        let mut i = 0usize;
        while i < cut.len() {
            let q = cut[i].0;
            let mut j = i;
            let start = b_face_cells.len();
            while j < cut.len() && cut[j].0 == q {
                let f = cut[j].1 as usize;
                let o = m.owner[f] as usize;
                let nb = m.neighbour[f] as usize;
                let owns = self.cell_part[o] == p as Label;
                let (here, there) = if owns { (o, nb) } else { (nb, o) };

                // §71.3: every metric below is a COPY, a NEGATION or the same
                // function of the same operands as the whole mesh used. None
                // is recomputed from points, and negation, `|.|` and a
                // reversed pair of projections are all exact in IEEE 754.
                let sf = if owns { m.sf[f] } else { -m.sf[f] };
                let cf = m.cf[f];
                let d_own = cf - m.c[o];
                let d_nbr = m.c[nb] - cf;
                let a = m.sf[f].dot(d_own).abs();
                let b = m.sf[f].dot(d_nbr).abs();

                b_face_cells.push(self.local_of[here]);
                b_sf.push(sf);
                b_mag_sf.push(m.mag_sf[f]);
                b_cf.push(cf);
                b_delta_coeffs.push(m.delta_coeffs[f]);
                b_non_orth_corr.push(if owns {
                    m.non_orth_corr[f]
                } else {
                    -m.non_orth_corr[f]
                });
                let d_here = cf - m.c[here];
                b_y.push(crate::mesh::geometry::floor_along(
                    sf.normalised().dot(d_here),
                    d_here,
                ));
                b_weights.push(if owns {
                    m.weights[f]
                } else {
                    crate::mesh::geometry::weight_from_offsets(b, a)
                });
                b_nbr_cell.push(self.halo_of[there]);
                b_nbr_face.push(-1);
                // Cyclic, NOT Processor. `PatchKind::Processor` is a host-side
                // label that no kernel honours: every coupled branch in
                // `cuda/*.cu` tests `bKind == OFPATCH_CYCLIC`, so a face
                // marked Processor would silently take the UNCOUPLED path and
                // integrate the wrong flux. §71.5.
                b_kind.push(PatchKind::Cyclic as Label);
                b_patch.push(patches.len() as Label);
                global_face.push(cut[j].1);
                j += 1;
            }
            proc_patches.push((patches.len(), q));
            patches.push(PatchInfo {
                name: format!("procBoundary{p}to{q}"),
                type_name: "processor".to_string(),
                kind: PatchKind::Cyclic,
                start,
                size: j - i,
                nbr_patch: None,
            });
            i = j;
        }

        pm.b_face_cells = b_face_cells;
        pm.b_sf = b_sf;
        pm.b_mag_sf = b_mag_sf;
        pm.b_cf = b_cf;
        pm.b_delta_coeffs = b_delta_coeffs;
        pm.b_non_orth_corr = b_non_orth_corr;
        pm.b_y = b_y;
        pm.b_nbr_cell = b_nbr_cell;
        pm.b_nbr_face = b_nbr_face;
        pm.b_weights = b_weights;
        pm.b_kind = b_kind;
        pm.b_patch = b_patch;
        pm.patches = patches;
        pm.global_face = global_face;

        // The merged row map of §70 is built LAST, from `global_face`, and it
        // is the whole reason this module exists: it puts each row's terms
        // back in whole-mesh face order, so the cut face that just moved from
        // `upper` to `boundary_coeffs` keeps its place in the sum.
        pm.build_cell_face_maps();

        // ---- the receive half of the exchange plan ------------------------
        let mut nbr_parts: Vec<Label> = Vec::new();
        let mut recv_offset = vec![0 as Label];
        for (i, &g) in need.iter().enumerate() {
            let q = self.cell_part[g as usize];
            if nbr_parts.last() != Some(&q) {
                nbr_parts.push(q);
                if i > 0 {
                    recv_offset.push(i as Label);
                }
            }
        }
        // A part with no halo has no neighbours and no slices: `recv_offset`
        // is then the single entry `[0]`, not `[0, 0]`, so that its length is
        // `nbr_parts.len() + 1` in that case too.
        if !nbr_parts.is_empty() {
            recv_offset.push(nh as Label);
        }

        let mut global_cell = Vec::with_capacity(np + nh);
        global_cell.extend(owned.iter().copied());
        global_cell.extend(need.iter().copied());

        Ok(PartMesh {
            part: p,
            mesh: pm,
            n_halo: nh,
            global_cell,
            nbr_parts,
            send_offset: Vec::new(),
            send_index: Vec::new(),
            recv_offset,
            proc_patches,
        })
    }
}

/// Read `v[i]`, or `dflt` when the array is short.
///
/// A `HostMesh` written by hand in a test may carry none of the optional
/// boundary metrics, and a decomposition of it should still be a mesh rather
/// than a panic. The defaults are the ones `geometry::compute` itself uses for
/// an uncoupled face.
#[inline]
fn pick<T: Copy>(v: &[T], i: usize, dflt: T) -> T {
    v.get(i).copied().unwrap_or(dflt)
}

/// The mesh arrays this module indexes, checked once so that a short array is
/// a named error rather than a panic three hundred lines later.
fn check_lengths(m: &HostMesh) -> Result<()> {
    let nif = m.n_internal_faces;
    let nbf = m.n_boundary_faces;
    let n = m.n_cells;
    let want: [(&str, usize, usize); 12] = [
        ("owner", m.owner.len(), nif),
        ("neighbour", m.neighbour.len(), nif),
        ("sf", m.sf.len(), nif),
        ("mag_sf", m.mag_sf.len(), nif),
        ("cf", m.cf.len(), nif),
        ("weights", m.weights.len(), nif),
        ("delta_coeffs", m.delta_coeffs.len(), nif),
        ("non_orth_corr", m.non_orth_corr.len(), nif),
        ("v", m.v.len(), n),
        ("c", m.c.len(), n),
        ("b_face_cells", m.b_face_cells.len(), nbf),
        ("b_nbr_cell", m.b_nbr_cell.len(), nbf),
    ];
    for (name, have, expect) in want {
        if have != expect {
            return Err(Error::Mesh(format!(
                "decompose: mesh array '{name}' has {have} entries, expected \
                 {expect}; the mesh must have been through compute_geometry \
                 before it can be cut"
            )));
        }
    }
    for f in 0..nif {
        for (which, l) in [("owner", m.owner[f]), ("neighbour", m.neighbour[f])] {
            if l < 0 || l as usize >= n {
                return Err(Error::Mesh(format!(
                    "decompose: internal face {f}'s {which} is {l}, outside \
                     [0, {n})"
                )));
            }
        }
    }
    for bf in 0..nbf {
        let c = m.b_face_cells[bf];
        if c < 0 || c as usize >= n {
            return Err(Error::Mesh(format!(
                "decompose: boundary face {bf} names cell {c}, outside [0, {n})"
            )));
        }
        let nc = m.b_nbr_cell[bf];
        if nc >= 0 && nc as usize >= n {
            return Err(Error::Mesh(format!(
                "decompose: boundary face {bf} names neighbour cell {nc}, \
                 outside [0, {n})"
            )));
        }
    }
    Ok(())
}

// ==========================================================================
//  Splitting a field and a matrix, and putting the answer back together
// ==========================================================================

impl Decomposition {
    /// Distribute a whole-mesh scalar field to part `p`.
    ///
    /// Returns `[n_cells + n_halo]` with the owned half filled and **the halo
    /// left at zero**. That is deliberate: the halo is the exchange's job, and
    /// pre-filling it here would let a run that forgot to exchange produce the
    /// right answer for the wrong reason, which is the failure this whole
    /// section exists to make impossible.
    pub fn split_field(&self, p: usize, global: &[Scalar]) -> Result<Vec<Scalar>> {
        let part = self.part(p)?;
        if global.len() != self.n_global_cells {
            return Err(Error::Config(format!(
                "decompose: the field has {} values, the mesh has {} cells",
                global.len(),
                self.n_global_cells
            )));
        }
        let mut out = vec![0.0 as Scalar; part.n_local()];
        for (i, o) in out.iter_mut().enumerate().take(part.mesh.n_cells) {
            *o = global[part.global_cell[i] as usize];
        }
        Ok(out)
    }

    /// The same for a label field - `is_fixed` and anything else a coupled
    /// face's neighbour is tested for.
    pub fn split_labels(&self, p: usize, global: &[Label]) -> Result<Vec<Label>> {
        let part = self.part(p)?;
        if global.len() != self.n_global_cells {
            return Err(Error::Config(format!(
                "decompose: the label field has {} values, the mesh has {} \
                 cells",
                global.len(),
                self.n_global_cells
            )));
        }
        let mut out = vec![0 as Label; part.n_local()];
        for (i, o) in out.iter_mut().enumerate().take(part.mesh.n_cells) {
            *o = global[part.global_cell[i] as usize];
        }
        Ok(out)
    }

    /// Put the parts' owned values back in whole-mesh order.
    ///
    /// Halo values are ignored: every cell has exactly one owner and only the
    /// owner's copy is read, so the result is a function of the owned data
    /// alone and cannot depend on whether the halo was up to date.
    pub fn gather_field(&self, per_part: &[Vec<Scalar>]) -> Result<Vec<Scalar>> {
        if per_part.len() != self.n_parts {
            return Err(Error::Config(format!(
                "decompose: {} field buffers for {} parts",
                per_part.len(),
                self.n_parts
            )));
        }
        let mut out = vec![0.0 as Scalar; self.n_global_cells];
        for (p, part) in self.parts.iter().enumerate() {
            if per_part[p].len() < part.mesh.n_cells {
                return Err(Error::Config(format!(
                    "decompose: part {p}'s buffer holds {} values but the part \
                     owns {} cells",
                    per_part[p].len(),
                    part.mesh.n_cells
                )));
            }
            for i in 0..part.mesh.n_cells {
                out[part.global_cell[i] as usize] = per_part[p][i];
            }
        }
        Ok(out)
    }

    /// Distribute an assembled LDU matrix to part `p`.
    ///
    /// **Every coefficient is a copy or an exact negation of a whole-mesh
    /// coefficient. Nothing is re-assembled.** SPEC-LIT §71.6 is the argument
    /// for why that is the only way to get a decomposed matrix that is the
    /// serial matrix to the bit, and the rule in three lines is:
    ///
    /// ```text
    /// interior face  i   ->  upper[i], lower[i] copied from global face g
    /// inherited face j   ->  internal_coeffs[j], boundary_coeffs[j] copied
    /// CUT face       j   ->  boundary_coeffs[j] = -upper[g]   this part owns g
    ///                        boundary_coeffs[j] = -lower[g]   it does not
    ///                        internal_coeffs[j] = 0
    /// ```
    ///
    /// The sign is not a convention pulled out of the air. `lduAmul` applies a
    /// coupled boundary face as `sum -= boundary_coeffs * psi_N` and an owned
    /// internal face as `sum += upper * psi_N`; IEEE negation is exact and
    /// `a - (-x)*y` and `a + x*y` are the same operation on the same bits, so
    /// the cut face contributes the identical term it contributed before the
    /// cut - in the identical place in the row, because §70's merged row map
    /// keys the order on the global face id.
    ///
    /// `internal_coeffs` is zero on a cut face because the whole mesh already
    /// put that face's diagonal share in `diag`; adding it again would double
    /// it. Zero also makes the fold
    /// (`ldu_ops::add_boundary_contributions`) a no-op there, so a part may be
    /// folded before or after it is split with the same result.
    pub fn split_matrix(&self, m: &HostMesh, p: usize, a: &HostLduMatrix) -> Result<HostLduMatrix> {
        let part = self.part(p)?;
        if a.n_cells != self.n_global_cells
            || a.n_internal_faces != self.n_global_internal_faces
            || a.n_boundary_faces != self.n_global_boundary_faces
        {
            return Err(Error::Config(format!(
                "decompose: the matrix is {}/{}/{} cells/faces/boundary faces, \
                 the mesh {}/{}/{}",
                a.n_cells,
                a.n_internal_faces,
                a.n_boundary_faces,
                self.n_global_cells,
                self.n_global_internal_faces,
                self.n_global_boundary_faces
            )));
        }
        if m.n_internal_faces != self.n_global_internal_faces {
            return Err(Error::Config(
                "decompose: split_matrix was given a different mesh from the \
                 one that was cut"
                    .to_string(),
            ));
        }

        let pm = &part.mesh;
        let n_if = pm.n_internal_faces;
        let n_bf = pm.n_boundary_faces;
        let nif_global = self.n_global_internal_faces;

        let mut out = HostLduMatrix {
            n_cells: pm.n_cells,
            n_internal_faces: n_if,
            n_boundary_faces: n_bf,
            diag: Vec::with_capacity(pm.n_cells),
            upper: Vec::with_capacity(n_if),
            lower: Vec::with_capacity(n_if),
            source: Vec::with_capacity(pm.n_cells),
            internal_coeffs: Vec::with_capacity(n_bf),
            boundary_coeffs: Vec::with_capacity(n_bf),
            // Halo-extended: `lduSetValues` reads these at a coupled face's
            // neighbour, which on a part is a ghost cell. Filled by the
            // exchange, not here (see `split_field`).
            is_fixed: vec![0; part.n_local()],
            fixed_value: vec![0.0; part.n_local()],
        };

        for i in 0..pm.n_cells {
            let g = part.global_cell[i] as usize;
            out.diag.push(a.diag[g]);
            out.source.push(a.source[g]);
            out.is_fixed[i] = a.is_fixed.get(g).copied().unwrap_or(0);
            out.fixed_value[i] = a.fixed_value.get(g).copied().unwrap_or(0.0);
        }

        for i in 0..n_if {
            let g = pm.global_face[i] as usize;
            out.upper.push(a.upper[g]);
            out.lower.push(a.lower[g]);
        }

        for j in 0..n_bf {
            let g = pm.global_face[n_if + j] as usize;
            if g >= nif_global {
                let bf = g - nif_global;
                out.internal_coeffs.push(a.internal_coeffs[bf]);
                out.boundary_coeffs.push(a.boundary_coeffs[bf]);
            } else {
                let owns = self.cell_part[m.owner[g] as usize] == p as Label;
                out.internal_coeffs.push(0.0);
                out.boundary_coeffs
                    .push(if owns { -a.upper[g] } else { -a.lower[g] });
            }
        }

        Ok(out)
    }

    fn part(&self, p: usize) -> Result<&PartMesh> {
        self.parts.get(p).ok_or_else(|| {
            Error::Config(format!(
                "decompose: part {p} asked for, the decomposition has {}",
                self.n_parts
            ))
        })
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::mesh::topology::tests::box_mesh;

    /// A box of hexahedra with its geometry computed and its row map built -
    /// an ordinary mesh, indistinguishable from one read off disk.
    ///
    /// With `cyclic`, the `xmin`/`xmax` patches become a couple, which is the
    /// case that matters most here: a cyclic patch is the ONLY place an
    /// undecomposed mesh has `b_nbr_cell >= 0`, so it is the only mesh on
    /// which the cut can land on a face that was already coupled.
    pub(crate) fn boxes(n: [usize; 3], cyclic: bool) -> HostMesh {
        let d = Vec3::new(0.5, 0.25, 2.0);
        let (mut m, points, faces) = box_mesh(n, d);
        if cyclic {
            for (p, nbr) in [(0usize, 1usize), (1, 0)] {
                m.patches[p].kind = PatchKind::Cyclic;
                m.patches[p].type_name = "cyclic".to_string();
                m.patches[p].nbr_patch = Some(nbr);
            }
        }
        m.compute_geometry(&points, &faces).expect("geometry");
        m.build_cell_face_maps();
        m
    }

    /// Round robin: cell `c` to part `c % P`. The worst partition there is -
    /// almost every face is cut - which is exactly why the tests use it. A
    /// good partition hides ordering bugs; this one cannot.
    pub(crate) fn round_robin(n: usize, p: usize) -> PartitionMethod {
        PartitionMethod::Explicit((0..n).map(|c| (c % p) as Label).collect())
    }

    // ----------------------------------------------------------------------
    //  The curve
    // ----------------------------------------------------------------------

    /// Both properties the Hilbert sort relies on: the index is a bijection on
    /// the lattice, and consecutive indices are geometric neighbours.
    ///
    /// The second is the one that matters - it is why cutting the sorted list
    /// into runs gives compact parts - and it is the one an off-by-one in the
    /// Gray-code fold silently destroys while leaving the bijection intact.
    #[test]
    fn the_hilbert_curve_visits_every_point_and_never_jumps() {
        for bits in 1..=4u32 {
            let side = 1u32 << bits;
            let n = (side as usize).pow(3);
            let mut point_of = vec![None; n];
            for z in 0..side {
                for y in 0..side {
                    for x in 0..side {
                        let h = hilbert_index([x, y, z], bits) as usize;
                        assert!(h < n, "index {h} out of range at bits={bits}");
                        assert!(
                            point_of[h].is_none(),
                            "bits={bits}: index {h} is claimed by two points"
                        );
                        point_of[h] = Some([x, y, z]);
                    }
                }
            }
            for h in 1..n {
                let a = point_of[h - 1].expect("bijection");
                let b = point_of[h].expect("bijection");
                let step: u32 = (0..3)
                    .map(|i| a[i].abs_diff(b[i]))
                    .sum();
                assert_eq!(
                    step, 1,
                    "bits={bits}: steps {h}-1 -> {h} from {a:?} to {b:?}, which \
                     is not an edge of the lattice"
                );
            }
        }
    }

    // ----------------------------------------------------------------------
    //  The partition
    // ----------------------------------------------------------------------

    #[test]
    fn every_partition_method_is_a_function_of_the_mesh_alone() {
        let m = boxes([4, 3, 2], false);
        for method in [PartitionMethod::Linear, PartitionMethod::Hilbert] {
            for p in 1..=4 {
                let a = partition(&m, p, &method).expect("partition");
                let b = partition(&m, p, &method).expect("partition");
                assert_eq!(a, b, "{} at P={p} is not deterministic", method.as_str());
                assert_eq!(a.len(), m.n_cells);
                let mut count = vec![0usize; p];
                for &q in &a {
                    count[q as usize] += 1;
                }
                assert!(
                    count.iter().all(|&c| c > 0),
                    "{} at P={p} left a part empty: {count:?}",
                    method.as_str()
                );
                let (lo, hi) = (
                    *count.iter().min().unwrap(),
                    *count.iter().max().unwrap(),
                );
                assert!(
                    hi - lo <= 1,
                    "{} at P={p} is unbalanced: {count:?}",
                    method.as_str()
                );
            }
        }
    }

    /// The refusals §13.4 owes: a request the decomposition cannot honour
    /// fails by name, with the number that was wrong in the message.
    #[test]
    fn an_impossible_partition_is_refused_by_name() {
        let m = boxes([2, 2, 2], false);
        let cases: [(usize, PartitionMethod, &str); 4] = [
            (0, PartitionMethod::Hilbert, "0 parts"),
            (9, PartitionMethod::Hilbert, "only 8"),
            (
                2,
                PartitionMethod::Explicit(vec![0, 1, 2, 0, 1, 0, 1, 0]),
                "outside [0, 2)",
            ),
            (
                3,
                PartitionMethod::Explicit(vec![0, 1, 0, 1, 0, 1, 0, 1]),
                "leaves part 2 empty",
            ),
        ];
        for (p, method, wanted) in cases {
            let e = partition(&m, p, &method).expect_err("must refuse");
            let msg = e.to_string();
            assert!(
                msg.contains(wanted),
                "the refusal for P={p} says {msg:?}, which does not mention \
                 {wanted:?}"
            );
        }
        let short = PartitionMethod::Explicit(vec![0, 1]);
        let msg = partition(&m, 2, &short).expect_err("must refuse").to_string();
        assert!(msg.contains("2 entries"), "{msg}");
    }

    // ----------------------------------------------------------------------
    //  The decomposition
    // ----------------------------------------------------------------------

    #[test]
    fn a_decomposition_accounts_for_every_cell_and_every_face() {
        for cyclic in [false, true] {
            let m = boxes([4, 3, 2], cyclic);
            for p in 1..=4 {
                for method in [
                    PartitionMethod::Linear,
                    PartitionMethod::Hilbert,
                    round_robin(m.n_cells, p),
                ] {
                    let d = Decomposition::build(&m, p, &method).expect("decompose");
                    let cells: usize = d.parts.iter().map(|x| x.mesh.n_cells).sum();
                    assert_eq!(cells, m.n_cells);

                    let inner: usize = d.parts.iter().map(|x| x.mesh.n_internal_faces).sum();
                    assert_eq!(
                        inner + d.n_cut_faces,
                        m.n_internal_faces,
                        "P={p} {}: {inner} interior + {} cut != {} internal faces",
                        method.as_str(),
                        d.n_cut_faces,
                        m.n_internal_faces
                    );

                    let bnd: usize = d.parts.iter().map(|x| x.mesh.n_boundary_faces).sum();
                    assert_eq!(
                        bnd,
                        m.n_boundary_faces + 2 * d.n_cut_faces,
                        "P={p} {}: a cut face must become a boundary face on \
                         BOTH sides",
                        method.as_str()
                    );

                    // Every local cell index reachable from a face is in range,
                    // owned or halo.
                    for part in &d.parts {
                        let n_local = part.n_local() as Label;
                        for f in 0..part.mesh.n_internal_faces {
                            for l in [part.mesh.owner[f], part.mesh.neighbour[f]] {
                                assert!(l >= 0 && l < part.mesh.n_cells as Label);
                            }
                            assert!(part.mesh.owner[f] < part.mesh.neighbour[f]);
                        }
                        for bf in 0..part.mesh.n_boundary_faces {
                            let c = part.mesh.b_face_cells[bf];
                            assert!(c >= 0 && c < part.mesh.n_cells as Label);
                            let nc = part.mesh.b_nbr_cell[bf];
                            assert!(nc >= -1 && nc < n_local);
                        }
                    }
                }
            }
        }
    }

    /// A cut in one piece is not a cut. Every array of the single part must be
    /// the whole mesh's array, and `global_face` must come out the identity -
    /// which is what makes `P = 1` go through exactly the same code as `P = 4`
    /// rather than down a special case that is never exercised.
    #[test]
    fn a_one_part_decomposition_is_the_mesh_itself() {
        for cyclic in [false, true] {
            let m = boxes([3, 2, 2], cyclic);
            let d = Decomposition::build(&m, 1, &PartitionMethod::Hilbert).expect("decompose");
            let p = &d.parts[0];
            assert_eq!(p.n_halo, 0);
            assert!(p.nbr_parts.is_empty());
            assert_eq!(p.recv_offset, vec![0]);
            assert_eq!(p.send_offset, vec![0]);
            assert_eq!(p.mesh.n_cells, m.n_cells);
            assert_eq!(p.mesh.n_internal_faces, m.n_internal_faces);
            assert_eq!(p.mesh.n_boundary_faces, m.n_boundary_faces);
            assert_eq!(p.mesh.owner, m.owner);
            assert_eq!(p.mesh.neighbour, m.neighbour);
            assert_eq!(p.mesh.v, m.v);
            assert_eq!(p.mesh.c, m.c);
            assert_eq!(p.mesh.sf, m.sf);
            assert_eq!(p.mesh.weights, m.weights);
            assert_eq!(p.mesh.delta_coeffs, m.delta_coeffs);
            assert_eq!(p.mesh.non_orth_corr, m.non_orth_corr);
            assert_eq!(p.mesh.b_face_cells, m.b_face_cells);
            assert_eq!(p.mesh.b_nbr_cell, m.b_nbr_cell);
            assert_eq!(p.mesh.b_nbr_face, m.b_nbr_face);
            assert_eq!(p.mesh.b_weights, m.b_weights);
            assert_eq!(p.mesh.b_delta_coeffs, m.b_delta_coeffs);
            assert_eq!(p.mesh.b_y, m.b_y);
            assert_eq!(p.mesh.b_kind, m.b_kind);
            assert_eq!(p.mesh.rf_offset, m.rf_offset);
            assert_eq!(p.mesh.rf_face, m.rf_face);
            assert_eq!(p.mesh.rf_flags, m.rf_flags);
            let identity: Vec<Label> =
                (0..(m.n_internal_faces + m.n_boundary_faces) as Label).collect();
            assert_eq!(p.mesh.global_face, identity);
        }
    }

    /// The plan has to be symmetric with no handshake: what part `p` packs for
    /// `q` must be, cell for cell and in the same order, what `q` expects from
    /// `p`. Checked in GLOBAL ids, because the two parts number those cells
    /// differently and agreeing locally would prove nothing.
    #[test]
    fn the_exchange_plan_is_symmetric_in_global_cell_ids() {
        let m = boxes([4, 3, 2], true);
        for p in 2..=4 {
            for method in [PartitionMethod::Hilbert, round_robin(m.n_cells, p)] {
                let d = Decomposition::build(&m, p, &method).expect("decompose");
                for a in &d.parts {
                    for (i, &qq) in a.nbr_parts.iter().enumerate() {
                        let b = &d.parts[qq as usize];
                        let j = b
                            .nbr_parts
                            .iter()
                            .position(|&r| r as usize == a.part)
                            .expect("symmetric neighbour list");
                        let sent: Vec<Label> = (a.send_offset[i]..a.send_offset[i + 1])
                            .map(|k| a.global_cell[a.send_index[k as usize] as usize])
                            .collect();
                        let expected: Vec<Label> = (b.recv_offset[j]..b.recv_offset[j + 1])
                            .map(|k| b.global_cell[b.mesh.n_cells + k as usize])
                            .collect();
                        assert_eq!(
                            sent,
                            expected,
                            "part {} -> part {} disagrees ({}, P={p})",
                            a.part,
                            qq,
                            method.as_str()
                        );
                        assert!(sent.windows(2).all(|w| w[0] < w[1]), "not ascending");
                    }
                }
            }
        }
    }

    /// The halo is ordered by (owning part, global cell id) and by nothing
    /// else. That ordering is what makes each neighbour's contribution one
    /// contiguous run, which is what removes the unpack kernel - the only
    /// scatter the exchange could otherwise have had.
    #[test]
    fn the_halo_is_grouped_by_owning_part_and_ascending_within_a_group() {
        let m = boxes([4, 3, 2], true);
        for p in 2..=4 {
            let d = Decomposition::build(&m, p, &round_robin(m.n_cells, p)).expect("decompose");
            for part in &d.parts {
                let n = part.mesh.n_cells;
                assert_eq!(part.recv_offset.len(), part.nbr_parts.len() + 1);
                assert_eq!(*part.recv_offset.last().unwrap() as usize, part.n_halo);
                for (i, &q) in part.nbr_parts.iter().enumerate() {
                    let (a, b) = (part.recv_offset[i] as usize, part.recv_offset[i + 1] as usize);
                    assert!(a < b, "empty halo group");
                    for k in a..b {
                        let g = part.global_cell[n + k];
                        assert_eq!(d.cell_part[g as usize], q);
                        if k > a {
                            assert!(g > part.global_cell[n + k - 1]);
                        }
                    }
                }
                assert!(part.nbr_parts.windows(2).all(|w| w[0] < w[1]));
            }
        }
    }

    /// SPEC-LIT §71.3. Every metric of a cut face is the whole mesh's metric,
    /// copied or exactly negated - not recomputed. A recomputation would agree
    /// to round-off and differ in the last bit, and one bit in `Sf` is one bit
    /// in `upper[f]`.
    #[test]
    fn a_cut_face_carries_the_whole_meshs_metrics_exactly() {
        let m = boxes([4, 3, 2], false);
        let nif = m.n_internal_faces;
        let d = Decomposition::build(&m, 3, &round_robin(m.n_cells, 3)).expect("decompose");
        let mut seen = 0usize;
        for part in &d.parts {
            let pm = &part.mesh;
            for bf in 0..pm.n_boundary_faces {
                let g = pm.global_face[pm.n_internal_faces + bf] as usize;
                if g >= nif {
                    continue; // an inherited boundary face
                }
                seen += 1;
                let f = g;
                let here = part.global_cell[pm.b_face_cells[bf] as usize];
                let owns = here == m.owner[f];
                assert_eq!(pm.b_mag_sf[bf], m.mag_sf[f], "|Sf| moved");
                assert_eq!(pm.b_cf[bf], m.cf[f], "Cf moved");
                assert_eq!(pm.b_delta_coeffs[bf], m.delta_coeffs[f], "Delta moved");
                if owns {
                    assert_eq!(pm.b_sf[bf], m.sf[f]);
                    assert_eq!(pm.b_non_orth_corr[bf], m.non_orth_corr[f]);
                    assert_eq!(pm.b_weights[bf], m.weights[f]);
                } else {
                    assert_eq!(pm.b_sf[bf], -m.sf[f]);
                    assert_eq!(pm.b_non_orth_corr[bf], -m.non_orth_corr[f]);
                    // The neighbour's own share, from the SAME two
                    // projections with their roles swapped - NOT `1 - w`.
                    // `1 - w` is inexact for w below 1/2 (Sterbenz's lemma
                    // covers only [1/2, 2]), so it would put an ulp into every
                    // interpolation across a cut. The two projections and
                    // their sum are identical either way, so this is exact.
                    let a_proj = m.sf[f].dot(m.cf[f] - m.c[m.owner[f] as usize]).abs();
                    let b_proj = m.sf[f]
                        .dot(m.c[m.neighbour[f] as usize] - m.cf[f])
                        .abs();
                    assert_eq!(
                        pm.b_weights[bf],
                        crate::mesh::geometry::weight_from_offsets(b_proj, a_proj),
                        "face {f}: the neighbour side did not swap the roles"
                    );
                }
                assert_eq!(
                    pm.b_kind[bf],
                    PatchKind::Cyclic as Label,
                    "a processor face must be marked Cyclic on the DEVICE: \
                     every coupled branch in cuda/*.cu tests \
                     bKind == OFPATCH_CYCLIC, so PatchKind::Processor would \
                     take the uncoupled path and integrate the wrong flux"
                );
                let patch = &pm.patches[pm.b_patch[bf] as usize];
                assert_eq!(patch.type_name, "processor");
                assert!(pm.b_nbr_cell[bf] >= pm.n_cells as Label, "not in the halo");
            }
        }
        assert_eq!(seen, 2 * d.n_cut_faces, "every cut face has two halves");
        assert!(seen > 0, "the round-robin cut nothing, so nothing was tested");
    }

    /// A cyclic couple split across the cut keeps its own patch, its own
    /// metrics and its own `PatchKind` - only `b_nbr_cell` moves into the
    /// halo. That is `processorCyclic` with no new code, and it is only true
    /// because nothing is recomputed.
    #[test]
    fn a_cut_cyclic_couple_keeps_its_patch_and_its_geometry() {
        let m = boxes([4, 3, 2], true);
        let nif = m.n_internal_faces;
        // The round-robin cut splits a cyclic couple only when the two cells'
        // ids differ by something the part count does not divide - with
        // nx = 4 and P = 3 they do not - so every P is tried and the count at
        // the end is what asserts the case was reached at all.
        let mut checked = 0usize;
        let mut splits = 0usize;
        for np in 2..=4 {
            let d =
                Decomposition::build(&m, np, &round_robin(m.n_cells, np)).expect("decompose");
            splits += d.n_cut_couples;
            for part in &d.parts {
                let pm = &part.mesh;
                for bf in 0..pm.n_boundary_faces {
                    let g = pm.global_face[pm.n_internal_faces + bf] as usize;
                    if g < nif {
                        continue;
                    }
                    let src = g - nif;
                    if m.b_nbr_cell[src] < 0 {
                        continue;
                    }
                    checked += 1;
                    assert_eq!(pm.b_sf[bf], m.b_sf[src]);
                    assert_eq!(pm.b_delta_coeffs[bf], m.b_delta_coeffs[src]);
                    assert_eq!(pm.b_weights[bf], m.b_weights[src]);
                    assert_eq!(pm.b_non_orth_corr[bf], m.b_non_orth_corr[src]);
                    assert_eq!(pm.b_kind[bf], m.b_kind[src]);
                    assert_eq!(pm.patches[pm.b_patch[bf] as usize].type_name, "cyclic");
                    let nbr = pm.b_nbr_cell[bf];
                    assert!(nbr >= 0);
                    assert_eq!(
                        part.global_cell[nbr as usize], m.b_nbr_cell[src],
                        "the couple points at the wrong cell"
                    );
                }
            }
        }
        assert!(splits > 0, "no part count split the cyclic patch");
        assert!(checked > 0);
    }

    /// §70's merged row map is the reason this module exists, and on a part it
    /// finally has something to do: `global_face` is not the identity, so the
    /// builder takes its sorting path and a cut face lands back where the
    /// whole mesh had it.
    #[test]
    fn every_row_of_a_part_is_ascending_in_global_face_id() {
        let m = boxes([4, 3, 2], true);
        for p in 2..=4 {
            let d = Decomposition::build(&m, p, &round_robin(m.n_cells, p)).expect("decompose");
            let mut non_identity = 0usize;
            for part in &d.parts {
                let pm = &part.mesh;
                let n_if = pm.n_internal_faces;
                if pm
                    .global_face
                    .windows(2)
                    .any(|w| w[0] > w[1])
                {
                    non_identity += 1;
                }
                for c in 0..pm.n_cells {
                    let (a, b) = (pm.rf_offset[c] as usize, pm.rf_offset[c + 1] as usize);
                    let mut last = -1 as Label;
                    for j in a..b {
                        let f = pm.rf_face[j];
                        let slot = if pm.rf_flags[j] & crate::mesh::topology::RF_BOUNDARY != 0 {
                            n_if + f as usize
                        } else {
                            f as usize
                        };
                        let key = pm.global_face[slot];
                        assert!(
                            key > last,
                            "part {} cell {c}: row is not ascending in the \
                             global face id ({key} after {last})",
                            part.part
                        );
                        last = key;
                    }
                }
            }
            assert!(
                non_identity > 0,
                "P={p}: no part's global_face was permuted, so the sorting \
                 path was never entered"
            );
        }
    }

    /// Splitting a field and gathering it back is the identity on the owned
    /// values, and leaves the halo at zero so that a forgotten exchange
    /// cannot pass by luck.
    #[test]
    fn splitting_a_field_and_gathering_it_back_is_the_identity() {
        let m = boxes([4, 3, 2], true);
        let global: Vec<Scalar> = (0..m.n_cells).map(|c| 1.0 + c as Scalar * 0.25).collect();
        for p in 1..=4 {
            let d = Decomposition::build(&m, p, &round_robin(m.n_cells, p)).expect("decompose");
            let parts: Vec<Vec<Scalar>> = (0..p)
                .map(|q| d.split_field(q, &global).expect("split"))
                .collect();
            for (q, f) in parts.iter().enumerate() {
                assert_eq!(f.len(), d.parts[q].n_local());
                for (h, &v) in f.iter().enumerate().skip(d.parts[q].mesh.n_cells) {
                    assert_eq!(v, 0.0, "halo slot {h} must start empty");
                }
            }
            let back = d.gather_field(&parts).expect("gather");
            assert_eq!(back, global);
        }
    }
}
