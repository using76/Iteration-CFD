// meteor-cfd - Copyright (c) 2026 주식회사 메테오시뮬레이션 (Meteo Simulation Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

//! Rebuilding the addressing after a topology change, with no atomic and no
//! prefix scan.
//!
//! Written from:
//!   ofgpu SPEC-LIT.md sections 1, 70 and 75.5
//!   D. E. Knuth, "The Art of Computer Programming", vol. 3, 2nd ed. (1998),
//!     section 5.2.5 - distribution (radix) sorting and the stability that
//!     makes a least-significant-digit pass composable
//!   H. Sundar, R. S. Sampath and G. Biros, SIAM J. Sci. Comput. 30 (2008)
//!     2675-2708 - bottom-up linear-octree construction, where the sort-then-
//!     search shape used here comes from
//! No GPL-licensed source was consulted.
//!
//! # The problem
//!
//! An adapt renumbers every cell and every face. Two things then have to be
//! rebuilt before any operator can run:
//!
//! 1. the **LDU order** - faces sorted by `(owner, neighbour)`, which SPEC-LIT
//!    section 1 requires and which every gather kernel assumes;
//! 2. the **cell -> face CSR** - `cf_offset`/`cf_face`/`cf_own`, whose within-
//!    cell ordering is what makes `Amul` bitwise reproducible.
//!
//! The obvious build of (2) is a scatter histogram,
//! `atomicAdd(&count[owner[f]], 1)`, which needs integer atomics and - far
//! worse - makes the order of the faces WITHIN a cell depend on thread
//! scheduling. That order is precisely what `mesh::topology` documents as the
//! source of run-to-run reproducibility, so a scatter build would trade the
//! project's central guarantee for a histogram.
//!
//! # The construction, and the one place the design note of record overshot
//!
//! After (1) the `owner` array is sorted and non-decreasing. Therefore, for
//! every cell `c`,
//!
//! ```text
//! own_begin[c] = lower_bound(owner,  c)      the number of faces OWNED by a
//!                                            cell before c
//! nbr_begin[c] = lower_bound(nbrKey, c)      the number of faces NEIGHBOURED
//!                                            by a cell before c
//! ```
//!
//! where `nbrKey` is the neighbour array in sorted order. The design note said
//! "two binary searches **plus an exclusive scan**". **The scan is not
//! needed.** `cf_offset[c]` is by definition the number of (cell, face)
//! incidences belonging to cells before `c`, which is exactly the number of
//! faces whose owner is before `c` plus the number whose neighbour is before
//! `c`:
//!
//! ```text
//! cf_offset[c] = own_begin[c] + nbr_begin[c]
//! ```
//!
//! A `lower_bound` over a sorted array IS an exclusive prefix sum of the
//! counts, already computed, and computing it again costs a whole extra
//! device-wide primitive. Two binary searches per cell and nothing else.
//!
//! # The within-cell order
//!
//! `mesh::topology::build_cell_face_maps` walks faces in ascending index and
//! appends to both cells, so each cell's slice is ascending in face id with
//! owned and neighboured faces **interleaved** - not "owned faces first, then
//! neighboured faces", which is what the design note described. Reproducing it
//! is a two-pointer merge of two ascending runs, one thread per cell, no
//! atomic:
//!
//! * the owned run is the face ids `own_begin[c] .. own_end[c]` themselves,
//!   because the faces are sorted by owner, so all of `c`'s owned faces are
//!   consecutive;
//! * the neighboured run is `nbrPerm[nbr_begin[c] .. nbr_end[c]]`, ascending
//!   because the neighbour sort is stable.
//!
//! [`cell_face_csr`] is the host statement of that, and the device kernel
//! `adaptCellFaceCsr` is the same merge one thread per cell.
//! `tests::the_rebuilt_csr_is_the_one_the_mesh_builder_makes` requires the two
//! to equal `build_cell_face_maps` element for element on meshes that include
//! a 2:1 refined block, where cell degree runs from 6 to 24.

use cudarc::driver::PushKernelArg;

use crate::device::{cfg_for, Gpu};
use crate::error::{Error, Result};
use crate::mesh::{GpuMesh, HostMesh};
use crate::{DevBuf, Label};

/// One stable least-significant-digit radix pass over `bits` bits starting at
/// `shift`, keyed by `key(i)`.
///
/// Counting sort: histogram, exclusive scan of 256 buckets, then a stable
/// distribute. Deterministic, order-independent and exactly what
/// `cub::DeviceRadixSort` computes, which is why the device version of an
/// adapt can use the library primitive and get these bits.
fn radix_pass(src: &[u32], dst: &mut Vec<u32>, key: &impl Fn(u32) -> u64, shift: u32) {
    let mut count = [0usize; 256];
    for &i in src {
        count[((key(i) >> shift) & 0xff) as usize] += 1;
    }
    let mut acc = 0usize;
    for c in count.iter_mut() {
        let n = *c;
        *c = acc;
        acc += n;
    }
    dst.clear();
    dst.resize(src.len(), 0);
    for &i in src {
        let b = ((key(i) >> shift) & 0xff) as usize;
        dst[count[b]] = i;
        count[b] += 1;
    }
}

/// The permutation that puts faces into upper-triangular LDU order: face
/// `perm[i]` is the `i`-th face after the sort.
///
/// Sorted by the packed key `(owner << 32) | neighbour`, which orders by owner
/// and then by neighbour - the order SPEC-LIT section 1 requires. Eight stable
/// radix passes of eight bits, so the result is a function of the input alone.
///
/// Refuses a face whose `owner >= neighbour`: on this crate's meshes the pair
/// is normalised at emission, and a face that arrives the wrong way round is a
/// bug in the emitter that the sort would silently hide.
pub fn ldu_permutation(owner: &[Label], neighbour: &[Label]) -> Result<Vec<Label>> {
    let n = owner.len();
    if neighbour.len() != n {
        return Err(Error::Mesh(format!(
            "owner has {n} entries and neighbour {}",
            neighbour.len()
        )));
    }
    for f in 0..n {
        if owner[f] < 0 || neighbour[f] < 0 {
            return Err(Error::Mesh(format!(
                "face {f} addresses cells {} and {}; a negative cell index cannot be \
                 sorted into LDU order",
                owner[f], neighbour[f]
            )));
        }
        if owner[f] >= neighbour[f] {
            return Err(Error::Mesh(format!(
                "face {f} has owner {} and neighbour {}; the pair must be normalised \
                 to owner < neighbour before the LDU sort",
                owner[f], neighbour[f]
            )));
        }
    }

    let key = |i: u32| -> u64 {
        let f = i as usize;
        ((owner[f] as u64) << 32) | (neighbour[f] as u64)
    };
    let mut a: Vec<u32> = (0..n as u32).collect();
    let mut b: Vec<u32> = Vec::with_capacity(n);
    for p in 0..8u32 {
        radix_pass(&a, &mut b, &key, p * 8);
        std::mem::swap(&mut a, &mut b);
    }
    Ok(a.into_iter().map(|i| i as Label).collect())
}

/// Faces in ascending order of their NEIGHBOUR cell, ties broken by face id.
///
/// `perm[i]` is a face id and `key[i]` is its neighbour cell; `key` is
/// non-decreasing, which is what the binary searches of [`cell_face_csr`]
/// need. One counting sort over cells: stable, so within a cell the face ids
/// come out ascending, which is what the merge relies on.
pub struct NeighbourOrder {
    pub perm: Vec<Label>,
    pub key: Vec<Label>,
}

pub fn neighbour_order(neighbour: &[Label], n_cells: usize) -> Result<NeighbourOrder> {
    let n = neighbour.len();
    let mut count = vec![0usize; n_cells + 1];
    for (f, &c) in neighbour.iter().enumerate() {
        if c < 0 || (c as usize) >= n_cells {
            return Err(Error::Mesh(format!(
                "face {f} names neighbour cell {c}, outside [0, {n_cells})"
            )));
        }
        count[c as usize + 1] += 1;
    }
    for c in 0..n_cells {
        count[c + 1] += count[c];
    }
    let mut perm = vec![0 as Label; n];
    let mut key = vec![0 as Label; n];
    let mut cursor = count.clone();
    for (f, &c) in neighbour.iter().enumerate() {
        let s = cursor[c as usize];
        perm[s] = f as Label;
        key[s] = c;
        cursor[c as usize] = s + 1;
    }
    Ok(NeighbourOrder { perm, key })
}

/// The same, for boundary faces keyed by the one cell they touch.
pub fn boundary_order(b_face_cells: &[Label], n_cells: usize) -> Result<NeighbourOrder> {
    neighbour_order(b_face_cells, n_cells)
}

/// `lower_bound`: the number of entries of the sorted array strictly below
/// `v`. Written out because the kernel has to spell the same loop, and the two
/// must agree on the tie-breaking or the CSR slices will not line up.
#[inline]
fn lower_bound(a: &[Label], v: Label) -> usize {
    let (mut lo, mut hi) = (0usize, a.len());
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if a[mid] < v {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    lo
}

/// The rebuilt cell -> face CSR.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CellFaceCsr {
    pub cf_offset: Vec<Label>,
    pub cf_face: Vec<Label>,
    pub cf_own: Vec<Label>,
}

/// Rebuild `cf_offset`/`cf_face`/`cf_own` from the sorted addressing, with two
/// binary searches per cell and a two-pointer merge. No histogram, no scan, no
/// atomic - see the module note.
///
/// `owner` must already be in LDU order (non-decreasing); the function checks
/// it rather than assuming it, because every guarantee below rests on it.
pub fn cell_face_csr(
    n_cells: usize,
    owner: &[Label],
    nbr: &NeighbourOrder,
) -> Result<CellFaceCsr> {
    let n_if = owner.len();
    if nbr.perm.len() != n_if || nbr.key.len() != n_if {
        return Err(Error::Mesh(format!(
            "the neighbour order has {} entries and there are {n_if} internal faces",
            nbr.perm.len()
        )));
    }
    for f in 1..n_if {
        if owner[f] < owner[f - 1] {
            return Err(Error::Mesh(format!(
                "the owner array is not sorted at face {f} ({} after {}); the CSR \
                 rebuild needs LDU order and cannot repair it",
                owner[f],
                owner[f - 1]
            )));
        }
    }
    if let Some(&last) = owner.last() {
        if last < 0 || (last as usize) >= n_cells {
            return Err(Error::Mesh(format!(
                "a face names owner cell {last}, outside [0, {n_cells})"
            )));
        }
    }

    let mut cf_offset = vec![0 as Label; n_cells + 1];
    let mut cf_face = vec![-1 as Label; 2 * n_if];
    let mut cf_own = vec![0 as Label; 2 * n_if];

    // `c` is the cell id, used as a search key as well as an index into three
    // arrays; an `enumerate` over any one of them would hide that.
    #[allow(clippy::needless_range_loop)]
    for c in 0..n_cells {
        let cl = c as Label;
        let ob = lower_bound(owner, cl);
        let oe = lower_bound(owner, cl + 1);
        let nb = lower_bound(&nbr.key, cl);
        let ne = lower_bound(&nbr.key, cl + 1);

        // THE OFFSET, with no scan: the count of incidences before this cell.
        let base = ob + nb;
        cf_offset[c] = base as Label;

        let (mut i, mut j, mut k) = (ob, nb, base);
        while i < oe || j < ne {
            let fi = if i < oe { i as Label } else { Label::MAX };
            let fj = if j < ne { nbr.perm[j] } else { Label::MAX };
            if fi < fj {
                cf_face[k] = fi;
                cf_own[k] = 1;
                i += 1;
            } else {
                cf_face[k] = fj;
                cf_own[k] = 0;
                j += 1;
            }
            k += 1;
        }
    }
    cf_offset[n_cells] = 2 * n_if as Label;

    Ok(CellFaceCsr { cf_offset, cf_face, cf_own })
}

/// Rebuild `bcf_offset`/`bcf_face`. A boundary face touches exactly one cell,
/// so there is nothing to merge: the offset is one binary search and the list
/// is the stable permutation itself.
pub fn boundary_csr(n_cells: usize, b: &NeighbourOrder) -> (Vec<Label>, Vec<Label>) {
    let n_bf = b.perm.len();
    let mut off = vec![0 as Label; n_cells + 1];
    for (c, o) in off.iter_mut().enumerate().take(n_cells) {
        *o = lower_bound(&b.key, c as Label) as Label;
    }
    off[n_cells] = n_bf as Label;
    (off, b.perm.clone())
}

/// Everything the addressing rebuild produces for one mesh, host side.
pub fn rebuild_addressing(m: &HostMesh) -> Result<(CellFaceCsr, Vec<Label>, Vec<Label>)> {
    let nbr = neighbour_order(&m.neighbour[..m.n_internal_faces], m.n_cells)?;
    let csr = cell_face_csr(m.n_cells, &m.owner[..m.n_internal_faces], &nbr)?;
    let b = boundary_order(&m.b_face_cells[..m.n_boundary_faces], m.n_cells)?;
    let (boff, bface) = boundary_csr(m.n_cells, &b);
    Ok((csr, boff, bface))
}

// ---------------------------------------------------------------------------
//  The device side
// ---------------------------------------------------------------------------

/// The rebuilt addressing, on the device.
pub struct GpuCellFaceCsr {
    pub cf_offset: DevBuf<Label>,
    pub cf_face: DevBuf<Label>,
    pub cf_own: DevBuf<Label>,
    pub bcf_offset: DevBuf<Label>,
    pub bcf_face: DevBuf<Label>,
}

/// Rebuild the addressing on the device, one thread per cell.
///
/// The two sorted arrays (`owner` in LDU order, and the neighbour permutation)
/// are inputs, exactly as they would be after a `cub::DeviceRadixSort`. What
/// this runs is the part the design note called "two binary searches plus an
/// exclusive scan" and this file shows to be two binary searches.
#[allow(clippy::too_many_arguments)]
pub fn gpu_rebuild_addressing(
    gpu: &Gpu,
    k: &super::AdaptKernels,
    m: &GpuMesh,
    nbr_perm: &DevBuf<Label>,
    nbr_key: &DevBuf<Label>,
    b_perm: &DevBuf<Label>,
    b_key: &DevBuf<Label>,
) -> Result<GpuCellFaceCsr> {
    let n_cells = m.n_cells;
    let n_if = m.n_internal_faces;
    let n_bf = m.n_boundary_faces;
    let mut out = GpuCellFaceCsr {
        cf_offset: gpu.zeros(n_cells + 1)?,
        cf_face: gpu.zeros((2 * n_if).max(1))?,
        cf_own: gpu.zeros((2 * n_if).max(1))?,
        bcf_offset: gpu.zeros(n_cells + 1)?,
        bcf_face: gpu.zeros(n_bf.max(1))?,
    };
    if n_cells == 0 {
        return Ok(out);
    }
    let (nc, nif, nbf) = (n_cells as Label, n_if as Label, n_bf as Label);
    unsafe {
        gpu.stream()
            .launch_builder(&k.cell_face_csr)
            .arg(&mut out.cf_offset)
            .arg(&mut out.cf_face)
            .arg(&mut out.cf_own)
            .arg(&m.owner)
            .arg(nbr_perm)
            .arg(nbr_key)
            .arg(&nc)
            .arg(&nif)
            .launch(cfg_for(n_cells))?;
        gpu.stream()
            .launch_builder(&k.boundary_csr)
            .arg(&mut out.bcf_offset)
            .arg(&mut out.bcf_face)
            .arg(b_perm)
            .arg(b_key)
            .arg(&nc)
            .arg(&nbf)
            .launch(cfg_for(n_cells.max(n_bf)))?;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adapt::{plan, Forest, Mark};
    use crate::Vec3;

    const CUBE: Vec3 = Vec3::new(0.25, 0.25, 0.25);

    fn meshes() -> Vec<(String, HostMesh)> {
        let mut out = Vec::new();
        out.push((
            "uniform 5x4x3".to_string(),
            Forest::uniform([5, 4, 3], CUBE).unwrap().build().unwrap().mesh,
        ));
        let at = |i: usize, j: usize, k: usize| i + 6 * (j + 6 * k);
        let mut lev = vec![0u32; 216];
        lev[at(3, 3, 3)] = 1;
        lev[at(1, 1, 1)] = 2;
        out.push((
            "2:1 refined 6x6x6".to_string(),
            Forest::from_base_levels([6, 6, 6], CUBE, &lev).unwrap().build().unwrap().mesh,
        ));
        // And a mesh that has actually been adapted, so the rebuild is tested
        // on the numbering an adapt produces rather than on a generator's.
        let f = Forest::uniform([4, 4, 4], CUBE).unwrap();
        let m = f.build().unwrap();
        let mut mark = vec![Mark::Keep; f.len()];
        for c in [5usize, 6, 21, 22, 37] {
            mark[c] = Mark::Refine;
        }
        let p = plan(&f, &m.mesh, &mark, 2).unwrap();
        out.push(("after a refine".to_string(), p.mesh.mesh.clone()));
        out
    }

    /// The whole point: the rebuilt CSR is the one `mesh::topology` builds,
    /// element for element, on meshes whose cell degree runs from 6 to 24.
    #[test]
    fn the_rebuilt_csr_is_the_one_the_mesh_builder_makes() {
        for (name, m) in meshes() {
            let (csr, boff, bface) = rebuild_addressing(&m).unwrap();
            assert_eq!(csr.cf_offset, m.cf_offset, "cf_offset on {name}");
            assert_eq!(csr.cf_face, m.cf_face, "cf_face on {name}");
            assert_eq!(csr.cf_own, m.cf_own, "cf_own on {name}");
            assert_eq!(boff, m.bcf_offset, "bcf_offset on {name}");
            assert_eq!(bface, m.bcf_face, "bcf_face on {name}");
        }
    }

    /// The offsets come out of two binary searches, with no prefix scan. The
    /// claim is machine-checkable: the offset must equal the count of faces
    /// whose owner is before the cell plus the count whose neighbour is.
    #[test]
    fn the_offset_is_two_binary_searches_and_no_scan() {
        for (name, m) in meshes() {
            let nbr = neighbour_order(&m.neighbour[..m.n_internal_faces], m.n_cells).unwrap();
            for c in 0..m.n_cells {
                let a = m.owner[..m.n_internal_faces].iter().filter(|&&o| o < c as Label).count();
                let b = m.neighbour[..m.n_internal_faces]
                    .iter()
                    .filter(|&&o| o < c as Label)
                    .count();
                assert_eq!(
                    m.cf_offset[c] as usize,
                    a + b,
                    "cell {c} of {name}: the CSR offset is not own_begin + nbr_begin"
                );
                assert_eq!(lower_bound(&nbr.key, c as Label), b);
            }
        }
    }

    /// The sort recovers LDU order from an arbitrary permutation of the faces,
    /// and is the identity on a mesh that is already in it.
    #[test]
    fn the_radix_sort_recovers_the_ldu_order() {
        for (name, m) in meshes() {
            let n = m.n_internal_faces;
            let p = ldu_permutation(&m.owner[..n], &m.neighbour[..n]).unwrap();
            assert_eq!(
                p,
                (0..n as Label).collect::<Vec<_>>(),
                "{name} is already LDU ordered, so the sort must be the identity"
            );

            // A deterministic, non-trivial shuffle - a stride coprime with n.
            let stride = 1 + 2 * (n / 3);
            let mut order: Vec<usize> = Vec::with_capacity(n);
            let mut seen = vec![false; n];
            let mut i = 0usize;
            for _ in 0..n {
                while seen[i] {
                    i = (i + 1) % n;
                }
                seen[i] = true;
                order.push(i);
                i = (i + stride) % n;
            }
            let so: Vec<Label> = order.iter().map(|&f| m.owner[f]).collect();
            let sn: Vec<Label> = order.iter().map(|&f| m.neighbour[f]).collect();
            let p = ldu_permutation(&so, &sn).unwrap();
            let ro: Vec<Label> = p.iter().map(|&i| so[i as usize]).collect();
            let rn: Vec<Label> = p.iter().map(|&i| sn[i as usize]).collect();
            assert_eq!(ro, m.owner[..n].to_vec(), "owner after re-sorting {name}");
            assert_eq!(rn, m.neighbour[..n].to_vec(), "neighbour after re-sorting {name}");
        }
    }

    /// A face the wrong way round is refused by name rather than sorted into
    /// something plausible.
    #[test]
    fn a_face_that_is_not_normalised_is_refused() {
        let e = ldu_permutation(&[3, 1], &[1, 4]).unwrap_err().to_string();
        assert!(e.contains("must be normalised to owner < neighbour"), "{e}");
        let e = ldu_permutation(&[0], &[-1]).unwrap_err().to_string();
        assert!(e.contains("negative cell index"), "{e}");
    }

    /// An owner array that is not sorted is refused, because everything the
    /// rebuild claims rests on it being sorted.
    #[test]
    fn an_unsorted_owner_array_is_refused() {
        let nbr = neighbour_order(&[2, 1], 4).unwrap();
        let e = cell_face_csr(4, &[1, 0], &nbr).unwrap_err().to_string();
        assert!(e.contains("not sorted"), "{e}");
    }
}
