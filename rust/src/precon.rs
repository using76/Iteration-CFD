// meteor-cfd - Copyright (c) 2026 주식회사 메테오시뮬레이션 (Meteo Simulation Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

//! Multi-colour incomplete factorisation - DIC and DILU.
//!
//! Written from:
//!   Saad, *Iterative Methods for Sparse Linear Systems*, 2nd ed. (2003),
//!     ch. 10 (ILU(0)/IC(0)) and §12.4 (multicolour ILU)
//!   ofgpu `SPEC-LIT.md` §21, which specifies the colouring, the one-kernel-
//!     per-colour schedule and the forward/backward sweep order, and §8.3,
//!     which deferred this until §21 existed.
//! No GPL-licensed source was consulted.
//!
//! # What was wrong before
//!
//! `preconditioner DIC;` and `preconditioner DILU;` parsed into the enum and
//! then silently ran Jacobi. That was defensible while there was nothing else
//! to run - a sequential sweep executed in parallel does not compute a *worse*
//! preconditioner, it computes an undefined one that changes with the block
//! schedule - but it was never announced. `SPEC-LIT` §21 now specifies the
//! parallel form, so the substitution is gone and so is the silence.
//!
//! # The factorisation
//!
//! With `A = L + D + U` in some ordering, the no-fill diagonal-based
//! incomplete factorisation is
//!
//! ```text
//! M = (Dt + L)·Dt^-1·(Dt + U),      Dt chosen so that diag(M) = diag(A)
//! Dt_v = A_vv - Σ_{u ≺ v} A_vu·A_uv / Dt_u
//! ```
//!
//! The `u ≺ v` is the only sequential thing about it. Saad §12.4 and
//! `SPEC-LIT` §21 take the ordering from a **colouring of the matrix graph**:
//! all of colour 0, then all of colour 1, and so on. No two neighbours share a
//! colour, so when a cell of colour `c` evaluates the sum, every term comes
//! from a strictly earlier colour and is already final - cells of its own
//! colour are, by construction, not its neighbours. One kernel per colour is
//! therefore correct with no ordering *inside* a launch, and the result is
//! schedule-independent and bitwise reproducible.
//!
//! DIC is the symmetric case (Cholesky), DILU the asymmetric one. Both use
//! only the off-diagonals the matrix already holds - no fill-in - so the
//! storage is the matrix plus one reciprocal-diagonal array.
//!
//! # What the colouring costs
//!
//! Fewer colours means fewer launches and more parallelism per launch. Greedy
//! colouring in the mesh's own cell order gives exactly 2 on a structured hex
//! mesh (it reproduces the parity colouring) and typically 5-8 on an
//! unstructured tetrahedral one, which is what `SPEC-LIT` §21 predicts.

use cudarc::driver::{CudaFunction, PushKernelArg};

use crate::device::{cfg_for, DevBuf, Gpu, KernelSet};
use crate::error::{Error, Result};
use crate::ldu::GpuLduMatrix;
use crate::mesh::{GpuMesh, HostMesh};
use crate::{Label, Scalar};

// ==========================================================================
//  The graph
// ==========================================================================

/// The LDU adjacency, borrowed.
///
/// The colouring needs nothing but the graph, and the graph exists in two
/// places - a [`HostMesh`] during setup and a [`GpuMesh`] afterwards. Taking a
/// view of it rather than one of those types means the algorithm is written
/// once and can be tested on a hand-built graph that is not a mesh at all.
pub struct Adjacency<'a> {
    pub n_cells: usize,
    pub n_internal_faces: usize,
    pub owner: &'a [Label],
    pub neighbour: &'a [Label],
    pub cf_offset: &'a [Label],
    pub cf_face: &'a [Label],
    pub cf_own: &'a [Label],
}

impl<'a> Adjacency<'a> {
    pub fn of(m: &'a HostMesh) -> Self {
        Self {
            n_cells: m.n_cells,
            n_internal_faces: m.n_internal_faces,
            owner: &m.owner,
            neighbour: &m.neighbour,
            cf_offset: &m.cf_offset,
            cf_face: &m.cf_face,
            cf_own: &m.cf_own,
        }
    }

    /// The cell across face `cf_face[j]` from the cell whose CSR slot `j` is.
    #[inline]
    fn across(&self, j: usize) -> usize {
        let f = self.cf_face[j] as usize;
        if self.cf_own[j] != 0 {
            self.neighbour[f] as usize
        } else {
            self.owner[f] as usize
        }
    }
}

/// The five arrays [`Adjacency`] borrows, owned - what a [`GpuMesh`] has to be
/// copied into before it can be coloured.
pub struct HostAdjacency {
    pub n_cells: usize,
    pub n_internal_faces: usize,
    pub owner: Vec<Label>,
    pub neighbour: Vec<Label>,
    pub cf_offset: Vec<Label>,
    pub cf_face: Vec<Label>,
    pub cf_own: Vec<Label>,
}

impl HostAdjacency {
    /// Copy the graph back from the device. Setup only: five device-to-host
    /// copies, and nothing in a time loop may do that.
    pub fn download(gpu: &Gpu, m: &GpuMesh) -> Result<Self> {
        Ok(Self {
            n_cells: m.n_cells,
            n_internal_faces: m.n_internal_faces,
            owner: gpu.download(&m.owner)?,
            neighbour: gpu.download(&m.neighbour)?,
            cf_offset: gpu.download(&m.cf_offset)?,
            cf_face: gpu.download(&m.cf_face)?,
            cf_own: gpu.download(&m.cf_own)?,
        })
    }

    pub fn view(&self) -> Adjacency<'_> {
        Adjacency {
            n_cells: self.n_cells,
            n_internal_faces: self.n_internal_faces,
            owner: &self.owner,
            neighbour: &self.neighbour,
            cf_offset: &self.cf_offset,
            cf_face: &self.cf_face,
            cf_own: &self.cf_own,
        }
    }
}

// ==========================================================================
//  The colouring
// ==========================================================================

/// A colouring of the LDU adjacency: no two cells joined by an internal face
/// share a colour.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Colouring {
    pub n_colours: usize,
    /// `[n_cells]` each cell's colour.
    pub colour: Vec<Label>,
    /// `[n_colours + 1]` where each colour's block starts in [`Self::cells`].
    pub offsets: Vec<usize>,
    /// `[n_cells]` cell indices grouped by colour.
    pub cells: Vec<Label>,
}

impl Colouring {
    /// Greedy colouring of the mesh's cell graph.
    ///
    /// Cells are visited in their own index order and each takes the lowest
    /// colour none of its already-coloured neighbours holds. That is the
    /// standard greedy algorithm; on a structured mesh in lexicographic order
    /// every already-coloured neighbour of a cell is of the opposite parity, so
    /// the result is the two-colour parity colouring - the best possible.
    pub fn greedy(g: &Adjacency<'_>) -> Self {
        Self::greedy_in_order(g, None)
    }

    /// The same, visiting cells in `order` instead of index order.
    ///
    /// Exposed because the *quality* of a greedy colouring depends on the
    /// visiting order, and a test that wants to prove the preconditioner does
    /// not depend on which valid colouring it was handed needs a second one.
    pub fn greedy_in_order(g: &Adjacency<'_>, order: Option<&[Label]>) -> Self {
        let n = g.n_cells;
        let mut colour = vec![-1 as Label; n];

        // `used[c]` is stamped with the cell currently being coloured, so the
        // scratch is cleared in O(degree) rather than O(n_colours) per cell.
        let mut used_by = vec![Label::MIN; n.max(1) + 1];
        let mut n_colours = 0usize;

        let visit = |i: usize| -> usize {
            match order {
                Some(o) => o[i] as usize,
                None => i,
            }
        };

        for i in 0..n {
            let c = visit(i);

            for j in g.cf_offset[c] as usize..g.cf_offset[c + 1] as usize {
                let nbr = g.across(j);
                let cn = colour[nbr];
                if cn >= 0 && (cn as usize) < used_by.len() {
                    used_by[cn as usize] = c as Label;
                }
            }

            // A cell of degree d always fits in one of the colours 0..=d, and
            // no cell has degree n, so `used_by` (sized n+1) cannot be
            // exhausted. The resize is there so that a malformed adjacency
            // cannot walk off the end rather than because the bound can fail.
            let mut pick = 0usize;
            while pick < used_by.len() && used_by[pick] == c as Label {
                pick += 1;
            }
            if pick >= used_by.len() {
                used_by.resize(pick + 1, Label::MIN);
            }

            colour[c] = pick as Label;
            n_colours = n_colours.max(pick + 1);
        }

        Self::from_colours(colour, n_colours)
    }

    /// Group an existing per-cell colour array into contiguous blocks.
    fn from_colours(colour: Vec<Label>, n_colours: usize) -> Self {
        let n = colour.len();
        let mut counts = vec![0usize; n_colours + 1];
        for &c in &colour {
            counts[c as usize + 1] += 1;
        }
        for i in 0..n_colours {
            counts[i + 1] += counts[i];
        }
        let offsets = counts.clone();

        let mut cursor = counts;
        let mut cells = vec![0 as Label; n];
        for (cell, &c) in colour.iter().enumerate() {
            let slot = cursor[c as usize];
            cells[slot] = cell as Label;
            cursor[c as usize] += 1;
        }

        Self {
            n_colours,
            colour,
            offsets,
            cells,
        }
    }

    /// The same colouring with the colour LABELS reversed, i.e. the same
    /// partition eliminated in the opposite order.
    ///
    /// Still a valid colouring - reversing labels cannot make two neighbours
    /// agree - but a different elimination ordering, hence a different (and
    /// equally legitimate) incomplete factorisation. `SPEC-LIT` §21's test
    /// asks what that does to the iteration count.
    pub fn with_reversed_colour_order(&self) -> Self {
        let top = self.n_colours.saturating_sub(1) as Label;
        let colour = self.colour.iter().map(|c| top - c).collect();
        Self::from_colours(colour, self.n_colours)
    }

    /// The same colouring with the cells shuffled INSIDE each colour.
    ///
    /// The elimination ordering is untouched; only the order the threads of
    /// one launch happen to visit cells changes. Because no two cells of a
    /// colour are neighbours, that must make no difference at all - which is
    /// the schedule-independence `SPEC-LIT` §21 is after, and is testable
    /// bit for bit.
    pub fn with_shuffled_cells_within_colours(&self, seed: u64) -> Self {
        let mut out = self.clone();
        let mut s = seed | 1;
        let mut next = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };

        for c in 0..self.n_colours {
            let (lo, hi) = (self.offsets[c], self.offsets[c + 1]);
            let block = &mut out.cells[lo..hi];
            if block.len() < 2 {
                continue;
            }
            for i in (1..block.len()).rev() {
                let j = (next() % (i as u64 + 1)) as usize;
                block.swap(i, j);
            }
        }
        out
    }

    /// Is this actually a colouring? Nothing downstream is correct if it is
    /// not, and the check is O(faces).
    pub fn is_valid(&self, g: &Adjacency<'_>) -> bool {
        if self.colour.len() != g.n_cells {
            return false;
        }
        for f in 0..g.n_internal_faces {
            let (o, n) = (g.owner[f] as usize, g.neighbour[f] as usize);
            if self.colour[o] == self.colour[n] {
                return false;
            }
        }
        true
    }
}

/// [`Colouring`], uploaded.
pub struct GpuColouring {
    pub n_colours: usize,
    pub colour: DevBuf<Label>,
    pub cells: DevBuf<Label>,
    /// Kept on the host: it is a launch bound, not kernel data.
    pub offsets: Vec<usize>,
}

impl GpuColouring {
    pub fn upload(gpu: &Gpu, c: &Colouring) -> Result<Self> {
        Ok(Self {
            n_colours: c.n_colours,
            colour: gpu.upload(&c.colour)?,
            cells: gpu.upload(&c.cells)?,
            offsets: c.offsets.clone(),
        })
    }
}

// ==========================================================================
//  Kernels
// ==========================================================================

pub struct PreconKernels {
    factor: CudaFunction,
    forward: CudaFunction,
    backward: CudaFunction,
}

impl PreconKernels {
    pub fn new(gpu: &Gpu) -> Result<Self> {
        let k = KernelSet::new(gpu, crate::kernels::PRECON)?;
        Ok(Self {
            factor: k.func("pcFactorColour")?,
            forward: k.func("pcForwardColour")?,
            backward: k.func("pcBackwardColour")?,
        })
    }
}

// ==========================================================================
//  The preconditioner
// ==========================================================================

/// A multi-colour DIC/DILU preconditioner bound to one mesh.
///
/// Holds the colouring and the kernels; the reciprocal diagonal lives in the
/// solver workspace next to the Jacobi one, because exactly one of the two is
/// ever live and there is no reason to pay for both.
pub struct MultiColour {
    pub colouring: GpuColouring,
    kernels: PreconKernels,
}

impl MultiColour {
    /// Colour the mesh and upload. Setup only - it copies the graph back to
    /// the host and allocates, so it may not run inside a time loop.
    pub fn new(gpu: &Gpu, m: &GpuMesh) -> Result<Self> {
        let owned = HostAdjacency::download(gpu, m)?;
        let g = owned.view();
        let c = Colouring::greedy(&g);
        Self::from_colouring(gpu, &g, &c)
    }

    /// Build from a colouring supplied by the caller.
    ///
    /// The colouring is validated here rather than trusted: an invalid one
    /// does not fail loudly on the device, it silently drops the terms where
    /// two neighbours share a colour and leaves a preconditioner that is
    /// merely worse than it should be. That is precisely the class of bug this
    /// module exists to remove.
    pub fn from_colouring(gpu: &Gpu, g: &Adjacency<'_>, c: &Colouring) -> Result<Self> {
        if !c.is_valid(g) {
            return Err(Error::Config(
                "multi-colour preconditioner: the colouring gives two \
                 face-neighbours the same colour, so the per-colour sweeps \
                 would not be independent (SPEC-LIT 21)"
                    .to_string(),
            ));
        }
        Ok(Self {
            colouring: GpuColouring::upload(gpu, c)?,
            kernels: PreconKernels::new(gpu)?,
        })
    }

    pub fn n_colours(&self) -> usize {
        self.colouring.n_colours
    }

    /// Compute `rD = 1/Dt`, colour by colour, in ascending colour order.
    ///
    /// `symmetric` selects the DIC (Cholesky) form, which uses `upper²` in
    /// place of `upper·lower`. The two agree exactly when the matrix really is
    /// symmetric; the caller is expected to have checked
    /// ([`crate::solver::matrix_is_symmetric`]) rather than to hope.
    pub fn factorise(
        &self,
        gpu: &Gpu,
        r_diag: &mut DevBuf<Scalar>,
        a: &GpuLduMatrix,
        m: &GpuMesh,
        symmetric: bool,
    ) -> Result<()> {
        self.check(a, m, r_diag)?;
        let sym = Label::from(symmetric);

        for c in 0..self.colouring.n_colours {
            let (start, count) = self.block(c);
            if count == 0 {
                continue;
            }
            unsafe {
                gpu.stream()
                    .launch_builder(&self.kernels.factor)
                    .arg(&mut *r_diag)
                    .arg(&a.diag)
                    .arg(&a.upper)
                    .arg(&a.lower)
                    .arg(&self.colouring.colour)
                    .arg(&self.colouring.cells)
                    .arg(&m.owner)
                    .arg(&m.neighbour)
                    .arg(&m.cf_offset)
                    .arg(&m.cf_face)
                    .arg(&m.cf_own)
                    .arg(&sym)
                    .arg(&start)
                    .arg(&count_label(count))
                    .launch(cfg_for(count))?;
            }
        }
        Ok(())
    }

    /// `y ← M^-1·y`, in place.
    ///
    /// Forward sweep over the colours in order, backward sweep in reverse -
    /// `SPEC-LIT` §21 step 2 and step 3. The caller puts `x` in `y` first;
    /// doing the copy here would cost a launch on the path where `y` and `x`
    /// are already the same buffer.
    pub fn apply(
        &self,
        gpu: &Gpu,
        y: &mut DevBuf<Scalar>,
        r_diag: &DevBuf<Scalar>,
        a: &GpuLduMatrix,
        m: &GpuMesh,
    ) -> Result<()> {
        self.check(a, m, y)?;

        for c in 0..self.colouring.n_colours {
            self.sweep(gpu, &self.kernels.forward, y, r_diag, a, m, c)?;
        }
        for c in (0..self.colouring.n_colours).rev() {
            self.sweep(gpu, &self.kernels.backward, y, r_diag, a, m, c)?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn sweep(
        &self,
        gpu: &Gpu,
        f: &CudaFunction,
        y: &mut DevBuf<Scalar>,
        r_diag: &DevBuf<Scalar>,
        a: &GpuLduMatrix,
        m: &GpuMesh,
        colour: usize,
    ) -> Result<()> {
        let (start, count) = self.block(colour);
        if count == 0 {
            return Ok(());
        }
        unsafe {
            gpu.stream()
                .launch_builder(f)
                .arg(&mut *y)
                .arg(r_diag)
                .arg(&a.upper)
                .arg(&a.lower)
                .arg(&self.colouring.colour)
                .arg(&self.colouring.cells)
                .arg(&m.owner)
                .arg(&m.neighbour)
                .arg(&m.cf_offset)
                .arg(&m.cf_face)
                .arg(&m.cf_own)
                .arg(&start)
                .arg(&count_label(count))
                .launch(cfg_for(count))?;
        }
        Ok(())
    }

    fn block(&self, colour: usize) -> (Label, usize) {
        let lo = self.colouring.offsets[colour];
        let hi = self.colouring.offsets[colour + 1];
        (lo as Label, hi - lo)
    }

    fn check<T>(&self, a: &GpuLduMatrix, m: &GpuMesh, v: &DevBuf<T>) -> Result<()> {
        if a.n_cells != m.n_cells {
            return Err(Error::Config(format!(
                "multi-colour preconditioner: matrix has {} cells, mesh has {}",
                a.n_cells, m.n_cells
            )));
        }
        if self.colouring.colour.len() != m.n_cells {
            return Err(Error::Config(format!(
                "multi-colour preconditioner: the colouring is for {} cells, \
                 the mesh has {}",
                self.colouring.colour.len(),
                m.n_cells
            )));
        }
        if v.len() < m.n_cells {
            return Err(Error::Config(format!(
                "multi-colour preconditioner: a vector holds {} values, the \
                 mesh has {} cells",
                v.len(),
                m.n_cells
            )));
        }
        Ok(())
    }
}

fn count_label(n: usize) -> Label {
    // A colour block cannot be larger than the mesh, and the mesh has already
    // been proved to fit in a Label by the time any of this runs.
    n as Label
}

// ==========================================================================
//  Tests
// ==========================================================================

#[cfg(test)]
#[allow(clippy::unnecessary_cast)]
mod tests {
    use super::*;
    use crate::mesh::PatchKind;
    use crate::types::Vec3;

    fn box_mesh(n: [usize; 3]) -> HostMesh {
        let d = Vec3::new(1.0 / n[0] as Scalar, 1.0 / n[1] as Scalar, 1.0 / n[2] as Scalar);
        let (mut m, pts, faces) = crate::mesh::topology::tests::box_mesh(n, d);
        for p in m.patches.iter_mut() {
            p.kind = PatchKind::Generic;
            p.type_name = "patch".to_string();
        }
        m.build_cell_face_maps();
        m.compute_geometry(&pts, &faces).expect("geometry");
        m
    }

    #[test]
    fn greedy_gives_a_valid_colouring_and_two_colours_on_a_hex_mesh() {
        for n in [[6usize, 5, 4], [3, 3, 3], [10, 1, 1]] {
            let m = box_mesh(n);
            let g = Adjacency::of(&m);
            let c = Colouring::greedy(&g);
            assert!(c.is_valid(&g), "{n:?}: greedy produced a bad colouring");
            // SPEC-LIT 21: "a structured hex mesh needs 2".
            assert_eq!(c.n_colours, 2, "{n:?} took {} colours", c.n_colours);
            assert_eq!(c.cells.len(), m.n_cells);
            assert_eq!(c.offsets.len(), c.n_colours + 1);
            assert_eq!(c.offsets[c.n_colours], m.n_cells);
        }
    }

    #[test]
    fn every_cell_appears_exactly_once_in_the_colour_blocks() {
        let m = box_mesh([5, 4, 3]);
        let c = Colouring::greedy(&Adjacency::of(&m));
        let mut seen = vec![false; m.n_cells];
        for colour in 0..c.n_colours {
            for i in c.offsets[colour]..c.offsets[colour + 1] {
                let cell = c.cells[i] as usize;
                assert!(!seen[cell], "cell {cell} listed twice");
                assert_eq!(c.colour[cell], colour as Label);
                seen[cell] = true;
            }
        }
        assert!(seen.into_iter().all(|s| s));
    }

    #[test]
    fn reversing_and_shuffling_both_stay_valid_colourings() {
        let m = box_mesh([5, 4, 3]);
        let g = Adjacency::of(&m);
        let c = Colouring::greedy(&g);

        let rev = c.with_reversed_colour_order();
        assert!(rev.is_valid(&g));
        assert_eq!(rev.n_colours, c.n_colours);

        let shuf = c.with_shuffled_cells_within_colours(12345);
        assert!(shuf.is_valid(&g));
        assert_eq!(shuf.colour, c.colour);
        assert_eq!(shuf.offsets, c.offsets);
        // Same membership, different order.
        for colour in 0..c.n_colours {
            let (lo, hi) = (c.offsets[colour], c.offsets[colour + 1]);
            let mut a = c.cells[lo..hi].to_vec();
            let mut b = shuf.cells[lo..hi].to_vec();
            a.sort_unstable();
            b.sort_unstable();
            assert_eq!(a, b);
        }
    }

    /// An unstructured graph the greedy colouring cannot do in two: a ring of
    /// an odd number of cells. Chromatic number 3, and greedy in index order
    /// finds exactly that.
    #[test]
    fn an_odd_ring_needs_three_colours() {
        let n = 7usize;
        let mut m = HostMesh {
            n_cells: n,
            n_internal_faces: n,
            ..HostMesh::default()
        };
        for i in 0..n {
            let j = (i + 1) % n;
            m.owner.push(i.min(j) as Label);
            m.neighbour.push(i.max(j) as Label);
        }
        m.build_cell_face_maps();

        let g = Adjacency::of(&m);
        let c = Colouring::greedy(&g);
        assert!(c.is_valid(&g));
        assert_eq!(c.n_colours, 3);
    }
}
