// meteor-cfd - Copyright (c) 2026 주식회사 메테오시뮬레이션 (Meteo Simulation Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

//! The `fvScalarMatrix` equivalent, device resident.
//!
//! Provenance: original. Lower/diagonal/upper storage is what a face-based
//! mesh gives you; the CSR export for AMGX/cuDSS is this project's own design.
//! No GPL-licensed source was consulted.
//!
//! A face-based mesh gives a matrix with exactly one off-diagonal entry per
//! face per direction, so the natural storage is lower/diagonal/upper:
//!
//! ```text
//! diag  [nCells]           A(c, c)
//! upper [nInternalFaces]   A(owner[f], neighbour[f])
//! lower [nInternalFaces]   A(neighbour[f], owner[f])
//! source[nCells]           right-hand side
//! ```
//!
//! `upper[f]` is therefore the OWNER's row and `lower[f]` the NEIGHBOUR's,
//! which fixes the product:
//!
//! ```text
//! Apsi[neighbour[f]] += lower[f]*psi[owner[f]]
//! Apsi[owner[f]]     += upper[f]*psi[neighbour[f]]
//! ```
//!
//! Boundary faces carry `internal_coeffs` / `boundary_coeffs` exactly as
//! `fvMatrix` does. At solve time `internal_coeffs` fold into `diag` and
//! `boundary_coeffs` into `source` - except on coupled faces (cyclic, and
//! since SPEC-LIT S47 conjugate interfaces too), where `boundary_coeffs` stay
//! in the matrix and multiply the neighbouring cell inside `amul`.
//!
//! The CSR export carries those coupled entries as well - SPEC-LIT S48.2. It
//! did not before, which made the exported matrix a DIFFERENT operator from
//! the one `amul` applies on any mesh with a coupled patch, and is why the
//! AMGX backend refused every periodic mesh.

use crate::device::{DevBuf, Gpu};
use crate::error::{Error, Result};
use crate::mesh::{GpuMesh, HostMesh};
use crate::{Label, Scalar};

pub struct GpuLduMatrix {
    pub n_cells: usize,
    pub n_internal_faces: usize,
    pub n_boundary_faces: usize,

    pub diag: DevBuf<Scalar>,
    pub upper: DevBuf<Scalar>,
    pub lower: DevBuf<Scalar>,
    pub source: DevBuf<Scalar>,

    /// `[n_bf]`
    pub internal_coeffs: DevBuf<Scalar>,
    /// `[n_bf]`
    pub boundary_coeffs: DevBuf<Scalar>,

    /// Scratch for `set_values`: which cells are constrained and to what.
    ///
    /// Allocated with the matrix rather than lazily, because the time loop is
    /// not allowed to allocate. The C++ version used a shared file-local cache
    /// for this, which would have collided between two matrices on two
    /// streams; here it simply belongs to the matrix.
    pub is_fixed: DevBuf<Label>,
    pub fixed_value: DevBuf<Scalar>,
}

impl GpuLduMatrix {
    pub fn new(gpu: &Gpu, m: &GpuMesh) -> Result<Self> {
        Ok(Self {
            n_cells: m.n_cells,
            n_internal_faces: m.n_internal_faces,
            n_boundary_faces: m.n_boundary_faces,

            diag: gpu.zeros(m.n_cells)?,
            upper: gpu.zeros(m.n_internal_faces)?,
            lower: gpu.zeros(m.n_internal_faces)?,
            source: gpu.zeros(m.n_cells)?,
            internal_coeffs: gpu.zeros(m.n_boundary_faces)?,
            boundary_coeffs: gpu.zeros(m.n_boundary_faces)?,

            is_fixed: gpu.zeros(m.n_cells)?,
            fixed_value: gpu.zeros(m.n_cells)?,
        })
    }

    /// Zero every coefficient. Called at the top of each assembly, so it runs
    /// on the solver's stream rather than the legacy default one.
    pub fn zero(&mut self, gpu: &Gpu) -> Result<()> {
        gpu.fill_zero(&mut self.diag)?;
        gpu.fill_zero(&mut self.upper)?;
        gpu.fill_zero(&mut self.lower)?;
        gpu.fill_zero(&mut self.source)?;
        gpu.fill_zero(&mut self.internal_coeffs)?;
        gpu.fill_zero(&mut self.boundary_coeffs)?;
        gpu.fill_zero(&mut self.is_fixed)?;
        gpu.fill_zero(&mut self.fixed_value)?;
        Ok(())
    }
}

/// CSR export of the LDU structure.
///
/// The sparsity pattern is fixed for a static mesh, so the pattern and the
/// LDU-entry -> CSR-slot permutation are built once on the host. Refilling the
/// values each assembly is then a pure gather, which is what lets AMGX, cuDSS
/// or cuSPARSE consume the matrix without the host ever seeing it.
pub struct GpuCsrMatrix {
    pub n_rows: usize,
    pub nnz: usize,

    /// `[n_rows + 1]`
    pub row_ptr: DevBuf<Label>,
    /// `[nnz]`, ascending within each row - what cuSPARSE and AMGX expect
    pub col_ind: DevBuf<Label>,
    pub val: DevBuf<Scalar>,

    /// Where each LDU entry lands in `val`
    pub diag_slot: DevBuf<Label>,
    pub upper_slot: DevBuf<Label>,
    pub lower_slot: DevBuf<Label>,
    /// `[n_bf]` where each COUPLED boundary face's off-diagonal lands, `-1`
    /// on an uncoupled face - SPEC-LIT §48.2. Before §48 there was no such
    /// column, so an exported matrix silently omitted every cyclic (and
    /// therefore every conjugate) coupling.
    pub coupled_slot: DevBuf<Label>,
    pub n_boundary_faces: usize,
}

/// The host-side pattern, before upload. Built once from the mesh.
#[derive(Debug, Clone)]
pub struct CsrPattern {
    pub n_rows: usize,
    pub nnz: usize,
    pub row_ptr: Vec<Label>,
    pub col_ind: Vec<Label>,
    pub diag_slot: Vec<Label>,
    pub upper_slot: Vec<Label>,
    pub lower_slot: Vec<Label>,
    /// `[n_bf]`, `-1` on an uncoupled face - SPEC-LIT §48.2.
    pub coupled_slot: Vec<Label>,
    /// How many boundary faces carry a coupled column. Zero on an ordinary
    /// mesh, which is what keeps that case byte-for-byte what it was.
    pub n_coupled: usize,
}

/// Where one column of a row came from.
#[derive(Clone, Copy)]
enum ColSource {
    Diagonal,
    /// `(internal face, this cell owns it)`
    Face(usize, bool),
    /// A coupled boundary face - SPEC-LIT §48.2.
    Coupled(usize),
}

impl CsrPattern {
    /// Build the pattern from the addressing. Rows are cells; each row holds
    /// the diagonal, one entry per incident internal face, and - since
    /// SPEC-LIT §48.2 - **one entry per incident COUPLED boundary face**,
    /// with column indices sorted ascending.
    ///
    /// # What §48 changed, and why it had to
    ///
    /// A coupled face's `boundary_coeffs` is an off-diagonal against a cell
    /// that is not a face neighbour. Before §48 the pattern had no column for
    /// it, `lduCsrFill` never wrote it, and the exported CSR was therefore
    /// **not the operator [`crate::ldu_ops::amul`] applies** on any mesh with
    /// `b_nbr_cell >= 0`. `pressure::amgx` refused such meshes outright for
    /// exactly that reason, which meant AMGX was unavailable on every
    /// periodic mesh; §47's conjugate interface would have made it unavailable
    /// on every multi-region mesh too.
    ///
    /// On a mesh with no coupled face the result is byte-for-byte what it was:
    /// `n_coupled` is zero, every `coupled_slot` is `-1`, and `nnz` is still
    /// `n_cells + 2 n_internal_faces`.
    ///
    /// # The one thing that is refused
    ///
    /// Two cells joined by BOTH an internal face and a coupled boundary face
    /// would put the same column in a row twice - a two-cell periodic mesh
    /// does exactly that. `amul` sums both terms and a single CSR entry can
    /// hold only one, so merging them would be silently wrong and dropping
    /// one would be worse. It is an error naming the cell.
    pub fn build(m: &HostMesh) -> Result<Self> {
        let n = m.n_cells;
        let n_bf = m.n_boundary_faces;
        let mut row_ptr = vec![0 as Label; n + 1];

        let coupled = |bf: usize| -> Option<Label> {
            match m.b_nbr_cell.get(bf).copied() {
                Some(c) if c >= 0 && (c as usize) < n => Some(c),
                _ => None,
            }
        };

        // Row length = 1 (diagonal) + incident internal faces + incident
        // COUPLED boundary faces. Both counts come from CSRs the mesh already
        // built.
        let mut n_coupled = 0usize;
        for c in 0..n {
            let deg = (m.cf_offset[c + 1] - m.cf_offset[c]) as usize;
            let mut cpl = 0usize;
            for j in m.bcf_offset[c] as usize..m.bcf_offset[c + 1] as usize {
                let bf = m.bcf_face[j];
                if bf >= 0 && coupled(bf as usize).is_some() {
                    cpl += 1;
                }
            }
            n_coupled += cpl;
            row_ptr[c + 1] = row_ptr[c] + 1 + (deg + cpl) as Label;
        }
        let nnz = row_ptr[n] as usize;

        let mut col_ind = vec![0 as Label; nnz];
        let mut diag_slot = vec![0 as Label; n];
        let mut upper_slot = vec![0 as Label; m.n_internal_faces];
        let mut lower_slot = vec![0 as Label; m.n_internal_faces];
        let mut coupled_slot = vec![-1 as Label; n_bf];

        // Columns for one row: the cell itself, every neighbour across an
        // incident internal face, and every cell across an incident coupled
        // boundary face. Collect, sort, then record where each entry went.
        let mut scratch: Vec<(Label, ColSource)> = Vec::with_capacity(16);

        for c in 0..n {
            scratch.clear();
            scratch.push((c as Label, ColSource::Diagonal));

            for j in m.cf_offset[c] as usize..m.cf_offset[c + 1] as usize {
                let f = m.cf_face[j] as usize;
                let is_owner = m.cf_own[j] != 0;
                let other = if is_owner { m.neighbour[f] } else { m.owner[f] };
                scratch.push((other, ColSource::Face(f, is_owner)));
            }

            for j in m.bcf_offset[c] as usize..m.bcf_offset[c + 1] as usize {
                let bf = m.bcf_face[j];
                if bf < 0 {
                    continue;
                }
                let bf = bf as usize;
                if let Some(other) = coupled(bf) {
                    scratch.push((other, ColSource::Coupled(bf)));
                }
            }

            scratch.sort_by_key(|&(col, _)| col);

            for w in scratch.windows(2) {
                if w[0].0 == w[1].0 {
                    return Err(Error::Mesh(format!(
                        "CsrPattern::build: cell {c} reaches cell {} through two \
                         different couplings (an internal face and a coupled \
                         boundary face, or two coupled boundary faces). The CSR \
                         has one entry per (row, column) and `amul` SUMS both \
                         terms, so exporting this matrix would silently drop one \
                         of them. Mesh the periodic pair with more than one cell \
                         across, or solve it with the LDU path",
                        w[0].0
                    )));
                }
            }

            let base = row_ptr[c] as usize;
            for (k, &(col, origin)) in scratch.iter().enumerate() {
                let slot = (base + k) as Label;
                col_ind[base + k] = col;
                match origin {
                    ColSource::Diagonal => diag_slot[c] = slot,
                    // A(owner, neighbour) is upper; A(neighbour, owner) is lower.
                    ColSource::Face(f, true) => upper_slot[f] = slot,
                    ColSource::Face(f, false) => lower_slot[f] = slot,
                    ColSource::Coupled(bf) => coupled_slot[bf] = slot,
                }
            }
        }

        Ok(Self {
            n_rows: n,
            nnz,
            row_ptr,
            col_ind,
            diag_slot,
            upper_slot,
            lower_slot,
            coupled_slot,
            n_coupled,
        })
    }

    pub fn upload(&self, gpu: &Gpu) -> Result<GpuCsrMatrix> {
        Ok(GpuCsrMatrix {
            n_rows: self.n_rows,
            nnz: self.nnz,
            row_ptr: gpu.upload(&self.row_ptr)?,
            col_ind: gpu.upload(&self.col_ind)?,
            val: gpu.zeros(self.nnz)?,
            diag_slot: gpu.upload(&self.diag_slot)?,
            upper_slot: gpu.upload(&self.upper_slot)?,
            lower_slot: gpu.upload(&self.lower_slot)?,
            coupled_slot: gpu.upload(&self.coupled_slot)?,
            n_boundary_faces: self.coupled_slot.len(),
        })
    }
}
