// meteor-cfd - Copyright (c) 2026 주식회사 메테오시뮬레이션 (Meteo Simulation Co., Ltd.)
// Source-available, not Open Source. Educational use is free; research,
// publication and commercial use require a licence - simul@msimul.com
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
//! `boundary_coeffs` into `source` - except on coupled (cyclic) faces, where
//! `boundary_coeffs` stay in the matrix and multiply the neighbouring cell
//! inside `amul`.

use crate::device::{DevBuf, Gpu};
use crate::error::Result;
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
}

/// The host-side pattern, before upload. Built once from the mesh.
pub struct CsrPattern {
    pub n_rows: usize,
    pub nnz: usize,
    pub row_ptr: Vec<Label>,
    pub col_ind: Vec<Label>,
    pub diag_slot: Vec<Label>,
    pub upper_slot: Vec<Label>,
    pub lower_slot: Vec<Label>,
}

impl CsrPattern {
    /// Build the pattern from the addressing. Rows are cells; each row holds
    /// the diagonal plus one entry per incident internal face, with column
    /// indices sorted ascending.
    pub fn build(m: &HostMesh) -> Self {
        let n = m.n_cells;
        let mut row_ptr = vec![0 as Label; n + 1];

        // Row length = 1 (diagonal) + number of incident internal faces,
        // which the cell->face CSR already counts.
        for c in 0..n {
            let deg = (m.cf_offset[c + 1] - m.cf_offset[c]) as usize;
            row_ptr[c + 1] = row_ptr[c] + 1 + deg as Label;
        }
        let nnz = row_ptr[n] as usize;

        let mut col_ind = vec![0 as Label; nnz];
        let mut diag_slot = vec![0 as Label; n];
        let mut upper_slot = vec![0 as Label; m.n_internal_faces];
        let mut lower_slot = vec![0 as Label; m.n_internal_faces];

        // Columns for one row: the cell itself plus every neighbour across an
        // incident face. Collect, sort, then record where each LDU entry went.
        let mut scratch: Vec<(Label, Option<(usize, bool)>)> = Vec::with_capacity(16);

        for c in 0..n {
            scratch.clear();
            scratch.push((c as Label, None));

            for j in m.cf_offset[c] as usize..m.cf_offset[c + 1] as usize {
                let f = m.cf_face[j] as usize;
                let is_owner = m.cf_own[j] != 0;
                let other = if is_owner { m.neighbour[f] } else { m.owner[f] };
                scratch.push((other, Some((f, is_owner))));
            }

            scratch.sort_by_key(|&(col, _)| col);

            let base = row_ptr[c] as usize;
            for (k, &(col, origin)) in scratch.iter().enumerate() {
                let slot = (base + k) as Label;
                col_ind[base + k] = col;
                match origin {
                    None => diag_slot[c] = slot,
                    // A(owner, neighbour) is upper; A(neighbour, owner) is lower.
                    Some((f, true)) => upper_slot[f] = slot,
                    Some((f, false)) => lower_slot[f] = slot,
                }
            }
        }

        Self { n_rows: n, nnz, row_ptr, col_ind, diag_slot, upper_slot, lower_slot }
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
        })
    }
}
