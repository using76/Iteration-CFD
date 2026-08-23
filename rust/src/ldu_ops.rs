// meteor-cfd - Copyright (c) 2026 주식회사 메테오시뮬레이션 (Meteo Simulation Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

//! Operations on the LDU system: the launchers for `cuda/ldu.cu`.
//!
//! Written from:
//!   Jasak, *Error Analysis and Estimation for the Finite Volume Method*,
//!     PhD thesis, Imperial College (1996), ch. 3
//!   Patankar, *Numerical Heat Transfer and Fluid Flow* (1980), 4.2-4.9
//!   Moukalled, Mangani & Darwish, *The Finite Volume Method in CFD* (2016),
//!     ch. 8
//!   Saad, *Iterative Methods for Sparse Linear Systems*, 2nd ed. (2003), 3.4
//!   ofgpu `SPEC-LIT.md` sections 1, 3, 4 and 5.2
//! No GPL-licensed source was consulted.
//!
//! # What lives where
//!
//! The arithmetic is in `cuda/ldu.cu`, which also carries the derivation of
//! the coupled-face sign convention and the reason every kernel is one thread
//! per cell. This file is the host half: it resolves the entry points once,
//! checks that the matrix and the mesh describe the same problem, and skips
//! the launch when there is nothing to launch (a zero-block grid is an illegal
//! configuration, not a no-op).
//!
//! # The order the assembly calls these in
//!
//! ```text
//! a.zero()
//! fv::fvm_*                  operators write diag/upper/lower/source and the
//!                            per-boundary-face pair
//! relax(alpha)               optional; steady state only. BEFORE the fold
//! set_values()               optional; wall-function cells, pressure reference
//! add_boundary_contributions()
//! solve, using amul()
//! ```
//!
//! Two of those positions are load-bearing.
//!
//! [`relax`] must come BEFORE [`add_boundary_contributions`]. It tests
//! dominance against the diagonal the solver will end up with, which means it
//! adds the not-yet-folded `internal_coeffs` itself; run after the fold it
//! would count them twice.
//!
//! [`amul`] must come AFTER it. The fold is what puts the boundary
//! contributions where `amul` expects them, and the coupled coefficient - the
//! one thing the fold deliberately leaves behind - is the only boundary term
//! `amul` applies itself.
//!
//! [`set_values`] is indifferent: it overwrites the rows it pins and zeroes
//! the boundary pair on them, so folding before or after adds nothing to a row
//! that is already final.

use cudarc::driver::{CudaFunction, PushKernelArg};

use crate::device::{cfg_for, DevBuf, Gpu, KernelSet};
use crate::error::{Error, Result};
use crate::ldu::{GpuCsrMatrix, GpuLduMatrix};
use crate::mesh::GpuMesh;
use crate::{Label, Scalar};

/// Entry points of `cuda/ldu.cu`, resolved once so the time loop never does a
/// string lookup.
pub struct LduKernels {
    neg_sum_diag: CudaFunction,
    add_boundary: CudaFunction,
    amul: CudaFunction,
    relax: CudaFunction,
    set_values: CudaFunction,
    csr_fill: CudaFunction,
}

impl LduKernels {
    pub fn new(gpu: &Gpu) -> Result<Self> {
        let k = KernelSet::new(gpu, crate::kernels::LDU)?;
        Ok(Self {
            neg_sum_diag: k.func("lduNegSumDiag")?,
            add_boundary: k.func("lduAddBoundaryContributions")?,
            amul: k.func("lduAmul")?,
            relax: k.func("lduRelax")?,
            set_values: k.func("lduSetValues")?,
            csr_fill: k.func("lduCsrFill")?,
        })
    }
}

/// A matrix and a mesh have to agree about how many cells and faces there are,
/// or every kernel below indexes past the end of something.
fn check_shape(a: &GpuLduMatrix, m: &GpuMesh, what: &str) -> Result<()> {
    if a.n_cells != m.n_cells
        || a.n_internal_faces != m.n_internal_faces
        || a.n_boundary_faces != m.n_boundary_faces
    {
        return Err(Error::Config(format!(
            "{what}: matrix is {}/{}/{} cells/faces/boundary faces but the \
             mesh is {}/{}/{}",
            a.n_cells,
            a.n_internal_faces,
            a.n_boundary_faces,
            m.n_cells,
            m.n_internal_faces,
            m.n_boundary_faces
        )));
    }
    Ok(())
}

// ==========================================================================
//  negSumDiag
// ==========================================================================

/// `diag[c] -= sum of column c's off-diagonal entries`.
///
/// The diagonal an operator needs if a uniform field is to be in its null
/// space: convection (SPEC-LIT 3.1) and diffusion (SPEC-LIT 3.2) both write
/// their off-diagonals first and then take this. See `cuda/ldu.cu` for why it
/// is the column sum and not the row sum.
pub fn neg_sum_diag(gpu: &Gpu, k: &LduKernels, a: &mut GpuLduMatrix, m: &GpuMesh) -> Result<()> {
    check_shape(a, m, "neg_sum_diag")?;

    let n = a.n_cells;
    if n == 0 {
        return Ok(());
    }
    let nl = n as Label;

    let f = k.neg_sum_diag.clone();
    unsafe {
        gpu.stream()
            .launch_builder(&f)
            .arg(&mut a.diag)
            .arg(&a.upper)
            .arg(&a.lower)
            .arg(&m.cf_offset)
            .arg(&m.cf_face)
            .arg(&m.cf_own)
            .arg(&nl)
            .launch(cfg_for(n))?;
    }
    Ok(())
}

// ==========================================================================
//  addBoundaryContributions
// ==========================================================================

/// Fold `internal_coeffs` into `diag` and `boundary_coeffs` into `source`.
///
/// A coupled (cyclic) face keeps its boundary coefficient in the matrix
/// instead, because the "known" value it multiplies is the live value in the
/// cell across the couple; [`amul`] applies it. Its internal coefficient still
/// folds, since that multiplies this cell's own value.
pub fn add_boundary_contributions(
    gpu: &Gpu,
    k: &LduKernels,
    a: &mut GpuLduMatrix,
    m: &GpuMesh,
) -> Result<()> {
    check_shape(a, m, "add_boundary_contributions")?;

    let n = a.n_cells;
    if n == 0 {
        return Ok(());
    }
    let nl = n as Label;

    let f = k.add_boundary.clone();
    unsafe {
        gpu.stream()
            .launch_builder(&f)
            .arg(&mut a.diag)
            .arg(&mut a.source)
            .arg(&a.internal_coeffs)
            .arg(&a.boundary_coeffs)
            .arg(&m.b_nbr_cell)
            .arg(&m.bcf_offset)
            .arg(&m.bcf_face)
            .arg(&nl)
            .launch(cfg_for(n))?;
    }
    Ok(())
}

// ==========================================================================
//  Amul
// ==========================================================================

/// `apsi = A.psi`, including the coupled-interface term.
///
/// Expects the boundary pair to have been folded already
/// ([`add_boundary_contributions`]); the coupled coefficient is the only
/// boundary term this applies.
pub fn amul(
    gpu: &Gpu,
    k: &LduKernels,
    apsi: &mut DevBuf<Scalar>,
    psi: &DevBuf<Scalar>,
    a: &GpuLduMatrix,
    m: &GpuMesh,
) -> Result<()> {
    check_shape(a, m, "amul")?;

    let n = a.n_cells;
    if apsi.len() < n || psi.len() < n {
        return Err(Error::Config(format!(
            "amul: psi has {} and A.psi has {} elements, but the matrix has {} \
             rows",
            psi.len(),
            apsi.len(),
            n
        )));
    }
    if n == 0 {
        return Ok(());
    }
    let nl = n as Label;

    let f = k.amul.clone();
    unsafe {
        gpu.stream()
            .launch_builder(&f)
            .arg(&mut *apsi)
            .arg(psi)
            .arg(&a.diag)
            .arg(&a.upper)
            .arg(&a.lower)
            .arg(&a.boundary_coeffs)
            .arg(&m.owner)
            .arg(&m.neighbour)
            .arg(&m.b_nbr_cell)
            .arg(&m.cf_offset)
            .arg(&m.cf_face)
            .arg(&m.cf_own)
            .arg(&m.bcf_offset)
            .arg(&m.bcf_face)
            .arg(&nl)
            .launch(cfg_for(n))?;
    }
    Ok(())
}

// ==========================================================================
//  relax
// ==========================================================================

/// Implicit under-relaxation by `alpha` (SPEC-LIT 5.2; Patankar 1980, 4.9).
///
/// ```text
/// diag' = max(diag, sum|off-diagonal|) / alpha
/// b'    = b + (diag' - diag)*psi
/// ```
///
/// `psi` is the CURRENT value of the field being solved for - the relaxation
/// leaves the fixed point alone precisely because the same `psi` appears on
/// both sides once it stops changing.
///
/// One per-cell kernel and no scratch array: the unrelaxed diagonal is needed
/// twice and a cell has everything it needs to compute it, so it stays in a
/// register. `cuda/ldu.cu` documents the three implementation decisions
/// (folded diagonal, coupled faces counted on both sides, sign preserved).
///
/// **Call this before [`add_boundary_contributions`].** `diag` on its own is
/// not yet the diagonal the solver will see, so this adds `internal_coeffs`
/// while testing dominance and then writes an increment that the fold
/// completes. Called after the fold it would count those coefficients twice
/// and over-relax every boundary cell.
///
/// `alpha` must be positive and finite; `alpha = 1` still enforces diagonal
/// dominance, which is the half of the rule that is about stability rather
/// than about step size.
pub fn relax(
    gpu: &Gpu,
    k: &LduKernels,
    a: &mut GpuLduMatrix,
    m: &GpuMesh,
    psi: &DevBuf<Scalar>,
    alpha: Scalar,
) -> Result<()> {
    check_shape(a, m, "relax")?;

    if !(alpha > 0.0) || !alpha.is_finite() {
        return Err(Error::Config(format!(
            "relax: the under-relaxation factor is {alpha}; it must be finite \
             and positive (the diagonal is divided by it)"
        )));
    }

    let n = a.n_cells;
    if psi.len() < n {
        return Err(Error::Config(format!(
            "relax: psi has {} elements, but the matrix has {} rows",
            psi.len(),
            n
        )));
    }
    if n == 0 {
        return Ok(());
    }
    let nl = n as Label;

    let f = k.relax.clone();
    unsafe {
        gpu.stream()
            .launch_builder(&f)
            .arg(&mut a.diag)
            .arg(&mut a.source)
            .arg(&a.upper)
            .arg(&a.lower)
            .arg(&a.internal_coeffs)
            .arg(&a.boundary_coeffs)
            .arg(psi)
            .arg(&m.b_nbr_cell)
            .arg(&m.cf_offset)
            .arg(&m.cf_face)
            .arg(&m.cf_own)
            .arg(&m.bcf_offset)
            .arg(&m.bcf_face)
            .arg(&alpha)
            .arg(&nl)
            .launch(cfg_for(n))?;
    }
    Ok(())
}

// ==========================================================================
//  setValues
// ==========================================================================

/// Pin every cell flagged in `a.is_fixed` to the matching `a.fixed_value`.
///
/// The row becomes `diag*psi = diag*value` and nothing else, and the
/// corresponding COLUMN is eliminated into the neighbours' sources so that a
/// symmetric matrix stays symmetric - see `cuda/ldu.cu`, where that choice is
/// derived and marked as ours.
///
/// The flags live on the matrix ([`GpuLduMatrix::is_fixed`],
/// [`GpuLduMatrix::fixed_value`]) rather than in a shared cache, so two
/// matrices being assembled for two different equations cannot collide.
/// `a.zero()` clears them, so they are written after it: from a kernel (a wall
/// function marking the cells it owns) or, at setup, from
/// [`set_fixed_cells`].
pub fn set_values(gpu: &Gpu, k: &LduKernels, a: &mut GpuLduMatrix, m: &GpuMesh) -> Result<()> {
    check_shape(a, m, "set_values")?;

    let n = a.n_cells;
    if n == 0 {
        return Ok(());
    }
    let nl = n as Label;

    let f = k.set_values.clone();
    unsafe {
        gpu.stream()
            .launch_builder(&f)
            .arg(&mut a.diag)
            .arg(&mut a.source)
            .arg(&mut a.upper)
            .arg(&mut a.lower)
            .arg(&mut a.internal_coeffs)
            .arg(&mut a.boundary_coeffs)
            .arg(&a.is_fixed)
            .arg(&a.fixed_value)
            .arg(&m.owner)
            .arg(&m.neighbour)
            .arg(&m.b_nbr_cell)
            .arg(&m.cf_offset)
            .arg(&m.cf_face)
            .arg(&m.cf_own)
            .arg(&m.bcf_offset)
            .arg(&m.bcf_face)
            .arg(&nl)
            .launch(cfg_for(n))?;
    }
    Ok(())
}

/// Fill the constraint scratch from a host list of `(cell, value)`.
///
/// Setup and testing only - it writes two whole `nCells` arrays from the host,
/// which nothing in the time loop is allowed to do. A wall function marks its
/// cells from a kernel instead.
pub fn set_fixed_cells(
    gpu: &Gpu,
    a: &mut GpuLduMatrix,
    cells: &[Label],
    values: &[Scalar],
) -> Result<()> {
    if cells.len() != values.len() {
        return Err(Error::Config(format!(
            "set_fixed_cells: {} cells but {} values",
            cells.len(),
            values.len()
        )));
    }

    let mut flags = vec![0 as Label; a.n_cells];
    let mut vals = vec![0.0 as Scalar; a.n_cells];

    for (&c, &v) in cells.iter().zip(values) {
        if c < 0 || c as usize >= a.n_cells {
            return Err(Error::Config(format!(
                "set_fixed_cells: cell {c} is outside [0, {})",
                a.n_cells
            )));
        }
        flags[c as usize] = 1;
        vals[c as usize] = v;
    }

    gpu.write(&mut a.is_fixed, &flags)?;
    gpu.write(&mut a.fixed_value, &vals)?;
    Ok(())
}

// ==========================================================================
//  csrFill
// ==========================================================================

/// Gather `diag`/`upper`/`lower` into the values of a prebuilt CSR.
///
/// The pattern and the LDU-entry to CSR-slot permutation come from
/// [`crate::ldu::CsrPattern::build`] and never change for a static mesh, so
/// this is a pure permutation write and the matrix reaches AMGX (or cuSPARSE,
/// or cuDSS) without the host seeing a coefficient.
///
/// The CSR has one column per incident internal face plus the diagonal, so a
/// coupled face's `boundary_coeffs` - an off-diagonal against a cell that is
/// not a face neighbour - has nowhere to go. On a mesh with cyclic patches the
/// exported matrix is therefore NOT the operator [`amul`] applies; the AMGX
/// backend refuses such a mesh up front for exactly this reason.
pub fn csr_fill(
    gpu: &Gpu,
    k: &LduKernels,
    csr: &mut GpuCsrMatrix,
    a: &GpuLduMatrix,
) -> Result<()> {
    if csr.n_rows != a.n_cells {
        return Err(Error::Config(format!(
            "csr_fill: the CSR has {} rows but the matrix has {}",
            csr.n_rows, a.n_cells
        )));
    }

    let expected = a.n_cells + 2 * a.n_internal_faces;
    if csr.nnz != expected {
        return Err(Error::Config(format!(
            "csr_fill: the CSR holds {} non-zeros; an LDU matrix with {} cells \
             and {} internal faces needs exactly {}",
            csr.nnz, a.n_cells, a.n_internal_faces, expected
        )));
    }

    let n = a.n_cells.max(a.n_internal_faces);
    if n == 0 {
        return Ok(());
    }
    let (ncl, nfl) = (a.n_cells as Label, a.n_internal_faces as Label);

    let f = k.csr_fill.clone();
    unsafe {
        gpu.stream()
            .launch_builder(&f)
            .arg(&mut csr.val)
            .arg(&a.diag)
            .arg(&a.upper)
            .arg(&a.lower)
            .arg(&csr.diag_slot)
            .arg(&csr.upper_slot)
            .arg(&csr.lower_slot)
            .arg(&ncl)
            .arg(&nfl)
            .launch(cfg_for(n))?;
    }
    Ok(())
}

// ==========================================================================
//  Tests
// ==========================================================================

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    use crate::ldu::CsrPattern;
    use crate::mesh::{HostMesh, PatchInfo, PatchKind};
    use crate::Vec3;

    // ----------------------------------------------------------------------
    //  A mesh small enough to write the matrix out by hand
    // ----------------------------------------------------------------------

    /// Four cells in a line, three internal faces, four boundary faces - two
    /// plain and one cyclic pair joining cell 0 to cell 3:
    ///
    /// ```text
    ///        bf2 (cyclic, -> cell 3)      bf3 (cyclic, -> cell 0)
    ///          |                            |
    ///   bf0 -- 0 --f0-- 1 --f1-- 2 --f2-- 3 -- bf1
    /// ```
    ///
    /// Written out here rather than taken from `blockgen` or from the geometry
    /// sweep so that a bug in either cannot make these tests agree with a
    /// broken mesh, and so that cells 1 and 2 exercise the no-boundary-face
    /// path while 0 and 3 exercise two boundary faces each, one of them
    /// coupled.
    ///
    /// The geometry is filled in with plausible, DISTINCT values: nothing
    /// below reads it except the boundary delta coefficients and weights, and
    /// distinct values are what make a wrong index visible.
    pub(crate) fn chain_mesh() -> HostMesh {
        let dx: Scalar = 0.25;
        let area: Scalar = dx * dx;

        let mut m = HostMesh {
            n_cells: 4,
            n_internal_faces: 3,
            n_boundary_faces: 4,
            n_points: 0,

            owner: vec![0, 1, 2],
            neighbour: vec![1, 2, 3],

            v: vec![dx * dx * dx; 4],
            c: (0..4)
                .map(|i| Vec3::new((i as Scalar + 0.5) * dx, 0.5 * dx, 0.5 * dx))
                .collect(),

            sf: vec![Vec3::new(area, 0.0, 0.0); 3],
            mag_sf: vec![area; 3],
            cf: (1..4)
                .map(|i| Vec3::new(i as Scalar * dx, 0.5 * dx, 0.5 * dx))
                .collect(),
            weights: vec![0.5; 3],
            delta_coeffs: vec![1.0 / dx; 3],
            non_orth_corr: vec![Vec3::ZERO; 3],

            // bf0 on cell 0 (-x), bf1 on cell 3 (+x), then the cyclic pair
            // bf2 on cell 0 (-y) and bf3 on cell 3 (+y).
            b_face_cells: vec![0, 3, 0, 3],
            b_sf: vec![
                Vec3::new(-area, 0.0, 0.0),
                Vec3::new(area, 0.0, 0.0),
                Vec3::new(0.0, -area, 0.0),
                Vec3::new(0.0, area, 0.0),
            ],
            b_mag_sf: vec![area; 4],
            b_cf: vec![
                Vec3::new(0.0, 0.5 * dx, 0.5 * dx),
                Vec3::new(4.0 * dx, 0.5 * dx, 0.5 * dx),
                Vec3::new(0.5 * dx, 0.0, 0.5 * dx),
                Vec3::new(3.5 * dx, dx, 0.5 * dx),
            ],
            // Deliberately all different, so an off-by-one shows up.
            b_delta_coeffs: vec![8.0, 4.0, 2.0, 16.0],
            b_y: vec![0.5 * dx; 4],
            b_nbr_cell: vec![-1, -1, 3, 0],
            b_weights: vec![1.0, 1.0, 0.35, 0.65],
            b_kind: vec![
                PatchKind::Generic as Label,
                PatchKind::Generic as Label,
                PatchKind::Cyclic as Label,
                PatchKind::Cyclic as Label,
            ],
            b_patch: vec![0, 1, 2, 3],

            ..Default::default()
        };

        let names = ["left", "right", "cyc0", "cyc1"];
        let kinds = [
            PatchKind::Generic,
            PatchKind::Generic,
            PatchKind::Cyclic,
            PatchKind::Cyclic,
        ];
        let nbr = [None, None, Some(3usize), Some(2usize)];

        m.patches = (0..4)
            .map(|p| PatchInfo {
                name: names[p].to_string(),
                type_name: if p >= 2 { "cyclic" } else { "patch" }.to_string(),
                kind: kinds[p],
                start: p,
                size: 1,
                nbr_patch: nbr[p],
            })
            .collect();

        m.build_cell_face_maps();
        m
    }

    /// The device, the mesh on it, and the kernels - or `None` when there is
    /// no device, so the suite still passes on a machine without one.
    ///
    /// Only `Gpu::new` is allowed to fail quietly. A kernel that will not load
    /// is a real failure and panics, rather than silently skipping the test
    /// that was supposed to catch it.
    fn ctx() -> Option<(Gpu, HostMesh, GpuMesh, LduKernels)> {
        let gpu = Gpu::new(0).ok()?;
        let hm = chain_mesh();
        let m = GpuMesh::upload(&gpu, &hm).expect("upload the mesh");
        let k = LduKernels::new(&gpu).expect("load cuda/ldu.cu");
        Some((gpu, hm, m, k))
    }

    // ----------------------------------------------------------------------
    //  Coefficients, chosen by hand
    // ----------------------------------------------------------------------

    const DIAG: [Scalar; 4] = [2.0, 3.0, 4.0, 5.0];
    /// `A(0,1)`, `A(1,2)`, `A(2,3)`
    const UPPER: [Scalar; 3] = [0.5, -1.0, 2.0];
    /// `A(1,0)`, `A(2,1)`, `A(3,2)`
    const LOWER: [Scalar; 3] = [1.5, 0.25, -0.75];
    const SOURCE: [Scalar; 4] = [10.0, 20.0, 30.0, 40.0];
    /// bf0 (cell 0), bf1 (cell 3), bf2 (cell 0, cyclic), bf3 (cell 3, cyclic)
    const IC: [Scalar; 4] = [0.3, 0.7, 1.25, 1.25];
    const BC: [Scalar; 4] = [-0.9, 1.1, 0.4, -0.6];

    const PSI: [Scalar; 4] = [1.1, -2.3, 0.7, 4.2];

    fn fill(gpu: &Gpu, a: &mut GpuLduMatrix) {
        gpu.write(&mut a.diag, &DIAG).expect("diag");
        gpu.write(&mut a.upper, &UPPER).expect("upper");
        gpu.write(&mut a.lower, &LOWER).expect("lower");
        gpu.write(&mut a.source, &SOURCE).expect("source");
        gpu.write(&mut a.internal_coeffs, &IC).expect("internalCoeffs");
        gpu.write(&mut a.boundary_coeffs, &BC).expect("boundaryCoeffs");
    }

    fn max_diff(a: &[Scalar], b: &[Scalar]) -> Scalar {
        assert_eq!(a.len(), b.len());
        a.iter()
            .zip(b)
            .fold(0.0 as Scalar, |m, (x, y)| m.max((x - y).abs()))
    }

    /// Two 4x4s agree entry by entry. A tolerance rather than `assert_eq!`
    /// because the literal `3.55` in the test below and the `2.0 + 0.3 + 1.25`
    /// the device computes are different roundings of the same number.
    fn assert_dense_close(got: &[[Scalar; 4]; 4], want: &[[Scalar; 4]; 4]) {
        for i in 0..4 {
            for j in 0..4 {
                assert!(
                    (got[i][j] - want[i][j]).abs() < 1e-13,
                    "A({i},{j}) = {} not {}",
                    got[i][j],
                    want[i][j]
                );
            }
        }
    }

    /// `A.psi` for a dense 4x4, so the expected values in these tests are
    /// arithmetic on a matrix written out in full rather than a second
    /// implementation of the thing being tested.
    fn dense_mul(a: &[[Scalar; 4]; 4], psi: &[Scalar; 4]) -> Vec<Scalar> {
        (0..4)
            .map(|i| (0..4).map(|j| a[i][j] * psi[j]).sum())
            .collect()
    }

    /// Gaussian elimination with partial pivoting, for checking that a
    /// constrained system really does return the value it was constrained to.
    fn dense_solve(a0: &[[Scalar; 4]; 4], b0: &[Scalar; 4]) -> [Scalar; 4] {
        let mut a = *a0;
        let mut b = *b0;

        for col in 0..4 {
            let mut piv = col;
            for r in col + 1..4 {
                if a[r][col].abs() > a[piv][col].abs() {
                    piv = r;
                }
            }
            a.swap(col, piv);
            b.swap(col, piv);

            let d = a[col][col];
            assert!(d.abs() > 1e-300, "singular test matrix at column {col}");

            for r in col + 1..4 {
                let fac = a[r][col] / d;
                for c in col..4 {
                    a[r][c] -= fac * a[col][c];
                }
                b[r] -= fac * b[col];
            }
        }

        let mut x = [0.0 as Scalar; 4];
        for i in (0..4).rev() {
            let mut s = b[i];
            for j in i + 1..4 {
                s -= a[i][j] * x[j];
            }
            x[i] = s / a[i][i];
        }
        x
    }

    /// The dense form of whatever the device currently holds.
    ///
    /// The coupled entry is `-boundary_coeffs`, the sign derived in the header
    /// of `cuda/ldu.cu`: an explicit `b += bc*psi_N` becomes an implicit
    /// `A.psi -= bc*psi_N`.
    fn dense_from_device(gpu: &Gpu, a: &GpuLduMatrix, hm: &HostMesh) -> [[Scalar; 4]; 4] {
        let diag = gpu.download(&a.diag).expect("diag");
        let upper = gpu.download(&a.upper).expect("upper");
        let lower = gpu.download(&a.lower).expect("lower");
        let bc = gpu.download(&a.boundary_coeffs).expect("boundaryCoeffs");

        let mut d = [[0.0 as Scalar; 4]; 4];
        for c in 0..4 {
            d[c][c] = diag[c];
        }
        for f in 0..hm.n_internal_faces {
            let (o, n) = (hm.owner[f] as usize, hm.neighbour[f] as usize);
            d[o][n] += upper[f];
            d[n][o] += lower[f];
        }
        for bf in 0..hm.n_boundary_faces {
            let nbr = hm.b_nbr_cell[bf];
            if nbr >= 0 {
                d[hm.b_face_cells[bf] as usize][nbr as usize] -= bc[bf];
            }
        }
        d
    }

    // ----------------------------------------------------------------------
    //  negSumDiag
    // ----------------------------------------------------------------------

    #[test]
    fn neg_sum_diag_subtracts_the_column_sum() {
        let Some((gpu, _hm, m, k)) = ctx() else { return };

        let mut a = GpuLduMatrix::new(&gpu, &m).expect("matrix");
        a.zero(&gpu).expect("zero");
        fill(&gpu, &mut a);

        neg_sum_diag(&gpu, &k, &mut a, &m).expect("negSumDiag");
        gpu.sync().expect("sync");

        // Column c holds LOWER[f] where c owns f, and UPPER[f] where c
        // neighbours it - not the other way round, which is the whole point.
        let expect = [
            DIAG[0] - LOWER[0],
            DIAG[1] - (UPPER[0] + LOWER[1]),
            DIAG[2] - (UPPER[1] + LOWER[2]),
            DIAG[3] - UPPER[2],
        ];

        let got = gpu.download(&a.diag).expect("diag");
        assert!(max_diff(&got, &expect) < 1e-15, "{got:?} vs {expect:?}");
    }

    /// The property negSumDiag exists for: with `upper == lower == gamma`, a
    /// uniform field is in the operator's null space. Round the cyclic couple
    /// too, where the coupled pair (`ic = bc`) has to cancel for the same
    /// reason - which is a check on the coupled sign convention as much as on
    /// negSumDiag.
    #[test]
    fn a_uniform_field_is_in_the_null_space_of_a_laplacian() {
        let Some((gpu, hm, m, k)) = ctx() else { return };

        let gamma: [Scalar; 3] = [1.7, 0.9, 3.1];
        let gamma_b: Scalar = 2.3;

        let mut a = GpuLduMatrix::new(&gpu, &m).expect("matrix");
        a.zero(&gpu).expect("zero");
        gpu.write(&mut a.upper, &gamma).expect("upper");
        gpu.write(&mut a.lower, &gamma).expect("lower");

        // Only the cyclic pair carries a boundary coefficient. A coupled face
        // is an internal face in disguise: it contributes -gamma_b to the
        // diagonal (through internal_coeffs) and +gamma_b against the cell
        // across the couple - and amul negates boundary_coeffs to get there,
        // so the two are written EQUAL. That equality is exactly the condition
        // for the row to sum to zero, which is what is checked below.
        let ic = [0.0, 0.0, -gamma_b, -gamma_b];
        let bc = [0.0, 0.0, -gamma_b, -gamma_b];
        gpu.write(&mut a.internal_coeffs, &ic).expect("ic");
        gpu.write(&mut a.boundary_coeffs, &bc).expect("bc");

        neg_sum_diag(&gpu, &k, &mut a, &m).expect("negSumDiag");
        add_boundary_contributions(&gpu, &k, &mut a, &m).expect("fold");

        let one = vec![1.0 as Scalar; 4];
        let d_one = gpu.upload(&one).expect("upload");
        let mut ap: DevBuf<Scalar> = gpu.zeros(4).expect("alloc");
        amul(&gpu, &k, &mut ap, &d_one, &a, &m).expect("amul");
        gpu.sync().expect("sync");

        let got = gpu.download(&ap).expect("Apsi");
        assert!(
            max_diff(&got, &vec![0.0; 4]) < 1e-14,
            "a uniform field should give zero, got {got:?}"
        );

        // And the dense form really is singular against the constant vector,
        // i.e. the row sums vanish - the same statement, read off the matrix.
        let d = dense_from_device(&gpu, &a, &hm);
        for (i, row) in d.iter().enumerate() {
            let s: Scalar = row.iter().sum();
            assert!(s.abs() < 1e-14, "row {i} sums to {s}");
        }
    }

    // ----------------------------------------------------------------------
    //  addBoundaryContributions + Amul
    // ----------------------------------------------------------------------

    #[test]
    fn amul_matches_a_dense_matrix_written_out_by_hand() {
        let Some((gpu, hm, m, k)) = ctx() else { return };

        let mut a = GpuLduMatrix::new(&gpu, &m).expect("matrix");
        a.zero(&gpu).expect("zero");
        fill(&gpu, &mut a);

        add_boundary_contributions(&gpu, &k, &mut a, &m).expect("fold");
        gpu.sync().expect("sync");

        // diag picks up EVERY internal coefficient, coupled included;
        // source picks up only the uncoupled boundary coefficients.
        let expect_diag = [
            DIAG[0] + IC[0] + IC[2],
            DIAG[1],
            DIAG[2],
            DIAG[3] + IC[1] + IC[3],
        ];
        let expect_source = [SOURCE[0] + BC[0], SOURCE[1], SOURCE[2], SOURCE[3] + BC[1]];

        assert!(max_diff(&gpu.download(&a.diag).expect("diag"), &expect_diag) < 1e-15);
        assert!(max_diff(&gpu.download(&a.source).expect("source"), &expect_source) < 1e-15);

        // The 4x4, written out. Rows are cells; the (0,3) and (3,0) entries
        // are the cyclic couple, which is -boundary_coeffs.
        let dense: [[Scalar; 4]; 4] = [
            [3.55, 0.5, 0.0, -0.4],
            [1.5, 3.0, -1.0, 0.0],
            [0.0, 0.25, 4.0, 2.0],
            [0.6, 0.0, -0.75, 6.95],
        ];
        // Guard the literal against a typo in the constants above.
        assert_dense_close(&dense, &dense_from_device(&gpu, &a, &hm));

        let d_psi = gpu.upload(&PSI.to_vec()).expect("psi");
        let mut ap: DevBuf<Scalar> = gpu.zeros(4).expect("alloc");
        amul(&gpu, &k, &mut ap, &d_psi, &a, &m).expect("amul");
        gpu.sync().expect("sync");

        let got = gpu.download(&ap).expect("Apsi");
        let want = dense_mul(&dense, &PSI);

        assert!(max_diff(&got, &want) < 1e-13, "{got:?} vs {want:?}");

        // The arithmetic, spelled out once, so the dense literal above is not
        // simply believed either.
        assert!((got[0] - (3.55 * 1.1 + 0.5 * -2.3 - 0.4 * 4.2)).abs() < 1e-13);
    }

    // ----------------------------------------------------------------------
    //  relax
    // ----------------------------------------------------------------------

    /// `max(|diag|, sum|off|)/alpha` for each row of the test matrix, computed
    /// from the coefficients rather than from the kernel.
    fn expected_relaxed_diag(alpha: Scalar, diag: &[Scalar; 4]) -> [Scalar; 4] {
        // Row 0: internal face 0 (owner) -> UPPER[0]; coupled bf2 -> BC[2].
        // Row 3: internal face 2 (neighbour) -> LOWER[2]; coupled bf3 -> BC[3].
        let sum_off = [
            UPPER[0].abs() + BC[2].abs(),
            LOWER[0].abs() + UPPER[1].abs(),
            LOWER[1].abs() + UPPER[2].abs(),
            LOWER[2].abs() + BC[3].abs(),
        ];
        let folded = [
            diag[0] + IC[0] + IC[2],
            diag[1],
            diag[2],
            diag[3] + IC[1] + IC[3],
        ];

        let mut out = [0.0 as Scalar; 4];
        for c in 0..4 {
            let dominant = folded[c].abs().max(sum_off[c]) / alpha;
            out[c] = if folded[c] < 0.0 { -dominant } else { dominant };
        }
        out
    }

    #[test]
    fn relax_enforces_the_diagonal_dominance_rule() {
        let Some((gpu, _hm, m, k)) = ctx() else { return };

        let alpha: Scalar = 0.7;

        let mut a = GpuLduMatrix::new(&gpu, &m).expect("matrix");
        a.zero(&gpu).expect("zero");
        fill(&gpu, &mut a);

        let d_psi = gpu.upload(&PSI.to_vec()).expect("psi");
        relax(&gpu, &k, &mut a, &m, &d_psi, alpha).expect("relax");
        // relax writes an increment against the diagonal it PREDICTS the fold
        // will produce, so folding afterwards is what completes the row - and
        // the result has to be exactly max(|D_folded|, sum|off|)/alpha.
        add_boundary_contributions(&gpu, &k, &mut a, &m).expect("fold");
        gpu.sync().expect("sync");

        let want = expected_relaxed_diag(alpha, &DIAG);
        let got = gpu.download(&a.diag).expect("diag");
        assert!(max_diff(&got, &want) < 1e-13, "{got:?} vs {want:?}");

        // b' - b = (diag' - diag)*psi, with both diagonals fully folded.
        let folded_unrelaxed = [
            DIAG[0] + IC[0] + IC[2],
            DIAG[1],
            DIAG[2],
            DIAG[3] + IC[1] + IC[3],
        ];
        let expect_source: Vec<Scalar> = (0..4)
            .map(|c| {
                let base = SOURCE[c] + if c == 0 { BC[0] } else if c == 3 { BC[1] } else { 0.0 };
                base + (want[c] - folded_unrelaxed[c]) * PSI[c]
            })
            .collect();

        let got_source = gpu.download(&a.source).expect("source");
        assert!(
            max_diff(&got_source, &expect_source) < 1e-13,
            "{got_source:?} vs {expect_source:?}"
        );
    }

    /// The reason the rule is written the way it is: relaxation must not move
    /// the answer, only the path to it. `A'.psi - b'` equals `A.psi - b` at the
    /// psi that was passed in, exactly.
    #[test]
    fn relax_leaves_the_residual_at_the_current_iterate_unchanged() {
        let Some((gpu, _hm, m, k)) = ctx() else { return };

        let d_psi = gpu.upload(&PSI.to_vec()).expect("psi");

        let residual = |alpha: Option<Scalar>| -> Vec<Scalar> {
            let mut a = GpuLduMatrix::new(&gpu, &m).expect("matrix");
            a.zero(&gpu).expect("zero");
            fill(&gpu, &mut a);
            if let Some(al) = alpha {
                relax(&gpu, &k, &mut a, &m, &d_psi, al).expect("relax");
            }
            add_boundary_contributions(&gpu, &k, &mut a, &m).expect("fold");

            let mut ap: DevBuf<Scalar> = gpu.zeros(4).expect("alloc");
            amul(&gpu, &k, &mut ap, &d_psi, &a, &m).expect("amul");
            gpu.sync().expect("sync");

            let ap = gpu.download(&ap).expect("Apsi");
            let b = gpu.download(&a.source).expect("source");
            ap.iter().zip(&b).map(|(x, y)| x - y).collect()
        };

        let plain = residual(None);
        for alpha in [0.3 as Scalar, 0.7, 1.0] {
            let relaxed = residual(Some(alpha));
            assert!(
                max_diff(&plain, &relaxed) < 1e-12,
                "alpha = {alpha}: {relaxed:?} vs {plain:?}"
            );
        }
    }

    /// `alpha = 1` on a matrix that is already dominant changes nothing at
    /// all - including on the cyclic rows, where the coupled coefficient
    /// appears on both the diagonal and the off-diagonal side of the test.
    #[test]
    fn relax_of_a_dominant_matrix_at_alpha_one_is_a_no_op() {
        let Some((gpu, _hm, m, k)) = ctx() else { return };

        let mut a = GpuLduMatrix::new(&gpu, &m).expect("matrix");
        a.zero(&gpu).expect("zero");
        fill(&gpu, &mut a);

        let before_diag = gpu.download(&a.diag).expect("diag");
        let before_source = gpu.download(&a.source).expect("source");

        let d_psi = gpu.upload(&PSI.to_vec()).expect("psi");
        relax(&gpu, &k, &mut a, &m, &d_psi, 1.0).expect("relax");
        gpu.sync().expect("sync");

        assert!(max_diff(&gpu.download(&a.diag).expect("diag"), &before_diag) < 1e-15);
        assert!(max_diff(&gpu.download(&a.source).expect("source"), &before_source) < 1e-15);
    }

    /// A row whose off-diagonals outweigh its diagonal is lifted to the sum,
    /// then divided by alpha. This is the half of the rule the previous test
    /// cannot see.
    #[test]
    fn relax_lifts_a_weak_diagonal_to_the_off_diagonal_sum() {
        let Some((gpu, hm, m, k)) = ctx() else { return };

        let alpha: Scalar = 0.5;
        let weak: [Scalar; 4] = [0.01, 0.02, 0.03, 0.04];

        let mut a = GpuLduMatrix::new(&gpu, &m).expect("matrix");
        a.zero(&gpu).expect("zero");
        fill(&gpu, &mut a);
        gpu.write(&mut a.diag, &weak).expect("diag");

        let d_psi = gpu.upload(&PSI.to_vec()).expect("psi");
        relax(&gpu, &k, &mut a, &m, &d_psi, alpha).expect("relax");
        add_boundary_contributions(&gpu, &k, &mut a, &m).expect("fold");
        gpu.sync().expect("sync");

        let want = expected_relaxed_diag(alpha, &weak);
        let got = gpu.download(&a.diag).expect("diag");
        assert!(max_diff(&got, &want) < 1e-13, "{got:?} vs {want:?}");

        // Every row is now dominant, by construction: |diag| >= sum|off|/alpha.
        let dense = dense_from_device(&gpu, &a, &hm);
        for (i, row) in dense.iter().enumerate() {
            let off: Scalar = (0..4).filter(|j| *j != i).map(|j| row[j].abs()).sum();
            assert!(
                row[i].abs() >= off / alpha - 1e-12,
                "row {i}: |{}| < {off}/{alpha}",
                row[i]
            );
        }
    }

    #[test]
    fn relax_rejects_a_non_positive_factor() {
        let Some((gpu, _hm, m, k)) = ctx() else { return };
        let mut a = GpuLduMatrix::new(&gpu, &m).expect("matrix");
        a.zero(&gpu).expect("zero");
        let psi = gpu.upload(&PSI.to_vec()).expect("psi");

        assert!(relax(&gpu, &k, &mut a, &m, &psi, 0.0).is_err());
        assert!(relax(&gpu, &k, &mut a, &m, &psi, -0.5).is_err());
        assert!(relax(&gpu, &k, &mut a, &m, &psi, Scalar::NAN).is_err());
    }

    // ----------------------------------------------------------------------
    //  setValues
    // ----------------------------------------------------------------------

    #[test]
    fn set_values_leaves_a_genuinely_decoupled_row() {
        let Some((gpu, hm, m, k)) = ctx() else { return };

        let mut a = GpuLduMatrix::new(&gpu, &m).expect("matrix");
        a.zero(&gpu).expect("zero");
        fill(&gpu, &mut a);

        // Cell 1 is interior; cell 3 carries both a plain and a coupled
        // boundary face, so its constraint has to survive the fold and has to
        // reach cell 0 across the couple.
        let fixed_cells: [Label; 2] = [1, 3];
        let fixed_values: [Scalar; 2] = [7.5, -3.25];
        set_fixed_cells(&gpu, &mut a, &fixed_cells, &fixed_values).expect("flags");

        set_values(&gpu, &k, &mut a, &m).expect("setValues");
        add_boundary_contributions(&gpu, &k, &mut a, &m).expect("fold");
        gpu.sync().expect("sync");

        let dense = dense_from_device(&gpu, &a, &hm);
        let source = gpu.download(&a.source).expect("source");

        for (i, &cell) in fixed_cells.iter().enumerate() {
            let c = cell as usize;
            let v = fixed_values[i];

            // The row: diagonal only.
            for j in 0..4 {
                if j != c {
                    assert!(
                        dense[c][j] == 0.0,
                        "row {c} still has A({c},{j}) = {}",
                        dense[c][j]
                    );
                }
            }
            assert!(dense[c][c] != 0.0, "row {c} has no diagonal left");
            assert!(
                (source[c] - dense[c][c] * v).abs() < 1e-13,
                "row {c}: source {} != diag {} * value {v}",
                source[c],
                dense[c][c]
            );

            // The column, eliminated into the neighbours' sources.
            for r in 0..4 {
                if r != c {
                    assert!(
                        dense[r][c] == 0.0,
                        "column {c} survives in row {r}: {}",
                        dense[r][c]
                    );
                }
            }
        }

        // A fixed cell's boundary pair is zeroed, so folding could not have
        // put anything back into a row that was already final.
        let ic = gpu.download(&a.internal_coeffs).expect("ic");
        let bc = gpu.download(&a.boundary_coeffs).expect("bc");
        for bf in 0..hm.n_boundary_faces {
            let c = hm.b_face_cells[bf];
            if fixed_cells.contains(&c) {
                assert_eq!(ic[bf], 0.0, "internalCoeffs[{bf}] survived");
                assert_eq!(bc[bf], 0.0, "boundaryCoeffs[{bf}] survived");
            }
        }

        // And the system means what it says: solving it returns the pinned
        // values exactly, and the unpinned cells solve against them.
        let x = dense_solve(&dense, &[source[0], source[1], source[2], source[3]]);
        assert!((x[1] - 7.5).abs() < 1e-12, "psi[1] = {}", x[1]);
        assert!((x[3] + 3.25).abs() < 1e-12, "psi[3] = {}", x[3]);
    }

    /// Column elimination has to move the coefficient it drops, not just drop
    /// it: cell 0's source must gain what its face and its cyclic couple to
    /// the fixed cells took away, or the answer changes.
    #[test]
    fn set_values_moves_the_eliminated_column_into_the_source() {
        let Some((gpu, hm, m, k)) = ctx() else { return };

        let build = |fixed: bool| -> ([[Scalar; 4]; 4], Vec<Scalar>) {
            let mut a = GpuLduMatrix::new(&gpu, &m).expect("matrix");
            a.zero(&gpu).expect("zero");
            fill(&gpu, &mut a);
            if fixed {
                set_fixed_cells(&gpu, &mut a, &[3], &[-3.25]).expect("flags");
                set_values(&gpu, &k, &mut a, &m).expect("setValues");
            }
            add_boundary_contributions(&gpu, &k, &mut a, &m).expect("fold");
            gpu.sync().expect("sync");
            (
                dense_from_device(&gpu, &a, &hm),
                gpu.download(&a.source).expect("source"),
            )
        };

        let (plain, plain_b) = build(false);
        let (pinned, pinned_b) = build(true);

        // Cell 0 sees cell 3 only across the cyclic couple; cell 2 sees it
        // across internal face 2. Both must have paid for it in the source.
        for r in [0usize, 2] {
            let moved = plain[r][3] * -3.25;
            assert!(
                (pinned_b[r] - (plain_b[r] - moved)).abs() < 1e-13,
                "row {r}: source {} should be {} - {moved}",
                pinned_b[r],
                plain_b[r]
            );
            assert_eq!(pinned[r][3], 0.0, "row {r} kept its column entry");
        }

        // Everything the constraint does not touch is untouched.
        assert!((pinned[0][1] - plain[0][1]).abs() < 1e-15);
        assert!((pinned_b[1] - plain_b[1]).abs() < 1e-15);
    }

    // ----------------------------------------------------------------------
    //  csrFill
    // ----------------------------------------------------------------------

    #[test]
    fn csr_fill_reproduces_the_uncoupled_operator() {
        let Some((gpu, hm, m, k)) = ctx() else { return };

        let mut a = GpuLduMatrix::new(&gpu, &m).expect("matrix");
        a.zero(&gpu).expect("zero");
        fill(&gpu, &mut a);
        add_boundary_contributions(&gpu, &k, &mut a, &m).expect("fold");

        let pattern = CsrPattern::build(&hm);
        let mut csr = pattern.upload(&gpu).expect("upload the pattern");
        csr_fill(&gpu, &k, &mut csr, &a).expect("csrFill");
        gpu.sync().expect("sync");

        let row_ptr = gpu.download(&csr.row_ptr).expect("rowPtr");
        let col_ind = gpu.download(&csr.col_ind).expect("colInd");
        let val = gpu.download(&csr.val).expect("val");

        // The CSR carries diag/upper/lower and has no column for a coupled
        // face, so the operator it represents is the dense one with the
        // couple removed. Anything else would mean the permutation is wrong.
        let mut dense = dense_from_device(&gpu, &a, &hm);
        for bf in 0..hm.n_boundary_faces {
            let nbr = hm.b_nbr_cell[bf];
            if nbr >= 0 {
                dense[hm.b_face_cells[bf] as usize][nbr as usize] = 0.0;
            }
        }

        let want = dense_mul(&dense, &PSI);
        let got: Vec<Scalar> = (0..4)
            .map(|r| {
                (row_ptr[r] as usize..row_ptr[r + 1] as usize)
                    .map(|j| val[j] * PSI[col_ind[j] as usize])
                    .sum()
            })
            .collect();

        assert!(max_diff(&got, &want) < 1e-13, "{got:?} vs {want:?}");
        assert_eq!(csr.nnz, hm.n_cells + 2 * hm.n_internal_faces);
    }

    #[test]
    fn csr_fill_rejects_a_pattern_from_a_different_mesh() {
        let Some((gpu, hm, m, k)) = ctx() else { return };

        let mut a = GpuLduMatrix::new(&gpu, &m).expect("matrix");
        a.zero(&gpu).expect("zero");

        let mut smaller = hm.clone();
        smaller.n_cells = 3;
        smaller.cf_offset.truncate(4);
        let mut csr = CsrPattern::build(&smaller).upload(&gpu).expect("upload");

        assert!(csr_fill(&gpu, &k, &mut csr, &a).is_err());
    }

    /// A grid dimension of zero is an invalid launch configuration, not a
    /// no-op, so every launcher has to notice for itself. This would fail with
    /// CUDA_ERROR_INVALID_VALUE the moment one of them forgot.
    #[test]
    fn an_empty_mesh_launches_nothing() {
        let Some(gpu) = Gpu::new(0).ok() else { return };
        let k = LduKernels::new(&gpu).expect("load cuda/ldu.cu");

        let hm = HostMesh {
            n_cells: 0,
            n_internal_faces: 0,
            n_boundary_faces: 0,
            cf_offset: vec![0],
            bcf_offset: vec![0],
            ..Default::default()
        };
        let m = GpuMesh::upload(&gpu, &hm).expect("upload");

        let mut a = GpuLduMatrix::new(&gpu, &m).expect("matrix");
        a.zero(&gpu).expect("zero");

        let empty: DevBuf<Scalar> = gpu.zeros(0).expect("alloc");
        let mut out: DevBuf<Scalar> = gpu.zeros(0).expect("alloc");

        neg_sum_diag(&gpu, &k, &mut a, &m).expect("negSumDiag");
        relax(&gpu, &k, &mut a, &m, &empty, 0.7).expect("relax");
        set_values(&gpu, &k, &mut a, &m).expect("setValues");
        add_boundary_contributions(&gpu, &k, &mut a, &m).expect("fold");
        amul(&gpu, &k, &mut out, &empty, &a, &m).expect("amul");
        gpu.sync().expect("sync");
    }
}
