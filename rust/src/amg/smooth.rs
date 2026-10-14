// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! The hierarchy's smoothers - AMG-03a.
//!
//! The row maps of one level, the setup of the three point smoothers
//! (l1-Jacobi, weighted Jacobi, Chebyshev), their Lanczos estimate of
//! the largest eigenvalue, the host twins written in `Scalar` in the
//! kernels' exact order, and the device driver of the three kernels of
//! `cuda/amg.cu`: `amgAmul`, `amgJacobi` (141.3.2) and `amgChebyshev`
//! (141.3.5). A sweep only launches kernels, so it captures into a CUDA
//! graph (SPEC-LIT §81, §141.3).
//!
//! Provenance: LITERATURE - written from Baker, Falgout, Kolev & Yang,
//! LLNL-TR-489114 (2011), eqs. 2.2, 4.1-4.3, 6.1-6.2, and Saad
//! (2003) Algorithms 6.15 and 12.1 (SPEC-LIT §141.3). No GPL-licensed
//! source was consulted.

use cudarc::driver::{CudaFunction, PushKernelArg};

use crate::amg::galerkin::{GpuLevel, HostLevel};
use crate::device::{cfg_for, DevBuf, Gpu, KernelSet};
use crate::error::{Error, Result};
use crate::ldu::{GpuLduMatrix, HostLduMatrix};
use crate::ldu_ops::{self, LduKernels};
use crate::mesh::topology::{RF_BOUNDARY, RF_OWNS};
use crate::mesh::{GpuMesh, HostMesh};
use crate::{Label, Scalar};

/// *DESIGN* default relaxation of weighted Jacobi (SPEC-LIT §141.3): the
/// degree-1 member of (141.3.6) on the 1-D interval `[1, 2]`, so the high
/// half of a 1-D Poisson matrix is damped by `1/3`.
pub const DEFAULT_OMEGA: f64 = 2.0 / 3.0;
/// *DESIGN* default Chebyshev degree (SPEC-LIT §141.3).
pub const DEFAULT_CHEBYSHEV_DEGREE: usize = 2;
/// The largest Chebyshev degree (SPEC-LIT §141.3).
pub const MAX_CHEBYSHEV_DEGREE: usize = 16;
/// The report's `a` of its section 4.1 (SPEC-LIT §141.3): `alpha = a * beta`.
pub const DEFAULT_CHEBYSHEV_FRACTION: f64 = 0.3;
/// Lanczos steps of (141.3.7)'s `theta_10` (SPEC-LIT §141.3).
pub const LANCZOS_STEPS: usize = 10;
/// PETSc's documented default factor on the estimate (SPEC-LIT §141.3).
pub const LAMBDA_SAFETY: f64 = 1.1;
/// The invariant-subspace stop of the Lanczos recurrence (SPEC-LIT §141.3).
pub const LANCZOS_BREAKDOWN: f64 = 1e-10;
/// The start vector's irrational multiplier, `frac((c + 1) * s) - 1/2`
/// (SPEC-LIT §141.3).
pub const LANCZOS_START: f64 = 0.6180339887498949;

/// Which point smoother a level runs (SPEC-LIT §141.3).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SmootherKind {
    /// l1-Jacobi of (141.3.3), the report's eq. 6.1 with blocks of size one.
    L1Jacobi,
    /// Weighted Jacobi of (141.3.4); `0 < omega < 2` or refused.
    WeightedJacobi { omega: f64 },
    /// Chebyshev of (141.3.5) on the interval of (141.3.7); `1 <= degree
    /// <= 16` and `0 < fraction < 1` or refused.
    Chebyshev { degree: usize, fraction: f64 },
}

impl Default for SmootherKind {
    fn default() -> Self {
        Self::L1Jacobi
    }
}

/// One level's rows in f64, for setup (SPEC-LIT §141.3): `diag[c]` and the
/// off-diagonal entries of row `c` at `offset[c]..offset[c + 1]` - column
/// `col`, value `val` - in the order `amul` sums them.
#[derive(Debug, Clone, PartialEq)]
pub struct HostRows {
    pub n_cells: usize,
    pub diag: Vec<f64>,
    pub offset: Vec<usize>,
    pub col: Vec<usize>,
    pub val: Vec<f64>,
}

/// A non-finite coefficient, named by array and index (SPEC-LIT §141.3).
fn finite(name: &str, i: usize) -> Error {
    Error::Config(format!(
        "amg::smooth: {name}[{i}] is not finite; a non-finite coefficient \
         is refused"
    ))
}

/// An addressing entry out of range (SPEC-LIT §141.3).
fn out_of_range(name: &str, i: usize, v: Label, n: usize) -> Error {
    Error::Mesh(format!(
        "amg::smooth: {name}[{i}] = {v} out of 0..{n}"
    ))
}

impl HostRows {
    /// The rows of a coarse level, through [`LevelRows::of_level`]'s order:
    /// an owned face gives `(neighbour, upper)` and a neighbour-side face
    /// gives `(owner, lower)`, each row's faces in ascending face id
    /// (SPEC-LIT §141.3).
    pub fn of_level(l: &HostLevel) -> Result<Self> {
        let n = l.n_cells;
        let nf = l.n_faces();
        if l.diag.len() != n {
            return Err(Error::Config(format!(
                "amg::smooth: length mismatch: diag has {} entries, the \
                 level has {n} cells",
                l.diag.len()
            )));
        }
        if l.owner.len() != nf || l.neighbour.len() != nf {
            return Err(Error::Config(format!(
                "amg::smooth: length mismatch: owner/neighbour have \
                 {}/{} entries, the level has {nf} faces",
                l.owner.len(),
                l.neighbour.len()
            )));
        }
        if l.upper.len() != nf || l.lower.len() != nf {
            return Err(Error::Config(format!(
                "amg::smooth: length mismatch: upper/lower have {}/{} \
                 entries, the level has {nf} faces",
                l.upper.len(),
                l.lower.len()
            )));
        }
        for (c, &d) in l.diag.iter().enumerate() {
            if !(d as f64).is_finite() {
                return Err(finite("diag", c));
            }
        }
        for (f, &o) in l.owner.iter().enumerate() {
            if o < 0 || o as usize >= n {
                return Err(out_of_range("owner", f, o, n));
            }
            if l.neighbour[f] < 0 || l.neighbour[f] as usize >= n {
                return Err(out_of_range("neighbour", f, l.neighbour[f], n));
            }
        }
        for (f, &u) in l.upper.iter().enumerate() {
            if !(u as f64).is_finite() {
                return Err(finite("upper", f));
            }
            if !(l.lower[f] as f64).is_finite() {
                return Err(finite("lower", f));
            }
        }
        let mut r = Self::fill_level(
            n, &l.owner, &l.neighbour, &l.upper, &l.lower,
        );
        r.diag = l.diag.iter().map(|&d| d as f64).collect();
        Ok(r)
    }

    /// The row structure itself: counts, then one ascending pass over the
    /// faces, which leaves every cell's entries ascending in face id.
    fn fill_level(
        n: usize,
        owner: &[Label],
        neighbour: &[Label],
        upper: &[Scalar],
        lower: &[Scalar],
    ) -> Self {
        let nf = owner.len();
        let mut offset = vec![0usize; n + 1];
        for f in 0..nf {
            offset[owner[f] as usize + 1] += 1;
            offset[neighbour[f] as usize + 1] += 1;
        }
        for c in 0..n {
            offset[c + 1] += offset[c];
        }
        let mut col = vec![0usize; 2 * nf];
        let mut val = vec![0.0f64; 2 * nf];
        let mut cursor = offset[..n].to_vec();
        for f in 0..nf {
            let io = cursor[owner[f] as usize];
            col[io] = neighbour[f] as usize;
            val[io] = upper[f] as f64;
            cursor[owner[f] as usize] = io + 1;
            let inb = cursor[neighbour[f] as usize];
            col[inb] = owner[f] as usize;
            val[inb] = lower[f] as f64;
            cursor[neighbour[f] as usize] = inb + 1;
        }
        Self {
            n_cells: n,
            diag: Vec::new(),
            offset,
            col,
            val,
        }
    }

    /// Level 0: the mesh's merged row map (`rf_offset`/`rf_face`/
    /// `rf_flags`) in its order - internal faces as [`Self::of_level`]
    /// does, a coupled boundary face giving `(b_nbr_cell,
    /// -boundary_coeffs)` and an uncoupled one nothing (SPEC-LIT §141.3).
    /// A coupled face whose neighbour is a halo cell is refused (`halo`).
    pub fn of_matrix(m: &HostMesh, a: &HostLduMatrix) -> Result<Self> {
        let n = m.n_cells;
        let (nif, nbf) = (m.n_internal_faces, m.n_boundary_faces);
        if a.diag.len() != n {
            return Err(Error::Config(format!(
                "amg::smooth: length mismatch: diag has {} entries, the \
                 mesh has {n} cells",
                a.diag.len()
            )));
        }
        if a.upper.len() != nif || a.lower.len() != nif {
            return Err(Error::Config(format!(
                "amg::smooth: length mismatch: upper/lower have {}/{} \
                 entries, the mesh has {nif} internal faces",
                a.upper.len(),
                a.lower.len()
            )));
        }
        if a.boundary_coeffs.len() != nbf {
            return Err(Error::Config(format!(
                "amg::smooth: length mismatch: boundary_coeffs has {} \
                 entries, the mesh has {nbf} boundary faces",
                a.boundary_coeffs.len()
            )));
        }
        if m.rf_offset.len() != n + 1 {
            return Err(Error::Config(format!(
                "amg::smooth: length mismatch: rf_offset has {} entries, \
                 the mesh has {n} cells",
                m.rf_offset.len()
            )));
        }
        let nrow = 2 * nif + nbf;
        if m.rf_face.len() != nrow || m.rf_flags.len() != nrow {
            return Err(Error::Config(format!(
                "amg::smooth: length mismatch: rf_face/rf_flags have \
                 {}/{} entries, the row map needs {nrow}",
                m.rf_face.len(),
                m.rf_flags.len()
            )));
        }
        for (c, &d) in a.diag.iter().enumerate() {
            if !(d as f64).is_finite() {
                return Err(finite("diag", c));
            }
        }
        for (f, &o) in m.owner.iter().enumerate() {
            if o < 0 || o as usize >= n {
                return Err(out_of_range("owner", f, o, n));
            }
            if m.neighbour[f] < 0 || m.neighbour[f] as usize >= n {
                return Err(out_of_range("neighbour", f, m.neighbour[f], n));
            }
            if !(a.upper[f] as f64).is_finite() {
                return Err(finite("upper", f));
            }
            if !(a.lower[f] as f64).is_finite() {
                return Err(finite("lower", f));
            }
        }
        for (bf, &d) in m.b_nbr_cell.iter().enumerate() {
            // A halo neighbour is refused BY NAME, before anything else
            // reads the face (SPEC-LIT §141.3, as §141.1).
            if d >= 0 && d as usize >= n {
                return Err(Error::Mesh(format!(
                    "amg::smooth: b_nbr_cell[{bf}] = {d} is a halo cell, \
                     which a level-0 row may not reach; a coupled face \
                     with a halo neighbour is refused"
                )));
            }
            if d >= 0 && !(a.boundary_coeffs[bf] as f64).is_finite() {
                return Err(finite("boundary_coeffs", bf));
            }
        }
        let mut r = Self {
            n_cells: n,
            diag: a.diag.iter().map(|&d| d as f64).collect(),
            offset: vec![0usize; n + 1],
            col: Vec::with_capacity(nrow),
            val: Vec::with_capacity(nrow),
        };
        for c in 0..n {
            for j in m.rf_offset[c] as usize..m.rf_offset[c + 1] as usize {
                let (f, fl) = (m.rf_face[j], m.rf_flags[j]);
                if fl & RF_BOUNDARY != 0 {
                    let d = m.b_nbr_cell[f as usize];
                    if d >= 0 {
                        r.col.push(d as usize);
                        r.val.push(-(a.boundary_coeffs[f as usize] as f64));
                    }
                } else if fl & RF_OWNS != 0 {
                    r.col.push(m.neighbour[f as usize] as usize);
                    r.val.push(a.upper[f as usize] as f64);
                } else {
                    r.col.push(m.owner[f as usize] as usize);
                    r.val.push(a.lower[f as usize] as f64);
                }
            }
            r.offset[c + 1] = r.col.len();
        }
        Ok(r)
    }

    /// `y = A x` in f64, row order - the setup arithmetic of Lanczos and
    /// of the energy check (SPEC-LIT §141.3). `x` is at least `n_cells`
    /// long; the rows of [`Self::of_matrix`] count only coupled boundary
    /// faces, so a row's slice is exactly its `amul` order.
    pub fn amul(&self, x: &[f64]) -> Vec<f64> {
        debug_assert!(x.len() >= self.n_cells);
        let mut y = vec![0.0f64; self.n_cells];
        for c in 0..self.n_cells {
            let mut sum = self.diag[c] * x[c];
            for j in self.offset[c]..self.offset[c + 1] {
                sum += self.val[j] * x[self.col[j]];
            }
            y[c] = sum;
        }
        y
    }
}

/// The row map of a level (SPEC-LIT §141.3, §70.2's convention): per cell
/// its faces in ascending face id, flagged [`RF_OWNS`] on the owner side
/// and 0 on the neighbour side. `2 * n_faces` entries.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LevelRows {
    pub n_cells: usize,
    pub offset: Vec<Label>,
    pub face: Vec<Label>,
    pub flags: Vec<Label>,
}

impl LevelRows {
    /// The row map of one coarse level: owner and neighbour checked
    /// against `n_cells`, lengths against the level (SPEC-LIT §141.3).
    pub fn of_level(l: &HostLevel) -> Result<Self> {
        let n = l.n_cells;
        let nf = l.n_faces();
        if l.owner.len() != nf || l.neighbour.len() != nf {
            return Err(Error::Config(format!(
                "amg::smooth: length mismatch: owner/neighbour have \
                 {}/{} entries, the level has {nf} faces",
                l.owner.len(),
                l.neighbour.len()
            )));
        }
        if l.diag.len() != n {
            return Err(Error::Config(format!(
                "amg::smooth: length mismatch: diag has {} entries, the \
                 level has {n} cells",
                l.diag.len()
            )));
        }
        for (f, &o) in l.owner.iter().enumerate() {
            if o < 0 || o as usize >= n {
                return Err(out_of_range("owner", f, o, n));
            }
            if l.neighbour[f] < 0 || l.neighbour[f] as usize >= n {
                return Err(out_of_range("neighbour", f, l.neighbour[f], n));
            }
        }
        let mut offset = vec![0 as Label; n + 1];
        for f in 0..nf {
            offset[l.owner[f] as usize + 1] += 1;
            offset[l.neighbour[f] as usize + 1] += 1;
        }
        for c in 0..n {
            offset[c + 1] += offset[c];
        }
        let mut face = vec![0 as Label; 2 * nf];
        let mut flags = vec![0 as Label; 2 * nf];
        let mut cursor: Vec<Label> = offset[..n].to_vec();
        for f in 0..nf {
            let o = l.owner[f] as usize;
            let io = cursor[o] as usize;
            face[io] = f as Label;
            flags[io] = RF_OWNS;
            cursor[o] = io as Label + 1;
            let nb = l.neighbour[f] as usize;
            let inb = cursor[nb] as usize;
            face[inb] = f as Label;
            flags[inb] = 0;
            cursor[nb] = inb as Label + 1;
        }
        Ok(Self {
            n_cells: n,
            offset,
            face,
            flags,
        })
    }
}

/// The sign of `sum(diag)` in f64 (SPEC-LIT §141.3); a zero or non-finite
/// sum is refused by name.
fn sigma_of(diag: &[f64]) -> Result<f64> {
    let sum: f64 = diag.iter().sum();
    if !sum.is_finite() || sum == 0.0 {
        return Err(Error::Config(format!(
            "amg::smooth: the diagonal sums to {sum}, so the sign sigma is \
             undefined; a zero or non-finite diagonal is refused"
        )));
    }
    Ok(if sum > 0.0 { 1.0 } else { -1.0 })
}

/// SPEC-LIT §141.3's Lanczos (Saad Algorithm 6.15) on
/// `S = sigma |D|^-1/2 A |D|^-1/2`, which has the spectrum of `D^-1 A`:
/// the largest Ritz value and the steps actually run. `steps` is capped at
/// `n_cells`.
pub fn lanczos_lambda_max(rows: &HostRows, steps: usize) -> Result<(f64, usize)> {
    let n = rows.n_cells;
    let sigma = sigma_of(&rows.diag)?;
    for (c, &d) in rows.diag.iter().enumerate() {
        if !(sigma * d > 0.0) {
            return Err(Error::Config(format!(
                "amg::smooth: row {c} has sigma * diagonal {} with sigma \
                 {sigma}, which is not positive; the row is refused",
                sigma * d
            )));
        }
    }
    let max_steps = steps.min(n);
    if max_steps == 0 {
        return Ok((0.0, 0));
    }
    let sqrt_d: Vec<f64> = rows.diag.iter().map(|&d| d.abs().sqrt()).collect();
    // The deterministic start (SPEC-LIT §141.3): not symmetric, so it is
    // not orthogonal to an antisymmetric top eigenvector.
    let mut v: Vec<f64> = (0..n)
        .map(|c| {
            let x = (c + 1) as f64 * LANCZOS_START;
            x - x.trunc() - 0.5
        })
        .collect();
    let norm: f64 = v.iter().map(|&x| x * x).sum::<f64>().sqrt();
    for x in v.iter_mut() {
        *x /= norm;
    }
    let mut alphas: Vec<f64> = Vec::with_capacity(max_steps);
    // betas[k] holds beta_{k+2} of the recurrence.
    let mut betas: Vec<f64> = Vec::with_capacity(max_steps - 1);
    let mut v_prev = vec![0.0f64; n];
    let mut beta_j = 0.0f64;
    let mut j_run = 0usize;
    for j in 1..=max_steps {
        // w = S v_j - beta_j v_{j-1}, with S v the scaled product.
        let vs: Vec<f64> = v
            .iter()
            .zip(&sqrt_d)
            .map(|(&x, &s)| x / s)
            .collect();
        let av = rows.amul(&vs);
        let mut w: Vec<f64> = (0..n)
            .map(|c| sigma * av[c] / sqrt_d[c] - beta_j * v_prev[c])
            .collect();
        let alpha_j: f64 = w.iter().zip(&v).map(|(&a, &b)| a * b).sum();
        for (wc, &vc) in w.iter_mut().zip(&v) {
            *wc -= alpha_j * vc;
        }
        let beta_next: f64 = w.iter().map(|&x| x * x).sum::<f64>().sqrt();
        alphas.push(alpha_j);
        j_run = j;
        // t_j = max(|alpha_1|..|alpha_j|, beta_2..beta_j).
        let mut t = 0.0f64;
        for &a in &alphas {
            t = t.max(a.abs());
        }
        for &b in &betas {
            t = t.max(b.abs());
        }
        if j == max_steps || beta_next <= LANCZOS_BREAKDOWN * t {
            break;
        }
        betas.push(beta_next);
        for c in 0..n {
            v_prev[c] = v[c];
            v[c] = w[c] / beta_next;
        }
        beta_j = beta_next;
    }
    let theta = tridiag_max_eig(&alphas, &betas)?;
    Ok((theta, j_run))
}

/// Largest eigenvalue of the symmetric tridiagonal with diagonal `diag`
/// and off-diagonal `off`, `off.len() + 1 == diag.len()`, by bisection on
/// the Sturm count inside Gershgorin's interval of the matrix, until the
/// bracket is `2^-50 max(|lo|, |hi|, f64::MIN_POSITIVE)` wide (SPEC-LIT
/// §141.3).
pub fn tridiag_max_eig(diag: &[f64], off: &[f64]) -> Result<f64> {
    if off.len() + 1 != diag.len() {
        return Err(Error::Config(format!(
            "amg::smooth: length mismatch: the off-diagonal has {} entries, \
             the diagonal {}, and a tridiagonal needs one fewer off-diagonal",
            off.len(),
            diag.len()
        )));
    }
    let n = diag.len();
    if n == 0 {
        return Ok(0.0);
    }
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for (i, &d) in diag.iter().enumerate() {
        let mut r = 0.0f64;
        if i > 0 {
            r += off[i - 1].abs();
        }
        if i + 1 < n {
            r += off[i].abs();
        }
        lo = lo.min(d - r);
        hi = hi.max(d + r);
    }
    // The Sturm count: how many eigenvalues are below x. A zero pivot is
    // stood in by the smallest positive normal, which keeps the count
    // deterministic at a measure-zero crossing.
    let count = |x: f64| -> usize {
        let mut cnt = 0usize;
        let mut q = diag[0] - x;
        if q < 0.0 {
            cnt += 1;
        }
        for i in 1..n {
            let qq = if q == 0.0 {
                diag[i] - x - off[i - 1] * off[i - 1] / f64::MIN_POSITIVE
            } else {
                diag[i] - x - off[i - 1] * off[i - 1] / q
            };
            if qq < 0.0 {
                cnt += 1;
            }
            q = qq;
        }
        cnt
    };
    // count(x) == n exactly when x is above the largest eigenvalue.
    let tol = |lo: f64, hi: f64| {
        2f64.powi(-50) * lo.abs().max(hi.abs()).max(f64::MIN_POSITIVE)
    };
    while hi - lo > tol(lo, hi) {
        let mid = 0.5 * (lo + hi);
        if mid <= lo || mid >= hi {
            break;
        }
        if count(mid) == n {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    Ok(lo)
}

/// `1 / T_degree((beta + alpha) / (beta - alpha))`, with
/// `T_k(t) = cosh(k acosh t)` for `t >= 1` (SPEC-LIT §141.3, (141.3.6)).
pub fn chebyshev_bound(degree: usize, alpha: f64, beta: f64) -> f64 {
    let t = (beta + alpha) / (beta - alpha);
    let k = degree as f64;
    1.0 / (k * t.acosh()).cosh()
}

/// The Chebyshev setup of one level (SPEC-LIT §141.3): the interval
/// `[alpha, beta]` of (141.3.7), the degree's bound of (141.3.6) and the
/// recurrence coefficients of (141.3.5), `c1[0] = 0`, each rounded once
/// from f64 to `Scalar`.
#[derive(Debug, Clone, PartialEq)]
pub struct ChebyshevSetup {
    pub lambda_ritz: f64,
    pub lanczos_steps: usize,
    pub alpha: f64,
    pub beta: f64,
    pub bound: f64,
    pub c1: Vec<Scalar>,
    pub c2: Vec<Scalar>,
}

/// One level's smoother setup (SPEC-LIT §141.3): the sign `sigma`, the
/// row checks, the smoothing diagonal `m` of (141.3.3)/(141.3.4) or
/// `m = diag` for Chebyshev, Gershgorin's `g` of (141.3.7) and, for
/// Chebyshev, the Lanczos run and the coefficients of (141.3.5). All in
/// f64, each `m`/`c` rounded once to `Scalar`.
#[derive(Debug, Clone, PartialEq)]
pub struct SmootherSetup {
    pub kind: SmootherKind,
    pub n_cells: usize,
    pub sigma: f64,
    pub gershgorin: f64,
    pub m: Vec<Scalar>,
    pub chebyshev: Option<ChebyshevSetup>,
}

impl SmootherSetup {
    /// The setup of `kind` on `rows`. Every check runs before anything is
    /// stored: the kind's constants, then `sigma`, then the rows
    /// (SPEC-LIT §141.3). `n_cells == 0` gives an empty setup.
    pub fn new(rows: &HostRows, kind: SmootherKind) -> Result<Self> {
        let mut s = Self::without_interval(rows, kind)?;
        if kind_is_chebyshev(kind) && s.n_cells > 0 {
            let (theta, ran) = lanczos_lambda_max(rows, LANCZOS_STEPS)?;
            let g = s.gershgorin;
            s.set_interval(theta, ran, g)?;
        }
        Ok(s)
    }

    /// Everything of [`Self::new`] but the Chebyshev interval: the kind's
    /// constants, `sigma`, the row checks, `m` and the rows' own Gershgorin
    /// bound, with `chebyshev` left `None`. The distributed hierarchy
    /// (SPEC-LIT §141.11) calls it on a part's rows, whose off-diagonal
    /// columns may name halo cells; it reads no column, only values.
    pub fn without_interval(rows: &HostRows, kind: SmootherKind) -> Result<Self> {
        if let SmootherKind::WeightedJacobi { omega } = kind {
            if !(omega > 0.0 && omega < 2.0) {
                return Err(Error::Config(format!(
                    "amg::smooth: omega is {omega}; weighted Jacobi \
                     converges only for 0 < omega < 2, so the omega \
                     outside that interval is refused"
                )));
            }
        }
        let (degree, fraction) = match kind {
            SmootherKind::Chebyshev { degree, fraction } => (degree, fraction),
            _ => (0, 0.0),
        };
        if kind_is_chebyshev(kind) {
            if !(1..=MAX_CHEBYSHEV_DEGREE).contains(&degree) {
                return Err(Error::Config(format!(
                    "amg::smooth: the Chebyshev degree is {degree}; it \
                     must be between 1 and {MAX_CHEBYSHEV_DEGREE}, so the \
                     degree outside that range is refused"
                )));
            }
            if !(fraction > 0.0 && fraction < 1.0) {
                return Err(Error::Config(format!(
                    "amg::smooth: the Chebyshev fraction is {fraction}; \
                     alpha = fraction * beta needs 0 < fraction < 1, so \
                     the fraction outside that interval is refused"
                )));
            }
        }
        let n = rows.n_cells;
        if n == 0 {
            return Ok(Self {
                kind,
                n_cells: 0,
                sigma: 0.0,
                gershgorin: 0.0,
                m: Vec::new(),
                chebyshev: None,
            });
        }
        let sigma = sigma_of(&rows.diag)?;
        let mut m = Vec::with_capacity(n);
        let mut gershgorin = 0.0f64;
        for c in 0..n {
            let a_cc = rows.diag[c];
            if !a_cc.is_finite() || !(sigma * a_cc > 0.0) {
                return Err(Error::Config(format!(
                    "amg::smooth: row {c} has diagonal {a_cc} with sigma \
                     {sigma}, so sigma * diagonal is not positive and \
                     finite; the diagonal is refused"
                )));
            }
            let mut off = 0.0f64;
            for j in rows.offset[c]..rows.offset[c + 1] {
                let v = rows.val[j];
                if !v.is_finite() {
                    return Err(Error::Config(format!(
                        "amg::smooth: row {c} has an off-diagonal entry \
                         {v} that is not finite; the coefficient is \
                         refused"
                    )));
                }
                off += v.abs();
            }
            gershgorin = gershgorin.max((a_cc.abs() + off) / a_cc.abs());
            let m_c = match kind {
                SmootherKind::L1Jacobi => a_cc + sigma * off,
                SmootherKind::WeightedJacobi { omega } => a_cc / omega,
                _ => a_cc,
            };
            m.push(m_c as Scalar);
        }
        Ok(Self {
            kind,
            n_cells: n,
            sigma,
            gershgorin,
            m,
            chebyshev: None,
        })
    }

    /// The Chebyshev interval of (141.3.7) and the coefficients of
    /// (141.3.5), from the Ritz value `theta`, the Lanczos steps `ran`
    /// that produced it and Gershgorin's bound `gershgorin` (stored), for a
    /// Chebyshev kind; any other kind is left as it is (SPEC-LIT §141.3).
    /// [`Self::new`] passes its own rows' values; the distributed hierarchy
    /// passes the whole level's (SPEC-LIT §141.11).
    pub fn set_interval(&mut self, theta: f64, ran: usize, gershgorin: f64) -> Result<()> {
        let SmootherKind::Chebyshev { degree, fraction } = self.kind else {
            return Ok(());
        };
        let beta = (LAMBDA_SAFETY * theta).min(gershgorin);
        let alpha = fraction * beta;
        if !(beta > 0.0 && beta.is_finite()) || !(alpha > 0.0 && alpha.is_finite()) {
            return Err(Error::Config(format!(
                "amg::smooth: the interval is [{alpha}, {beta}], which \
                 is not positive and finite; the interval is refused"
            )));
        }
        self.gershgorin = gershgorin;
        self.chebyshev = Some(chebyshev_coefficients(degree, alpha, beta, theta, ran));
        Ok(())
    }
}

fn kind_is_chebyshev(kind: SmootherKind) -> bool {
    matches!(kind, SmootherKind::Chebyshev { .. })
}

/// The recurrence of (141.3.5) on `[alpha, beta]`: `c1[0] = 0`, the rest
/// through the rho recursion, each rounded once to `Scalar`.
fn chebyshev_coefficients(
    degree: usize,
    alpha: f64,
    beta: f64,
    theta_ritz: f64,
    lanczos_steps: usize,
) -> ChebyshevSetup {
    let th = (beta + alpha) / 2.0;
    let delta = (beta - alpha) / 2.0;
    let s = th / delta;
    let mut rho = 1.0 / s;
    let bound = chebyshev_bound(degree, alpha, beta);
    let mut c1 = Vec::with_capacity(degree);
    let mut c2 = Vec::with_capacity(degree);
    c1.push(0.0 as Scalar);
    c2.push((1.0 / th) as Scalar);
    for _ in 1..degree {
        let rho_prev = rho;
        rho = 1.0 / (2.0 * s - rho_prev);
        c1.push((rho * rho_prev) as Scalar);
        c2.push((2.0 * rho / delta) as Scalar);
    }
    ChebyshevSetup {
        lambda_ritz: theta_ritz,
        lanczos_steps,
        alpha,
        beta,
        bound,
        c1,
        c2,
    }
}

/// Host twin of `amgAmul`: `y = A x` through the row map, in `Scalar`,
/// the kernel's order exactly (SPEC-LIT §141.3).
pub fn level_amul_host(
    l: &HostLevel,
    rows: &LevelRows,
    x: &[Scalar],
) -> Result<Vec<Scalar>> {
    if rows.n_cells != l.n_cells {
        return Err(Error::Config(format!(
            "amg::smooth: the row map has {} cells but the level has {} \
             cells",
            rows.n_cells, l.n_cells
        )));
    }
    if x.len() < l.n_cells {
        return Err(Error::Config(format!(
            "amg::smooth: length mismatch: x has {} entries, the level \
             needs {}",
            x.len(),
            l.n_cells
        )));
    }
    let n = l.n_cells;
    let mut y = vec![0 as Scalar; n];
    for c in 0..n {
        let mut sum = l.diag[c] * x[c];
        for j in rows.offset[c] as usize..rows.offset[c + 1] as usize {
            let f = rows.face[j] as usize;
            if rows.flags[j] & RF_OWNS != 0 {
                sum += l.upper[f] * x[l.neighbour[f] as usize];
            } else {
                sum += l.lower[f] * x[l.owner[f] as usize];
            }
        }
        y[c] = sum;
    }
    Ok(y)
}

/// Host twin of a sweep: `sweeps` sweeps of (141.3.2) for the Jacobi
/// kinds, and for Chebyshev one sweep is the whole degree-nu polynomial -
/// `nu` products, steps `k = 0..nu - 1`, step 0 with `first = 1`
/// (SPEC-LIT §141.3). Every product and add in `Scalar`, the kernel's
/// order exactly.
pub fn smooth_level_host(
    l: &HostLevel,
    rows: &LevelRows,
    s: &SmootherSetup,
    x: &mut [Scalar],
    b: &[Scalar],
    sweeps: usize,
) -> Result<()> {
    if s.n_cells != l.n_cells || rows.n_cells != l.n_cells {
        return Err(Error::Config(format!(
            "amg::smooth: the setup has {} cells and the row map {} but \
             the level has {} cells",
            s.n_cells, rows.n_cells, l.n_cells
        )));
    }
    if x.len() < l.n_cells || b.len() < l.n_cells {
        return Err(Error::Config(format!(
            "amg::smooth: length mismatch: x has {} and b has {} entries, \
             the level needs {}",
            x.len(),
            b.len(),
            l.n_cells
        )));
    }
    let n = l.n_cells;
    if n == 0 || sweeps == 0 {
        return Ok(());
    }
    let nu = match s.kind {
        SmootherKind::L1Jacobi | SmootherKind::WeightedJacobi { .. } => 0,
        SmootherKind::Chebyshev { degree, .. } => degree,
    };
    if nu == 0 {
        for _ in 0..sweeps {
            let y = level_amul_host(l, rows, x)?;
            for c in 0..n {
                x[c] = x[c] + (b[c] - y[c]) / s.m[c];
            }
        }
        return Ok(());
    }
    let cheb = s.chebyshev.as_ref().ok_or_else(|| {
        Error::Config(
            "amg::smooth: a Chebyshev smoother of degree > 0 without a \
             Chebyshev setup is refused"
                .to_string(),
        )
    })?;
    let mut d = vec![0 as Scalar; n];
    for _ in 0..sweeps {
        for k in 0..nu {
            let y = level_amul_host(l, rows, x)?;
            let (c1k, c2k) = (cheb.c1[k], cheb.c2[k]);
            let first = k == 0;
            for c in 0..n {
                let z = (b[c] - y[c]) / s.m[c];
                let dk = if first { c2k * z } else { c1k * d[c] + c2k * z };
                d[c] = dk;
                x[c] = x[c] + dk;
            }
        }
    }
    Ok(())
}

/// A level's row map on the device, with the resolved `amgAmul` (SPEC-LIT
/// §141.3). Setup uploads, so it runs outside a capture; `amul` only
/// launches.
pub struct GpuLevelRows {
    n_cells: usize,
    offset: DevBuf<Label>,
    face: DevBuf<Label>,
    flags: DevBuf<Label>,
    amul: CudaFunction,
}

impl GpuLevelRows {
    /// Upload the row map and resolve the kernel, once.
    pub fn new(gpu: &Gpu, rows: &LevelRows) -> Result<Self> {
        let lab1 = [0 as Label];
        let k = KernelSet::new(gpu, crate::kernels::AMG)?;
        Ok(Self {
            n_cells: rows.n_cells,
            offset: gpu.upload(if rows.offset.is_empty() {
                &lab1
            } else {
                &rows.offset
            })?,
            face: gpu.upload(if rows.face.is_empty() {
                &lab1
            } else {
                &rows.face
            })?,
            flags: gpu.upload(if rows.flags.is_empty() {
                &lab1
            } else {
                &rows.flags
            })?,
            amul: k.func("amgAmul")?,
        })
    }

    /// `amgAmul`: the level's product through the row map. Only a launch
    /// - no allocation, no copy, no sync, so it captures into a graph
    /// (SPEC-LIT §81, §141.3). `y` and `x` are at least `n_cells` long;
    /// the level's shape must match the row map's.
    pub fn amul(
        &self,
        gpu: &Gpu,
        level: &GpuLevel,
        y: &mut DevBuf<Scalar>,
        x: &DevBuf<Scalar>,
    ) -> Result<()> {
        if level.n_cells != self.n_cells {
            return Err(Error::Config(format!(
                "amg::smooth: the row map has {} cells but the level has \
                 {} cells",
                self.n_cells, level.n_cells
            )));
        }
        if y.len() < self.n_cells || x.len() < self.n_cells {
            return Err(Error::Config(format!(
                "amg::smooth: length mismatch: y has {} and x has {} \
                 entries, the level needs {}",
                y.len(),
                x.len(),
                self.n_cells
            )));
        }
        if self.n_cells == 0 {
            return Ok(());
        }
        let nl = self.n_cells as Label;
        let f = self.amul.clone();
        unsafe {
            gpu.stream()
                .launch_builder(&f)
                .arg(&mut *y)
                .arg(x)
                .arg(&level.diag)
                .arg(&level.upper)
                .arg(&level.lower)
                .arg(&level.owner)
                .arg(&level.neighbour)
                .arg(&self.offset)
                .arg(&self.face)
                .arg(&self.flags)
                .arg(&nl)
                .launch(cfg_for(self.n_cells))?;
        }
        Ok(())
    }
}

/// One level's smoother on the device (SPEC-LIT §141.3): the smoothing
/// diagonal `m` uploaded from the setup, the scratch `y` and `d` (`n_cells`
/// each, zeroed at setup - step 0 of (141.3.5) never reads `d`), the
/// Chebyshev coefficients on the host, and the resolved `amgJacobi` and
/// `amgChebyshev`.
pub struct GpuSmoother {
    n_cells: usize,
    degree: usize,
    m: DevBuf<Scalar>,
    y: DevBuf<Scalar>,
    d: DevBuf<Scalar>,
    c1: Vec<Scalar>,
    c2: Vec<Scalar>,
    jacobi: CudaFunction,
    chebyshev: CudaFunction,
}

impl GpuSmoother {
    /// Upload the setup's diagonal, allocate the scratch and resolve the
    /// two update kernels, once. Setup allocates, so it runs outside a
    /// capture; the sweeps only launch.
    pub fn new(gpu: &Gpu, s: &SmootherSetup) -> Result<Self> {
        let k = KernelSet::new(gpu, crate::kernels::AMG)?;
        let lab1 = [0 as Scalar];
        let degree = match s.kind {
            SmootherKind::Chebyshev { degree, .. } => degree,
            _ => 0,
        };
        Ok(Self {
            n_cells: s.n_cells,
            degree,
            m: gpu.upload(if s.m.is_empty() { &lab1 } else { &s.m })?,
            y: gpu.zeros(s.n_cells.max(1))?,
            d: gpu.zeros(s.n_cells.max(1))?,
            c1: s
                .chebyshev
                .as_ref()
                .map(|c| c.c1.clone())
                .unwrap_or_default(),
            c2: s
                .chebyshev
                .as_ref()
                .map(|c| c.c2.clone())
                .unwrap_or_default(),
            jacobi: k.func("amgJacobi")?,
            chebyshev: k.func("amgChebyshev")?,
        })
    }

    /// The cells the smoother was set up for.
    pub fn n_cells(&self) -> usize {
        self.n_cells
    }

    /// The shape and length checks the sweeps share: the setup's
    /// `n_cells` against the level or matrix, then the fields.
    fn check(&self, cells: usize, x: &DevBuf<Scalar>, b: &DevBuf<Scalar>) -> Result<()> {
        if cells != self.n_cells {
            return Err(Error::Config(format!(
                "amg::smooth: the smoother was set up for {} cells but \
                 the level has {} cells",
                self.n_cells, cells
            )));
        }
        if x.len() < self.n_cells || b.len() < self.n_cells {
            return Err(Error::Config(format!(
                "amg::smooth: length mismatch: x has {} and b has {} \
                 entries, the level needs {}",
                x.len(),
                b.len(),
                self.n_cells
            )));
        }
        Ok(())
    }

    /// The two update launches a step can end with: (141.3.2) for the
    /// Jacobi kinds, one step of (141.3.5) for Chebyshev. Only launches.
    fn update(
        &mut self,
        gpu: &Gpu,
        k: usize,
        x: &mut DevBuf<Scalar>,
        b: &DevBuf<Scalar>,
        nl: Label,
    ) -> Result<()> {
        if self.degree == 0 {
            let f = self.jacobi.clone();
            unsafe {
                gpu.stream()
                    .launch_builder(&f)
                    .arg(&mut *x)
                    .arg(b)
                    .arg(&self.y)
                    .arg(&self.m)
                    .arg(&nl)
                    .launch(cfg_for(self.n_cells))?;
            }
        } else {
            let first: Label = if k == 0 { 1 } else { 0 };
            let c1k = self.c1[k];
            let c2k = self.c2[k];
            let f = self.chebyshev.clone();
            unsafe {
                gpu.stream()
                    .launch_builder(&f)
                    .arg(&mut *x)
                    .arg(&mut self.d)
                    .arg(b)
                    .arg(&self.y)
                    .arg(&self.m)
                    .arg(&c1k)
                    .arg(&c2k)
                    .arg(&first)
                    .arg(&nl)
                    .launch(cfg_for(self.n_cells))?;
            }
        }
        Ok(())
    }

    /// A coarse (or any boundary-free) level: per step
    /// `rows.amul(y = A x)`, then the update. Only launches, so it
    /// captures into a graph (SPEC-LIT §81, §141.3). `sweeps == 0` and
    /// `n_cells == 0` launch nothing.
    pub fn smooth_level(
        &mut self,
        gpu: &Gpu,
        level: &GpuLevel,
        rows: &GpuLevelRows,
        x: &mut DevBuf<Scalar>,
        b: &DevBuf<Scalar>,
        sweeps: usize,
    ) -> Result<()> {
        self.check(level.n_cells, x, b)?;
        if rows.n_cells != level.n_cells {
            return Err(Error::Config(format!(
                "amg::smooth: the row map has {} cells but the level has \
                 {} cells",
                rows.n_cells, level.n_cells
            )));
        }
        if self.n_cells == 0 || sweeps == 0 {
            return Ok(());
        }
        let nl = self.n_cells as Label;
        for _ in 0..sweeps {
            for k in 0..self.degree.max(1) {
                rows.amul(gpu, level, &mut self.y, x)?;
                self.update(gpu, k, x, b, nl)?;
            }
        }
        Ok(())
    }

    /// Level 0: per step `ldu_ops::amul(gpu, k, &mut y, x, a, mesh)`, then
    /// the same update. Only launches, so it captures into a graph
    /// (SPEC-LIT §81, §141.3). `sweeps == 0` and `n_cells == 0` launch
    /// nothing.
    pub fn smooth_fine(
        &mut self,
        gpu: &Gpu,
        k: &LduKernels,
        mesh: &GpuMesh,
        a: &GpuLduMatrix,
        x: &mut DevBuf<Scalar>,
        b: &DevBuf<Scalar>,
        sweeps: usize,
    ) -> Result<()> {
        self.check(a.n_cells, x, b)?;
        if self.n_cells == 0 || sweeps == 0 {
            return Ok(());
        }
        let nl = self.n_cells as Label;
        for _ in 0..sweeps {
            for step in 0..self.degree.max(1) {
                ldu_ops::amul(gpu, k, &mut self.y, x, a, mesh)?;
                self.update(gpu, step, x, b, nl)?;
            }
        }
        Ok(())
    }

    /// A level whose product the caller forms (SPEC-LIT §141.11): per step
    /// `amul(y, x)` writes `y = A x` into the smoother's scratch, then the
    /// same update as [`Self::smooth_level`]. `x` is `&mut` in the product
    /// because a decomposed level fills its halo slots there first. `cells`
    /// is the level's owned-cell count; `sweeps == 0` and `n_cells == 0`
    /// call nothing.
    pub fn smooth_with<F>(
        &mut self,
        gpu: &Gpu,
        cells: usize,
        x: &mut DevBuf<Scalar>,
        b: &DevBuf<Scalar>,
        sweeps: usize,
        mut amul: F,
    ) -> Result<()>
    where
        F: FnMut(&mut DevBuf<Scalar>, &mut DevBuf<Scalar>) -> Result<()>,
    {
        self.check(cells, x, b)?;
        if self.n_cells == 0 || sweeps == 0 {
            return Ok(());
        }
        let nl = self.n_cells as Label;
        for _ in 0..sweeps {
            for step in 0..self.degree.max(1) {
                amul(&mut self.y, x)?;
                self.update(gpu, step, x, b, nl)?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::blockgen::{BlockSpec, GradedAxis, build_mesh};
    use crate::capture::{Snapshot, capture_replays_bitwise};
    use crate::ldu::{GpuLduMatrix, HostLduMatrix};
    use crate::mesh::GpuMesh;

    /// A uniform 6x6x6 box, every patch a wall.
    fn box6() -> HostMesh {
        let axis = |n: usize| GradedAxis {
            lo: 0.0,
            hi: n as Scalar,
            n,
            expansion: 1.0,
            two_sided: false,
        };
        let b = BlockSpec {
            x: axis(6),
            y: axis(6),
            z: axis(6),
            windows: Vec::new(),
            patch_name: BlockSpec::default().patch_name,
            patch_type: ["wall"; 6].map(String::from),
            cyclic: Vec::new(),
        };
        build_mesh(&b).expect("box mesh")
    }

    /// The symmetric level of the gate table: `a_ab = a_ba = -(1/10 +
    /// ((11a + 3b) mod 7)/3)` for an edge `a < b`, `a_cc = 20 + c/9`.
    fn sym6(m: &HostMesh) -> HostLevel {
        let mut l = HostLevel {
            n_cells: m.n_cells,
            owner: m.owner.clone(),
            neighbour: m.neighbour.clone(),
            diag: vec![0 as Scalar; m.n_cells],
            upper: vec![0 as Scalar; m.n_internal_faces],
            lower: vec![0 as Scalar; m.n_internal_faces],
        };
        for (c, d) in l.diag.iter_mut().enumerate() {
            *d = (20.0 + c as f64 / 9.0) as Scalar;
        }
        for f in 0..m.n_internal_faces {
            let a = m.owner[f].min(m.neighbour[f]) as usize;
            let b = m.owner[f].max(m.neighbour[f]) as usize;
            let v = -(0.1 + ((11 * a + 3 * b) % 7) as f64 / 3.0);
            l.upper[f] = v as Scalar;
            l.lower[f] = v as Scalar;
        }
        l
    }

    /// The graded 6^3 laplacian as a solve receives it, the level-0
    /// fixture of the capture gate.
    fn graded_poisson() -> (HostMesh, HostLduMatrix) {
        let g = GradedAxis {
            lo: 0.0,
            hi: 1.0,
            n: 6,
            expansion: 3.0,
            two_sided: false,
        };
        let b = BlockSpec {
            x: g.clone(),
            y: g.clone(),
            z: g,
            windows: Vec::new(),
            patch_name: BlockSpec::default().patch_name,
            patch_type: ["wall"; 6].map(String::from),
            cyclic: Vec::new(),
        };
        let m = build_mesh(&b).expect("graded mesh");
        let mut a = HostLduMatrix::zeros(&m);
        for f in 0..m.n_internal_faces {
            let s = m.mag_sf[f] * m.delta_coeffs[f];
            a.upper[f] = s;
            a.lower[f] = s;
            a.diag[m.owner[f] as usize] -= s;
            a.diag[m.neighbour[f] as usize] -= s;
        }
        for bf in 0..m.n_boundary_faces {
            let c = m.b_face_cells[bf] as usize;
            a.internal_coeffs[bf] = -(m.b_mag_sf[bf] * m.b_delta_coeffs[bf]);
            a.diag[c] += a.internal_coeffs[bf];
        }
        (m, a)
    }

    /// The registry's gate (`src/capture/registry.rs`, row
    /// `src/amg/smooth.rs`): the three smoothers on a coarse level and
    /// l1-Jacobi on level 0 capture into a CUDA graph and replay bitwise
    /// (SPEC-LIT 81, 141.3).
    #[test]
    fn smoothers_replay_under_capture() {
        let Some(gpu) = Gpu::new(0).ok() else { return };

        let report = capture_replays_bitwise(
            &gpu,
            "the AMG smoothers (SPEC-LIT 141.3)",
            || {
                let m = box6();
                let l = sym6(&m);
                let hrows = HostRows::of_level(&l)?;
                let lrows = LevelRows::of_level(&l)?;
                let level = l.upload(&gpu)?;
                let rows = GpuLevelRows::new(&gpu, &lrows)?;
                let s1 = GpuSmoother::new(
                    &gpu,
                    &SmootherSetup::new(&hrows, SmootherKind::L1Jacobi)?,
                )?;
                let s2 = GpuSmoother::new(
                    &gpu,
                    &SmootherSetup::new(
                        &hrows,
                        SmootherKind::WeightedJacobi {
                            omega: DEFAULT_OMEGA,
                        },
                    )?,
                )?;
                let s3 = GpuSmoother::new(
                    &gpu,
                    &SmootherSetup::new(
                        &hrows,
                        SmootherKind::Chebyshev {
                            degree: 2,
                            fraction: 0.3,
                        },
                    )?,
                )?;
                let (gm, ga) = graded_poisson();
                let mesh = GpuMesh::upload(&gpu, &gm)?;
                let mut mat = GpuLduMatrix::new(&gpu, &mesh)?;
                gpu.write(&mut mat.diag, &ga.diag)?;
                gpu.write(&mut mat.upper, &ga.upper)?;
                gpu.write(&mut mat.lower, &ga.lower)?;
                gpu.write(&mut mat.boundary_coeffs, &ga.boundary_coeffs)?;
                let frows = HostRows::of_matrix(&gm, &ga)?;
                let sf = GpuSmoother::new(
                    &gpu,
                    &SmootherSetup::new(&frows, SmootherKind::L1Jacobi)?,
                )?;
                let k = LduKernels::new(&gpu)?;
                let n = l.n_cells;
                let field = || -> Vec<Scalar> {
                    (0..n)
                        .map(|c| {
                            (0.1 * c as f64 + ((7 * c + 3) % 11) as f64 / 13.0)
                                as Scalar
                        })
                        .collect()
                };
                let x1 = gpu.upload(&field())?;
                let x2 = gpu.upload(&field())?;
                let x3 = gpu.upload(&field())?;
                let x4 = gpu.upload(&field())?;
                let b: Vec<Scalar> = (0..n)
                    .map(|c| (1.0 + (c % 5) as f64 / 4.0) as Scalar)
                    .collect();
                let b = gpu.upload(&b)?;
                Ok((
                    level, rows, s1, s2, s3, mesh, mat, sf, k, x1, x2, x3,
                    x4, b,
                ))
            },
            |(level, rows, s1, s2, s3, mesh, mat, sf, k, x1, x2, x3, x4, b): &mut (
                GpuLevel,
                GpuLevelRows,
                GpuSmoother,
                GpuSmoother,
                GpuSmoother,
                GpuMesh,
                GpuLduMatrix,
                GpuSmoother,
                LduKernels,
                DevBuf<Scalar>,
                DevBuf<Scalar>,
                DevBuf<Scalar>,
                DevBuf<Scalar>,
                DevBuf<Scalar>,
            )| {
                s1.smooth_level(&gpu, level, rows, x1, b, 2)?;
                s2.smooth_level(&gpu, level, rows, x2, b, 2)?;
                s3.smooth_level(&gpu, level, rows, x3, b, 2)?;
                sf.smooth_fine(&gpu, k, mesh, mat, x4, b, 2)?;
                Ok(())
            },
            |(_level, _rows, s1, s2, s3, _mesh, _mat, sf, _k, x1, x2, x3, x4, _b): &(
                GpuLevel,
                GpuLevelRows,
                GpuSmoother,
                GpuSmoother,
                GpuSmoother,
                GpuMesh,
                GpuLduMatrix,
                GpuSmoother,
                LduKernels,
                DevBuf<Scalar>,
                DevBuf<Scalar>,
                DevBuf<Scalar>,
                DevBuf<Scalar>,
                DevBuf<Scalar>,
            )| {
                let mut state: Snapshot = Vec::new();
                state.push(("smooth x1", gpu.download(x1)?));
                state.push(("smooth x2", gpu.download(x2)?));
                state.push(("smooth x3", gpu.download(x3)?));
                state.push(("smooth x4", gpu.download(x4)?));
                state.push(("smooth s1 y", gpu.download(&s1.y)?));
                state.push(("smooth s1 d", gpu.download(&s1.d)?));
                state.push(("smooth s2 y", gpu.download(&s2.y)?));
                state.push(("smooth s2 d", gpu.download(&s2.d)?));
                state.push(("smooth s3 y", gpu.download(&s3.y)?));
                state.push(("smooth s3 d", gpu.download(&s3.d)?));
                state.push(("smooth sf y", gpu.download(&sf.y)?));
                state.push(("smooth sf d", gpu.download(&sf.d)?));
                Ok(state)
            },
        )
        .expect("SPEC-LIT 81.7: the AMG smoothers must capture and replay \
                 bitwise");
        println!("  AMG smooth: {report}");
    }
}
