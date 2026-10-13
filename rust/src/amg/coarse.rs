// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! The coarsest-level solve - AMG-03b.
//!
//! A dense Cholesky factor of the coarsest level, formed on the host in
//! f64 at setup, and one device kernel, `amgCoarseSolve` of
//! `cuda/amg.cu`, that runs the two triangular substitutions in a single
//! thread block (SPEC-LIT §141.4). A singular level - a pure-Neumann
//! laplacian and every Galerkin product of it - is projected and pinned
//! by (141.4.3)/(141.4.4); a level that is not symmetric or whose
//! factorisation meets a bad pivot is refused by name, or, under
//! `-permissive`, substituted by 50 l1-Jacobi sweeps through the
//! smoother of §141.3. A solve only launches, so it captures into a
//! CUDA graph (SPEC-LIT §81, §141.4).
//!
//! Provenance: LITERATURE - written from the LAPACK 3.9.0 `DPOTRF`
//! documentation and the entrywise recurrence of the open article
//! "Cholesky decomposition", via SPEC-LIT §141.4. No GPL-licensed
//! source was consulted.

use cudarc::driver::{CudaFunction, LaunchConfig, PushKernelArg};

use crate::amg::galerkin::{GpuLevel, HostLevel};
use crate::amg::smooth::{
    GpuLevelRows, GpuSmoother, HostRows, LevelRows, SmootherKind, SmootherSetup,
    smooth_level_host,
};
use crate::device::{DevBuf, Gpu, KernelSet};
use crate::error::{Error, Result};
use crate::ldu::GpuLduMatrix;
use crate::ldu_ops::LduKernels;
use crate::mesh::GpuMesh;
use crate::{Acc, Label, Scalar};

/// The largest level a dense factor is formed for (SPEC-LIT §141.4,
/// "Size"): `n(n+1)/2` doubles and `n^3/6` host operations at most. The
/// remedy for a larger level is a larger `max_levels` or a smaller
/// `n_cells_coarsest` (§141.3).
pub const MAX_DENSE_CELLS: usize = 1024;

/// The `-permissive` substitution's sweep count (SPEC-LIT §141.4).
pub const FALLBACK_SWEEPS: usize = 50;

/// The pivot floor `tau_p` of (141.4.5): a pivot `s` that is not finite
/// or not `> tau_p * A_ii` refuses the level as not SPD.
pub const PIVOT_FLOOR: f64 = 1e-13;

/// Threads of the one block `amgCoarseSolve` runs in (SPEC-LIT §141.4).
pub const COARSE_BLOCK: u32 = 256;

/// `tau_s` of (141.4.2), `1024` machine epsilons of [`Scalar`]: how far
/// a row sum may sit from zero and the level still count as singular
/// with the constant null vector.
pub fn singular_tolerance() -> f64 {
    1024.0 * Scalar::EPSILON as f64
}

/// Which way a level is solved (SPEC-LIT §141.4): the exact dense
/// Cholesky, the projected and pinned Cholesky of a singular level, or
/// the `-permissive` l1-Jacobi substitution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoarseMethod {
    /// The level is SPD: `m = n`, one factor, one pair of substitutions.
    Cholesky,
    /// The level is singular with the constant null vector: `m = n - 1`,
    /// the projection (141.4.3), the pin at the last cell, the
    /// re-projection (141.4.4).
    PinnedCholesky,
    /// The level is not SPD and the caller passed `-permissive`: 50
    /// l1-Jacobi sweeps from `x = 0`, not an exact solve.
    JacobiFallback,
}

/// The not-SPD refusal (SPEC-LIT §141.4): names the reason word and
/// `-permissive`, starts with the unit.
fn not_spd(reason: &str, detail: &str) -> Error {
    Error::Refused(format!(
        "amg::coarse: the level is not SPD ({reason}: {detail}); pass \
         -permissive to substitute {FALLBACK_SWEEPS} l1-Jacobi sweeps"
    ))
}

/// The dense matrix of (141.4.1) in f64, row by row: the diagonal scaled
/// by `sigma`, then every entry of the row accumulated into its column
/// in the row's entry order.
fn dense_of_rows(n: usize, sigma: f64, rows: &HostRows) -> Vec<f64> {
    let mut a = vec![0.0f64; n * n];
    for c in 0..n {
        a[c * n + c] = sigma * rows.diag[c];
        for j in rows.offset[c]..rows.offset[c + 1] {
            a[c * n + rows.col[j]] += sigma * rows.val[j];
        }
    }
    a
}

/// (141.4.2): the level is singular with the constant null vector when
/// every row sums to zero within `tau_s` of its entries' magnitude.
fn is_singular(n: usize, a: &[f64]) -> bool {
    let tau = singular_tolerance();
    (0..n).all(|c| {
        let mut sum = a[c * n + c];
        let mut mag = a[c * n + c].abs();
        for j in 0..n {
            if j != c {
                sum += a[c * n + j];
                mag += a[c * n + j].abs();
            }
        }
        sum.abs() <= tau * mag
    })
}

/// The packed Cholesky factor of (141.4.5) of the leading `m x m` block
/// of the dense `a` (`n` wide): `s = A_ij - sum L_ik L_jk`, the diagonal
/// `L_ii = sqrt(s)`, the strict part `L_ij = s / L_jj`. `L_ij` sits at
/// `i(i+1)/2 + j`. A pivot that is not finite or not `> tau_p A_ii`
/// refuses the level with the failing pivot's index - LAPACK's "the
/// leading minor of order i is not positive definite".
fn factor(n: usize, a: &[f64], m: usize) -> std::result::Result<Vec<f64>, usize> {
    let mut l = vec![0.0f64; m * (m + 1) / 2];
    for i in 0..m {
        for j in 0..=i {
            let mut s = a[i * n + j];
            for k in 0..j {
                s -= l[i * (i + 1) / 2 + k] * l[j * (j + 1) / 2 + k];
            }
            if j == i {
                if !s.is_finite() || !(s > PIVOT_FLOOR * a[i * n + i]) {
                    return Err(i);
                }
                l[i * (i + 1) / 2 + j] = s.sqrt();
            } else {
                l[i * (i + 1) / 2 + j] = s / l[j * (j + 1) / 2 + j];
            }
        }
    }
    Ok(l)
}

/// The setup of the coarsest-level solve (SPEC-LIT §141.4): the method,
/// the packed factor and, for the `-permissive` substitution, the
/// smoother and its announcement. Everything is formed at setup, in
/// f64, so a solve only runs the substitutions.
#[derive(Debug, Clone)]
pub struct CoarseSetup {
    /// How the level is solved.
    pub method: CoarseMethod,
    /// The cells the setup was built for.
    pub n_cells: usize,
    /// The factor's order: `n` (Cholesky), `n - 1` (pinned), `0`
    /// (fallback).
    pub order: usize,
    /// The sign of `sum(diag)`, `+1` or `-1` (141.3); the dense matrix
    /// is `sigma * A`.
    pub sigma: f64,
    /// The packed factor, `L_ij` at `i(i+1)/2 + j`, `order*(order+1)/2`
    /// entries; empty for the fallback.
    pub l: Vec<f64>,
    /// The substitution's smoother, `Some` only for `JacobiFallback`
    /// (`SmootherKind::L1Jacobi`).
    pub fallback: Option<SmootherSetup>,
    /// The substitution's announcement, `Some` only for
    /// `JacobiFallback`; also printed once on stderr at setup.
    pub announcement: Option<String>,
}

impl CoarseSetup {
    /// The not-SPD outcome (SPEC-LIT §141.4): a named refusal when
    /// strict, the `-permissive` substitution - the l1-Jacobi setup, its
    /// announcement, one `warn_once` - otherwise. Every field check the
    /// caller already ran still holds.
    fn not_spd_outcome(
        rows: &HostRows,
        permissive: bool,
        sigma: f64,
        reason: &str,
        detail: String,
    ) -> Result<Self> {
        let n = rows.n_cells;
        if !permissive {
            return Err(not_spd(reason, &detail));
        }
        let smoother = SmootherSetup::new(rows, SmootherKind::L1Jacobi)?;
        let text = format!(
            "amg::coarse: -permissive: the {n}-cell coarse level is not \
             SPD ({reason}: {detail}); substituting {FALLBACK_SWEEPS} \
             l1-Jacobi sweeps from x = 0, which is not an exact solve"
        );
        crate::io::contract::warn_once("amg::coarse", &text);
        Ok(Self {
            method: CoarseMethod::JacobiFallback,
            n_cells: n,
            order: 0,
            sigma,
            l: Vec::new(),
            fallback: Some(smoother),
            announcement: Some(text),
        })
    }

    /// The setup of the solve on `rows` (SPEC-LIT §141.4). The checks run
    /// in order - cells, finite, diagonal, symmetry, singularity, factor
    /// - and every refusal is by name. `permissive` only turns a not-SPD
    /// level into the l1-Jacobi substitution; the strict refusals stand.
    pub fn new(rows: &HostRows, permissive: bool) -> Result<Self> {
        let n = rows.n_cells;
        if n > MAX_DENSE_CELLS {
            return Err(Error::Config(format!(
                "amg::coarse: the level has {n} cells; a dense factor \
                 needs n(n+1)/2 doubles, so more than {MAX_DENSE_CELLS} \
                 cells is refused - use a larger max_levels or a smaller \
                 n_cells_coarsest"
            )));
        }
        if n == 0 {
            return Ok(Self {
                method: CoarseMethod::Cholesky,
                n_cells: 0,
                order: 0,
                sigma: 1.0,
                l: Vec::new(),
                fallback: None,
                announcement: None,
            });
        }
        for (c, &d) in rows.diag.iter().enumerate() {
            if !d.is_finite() {
                return Err(Error::Config(format!(
                    "amg::coarse: diag[{c}] is {d}, which is not finite; \
                     a non-finite coefficient is refused"
                )));
            }
        }
        for (j, &v) in rows.val.iter().enumerate() {
            if !v.is_finite() {
                return Err(Error::Config(format!(
                    "amg::coarse: val[{j}] is {v}, which is not finite; a \
                     non-finite coefficient is refused"
                )));
            }
        }
        let sum: f64 = rows.diag.iter().sum();
        if !sum.is_finite() || sum == 0.0 {
            return Err(Error::Config(format!(
                "amg::coarse: the diagonal sums to {sum}, so the sign \
                 sigma is undefined; a zero or non-finite diagonal is \
                 refused"
            )));
        }
        let sigma = if sum > 0.0 { 1.0 } else { -1.0 };
        let dense = dense_of_rows(n, sigma, rows);
        let mut asym = None;
        'outer: for i in 0..n {
            for j in 0..i {
                if dense[i * n + j] != dense[j * n + i] {
                    asym = Some((j, i));
                    break 'outer;
                }
            }
        }
        if let Some((i, j)) = asym {
            return Self::not_spd_outcome(
                rows,
                permissive,
                sigma,
                "symmetric",
                format!(
                    "A[{i},{j}] = {} but A[{j},{i}] = {}",
                    dense[i * n + j], dense[j * n + i]
                ),
            );
        }
        let singular = is_singular(n, &dense);
        let m = if singular { n - 1 } else { n };
        match factor(n, &dense, m) {
            Ok(l) => Ok(Self {
                method: if singular {
                    CoarseMethod::PinnedCholesky
                } else {
                    CoarseMethod::Cholesky
                },
                n_cells: n,
                order: m,
                sigma,
                l,
                fallback: None,
                announcement: None,
            }),
            Err(i) => Self::not_spd_outcome(
                rows,
                permissive,
                sigma,
                "pivot",
                format!(
                    "the leading minor of order {i} is not positive \
                     definite, so the factorisation could not be completed"
                ),
            ),
        }
    }

    /// Host twin of [`GpuCoarse::solve`], bit for bit: (141.4.6)'s
    /// substitutions in f64 in the kernel's order - a sequential loop,
    /// since per `i` the updates arrive in the same `j` order.
    /// `fallback_level` is needed only for [`CoarseMethod::JacobiFallback`]
    /// (`x = 0`, then 50 sweeps of the smoother); `None` there is refused.
    pub fn solve_host(
        &self,
        b: &[Scalar],
        fallback_level: Option<(&HostLevel, &LevelRows)>,
    ) -> Result<Vec<Scalar>> {
        let n = self.n_cells;
        if b.len() < n {
            return Err(Error::Config(format!(
                "amg::coarse: length mismatch: b has {} entries, the \
                 level needs {n}",
                b.len()
            )));
        }
        if n == 0 {
            return Ok(Vec::new());
        }
        match self.method {
            CoarseMethod::Cholesky | CoarseMethod::PinnedCholesky => {
                let pinned = self.method == CoarseMethod::PinnedCholesky;
                let m = self.order;
                let mut w: Vec<f64> = b[..n].iter().map(|&v| v as f64).collect();
                if pinned {
                    let mean = w.iter().sum::<f64>() / n as f64;
                    for wc in w.iter_mut() {
                        *wc -= mean;
                    }
                }
                for wc in w.iter_mut() {
                    *wc = self.sigma * *wc;
                }
                for j in 0..m {
                    w[j] /= self.l[j * (j + 1) / 2 + j];
                    for i in j + 1..m {
                        w[i] -= self.l[i * (i + 1) / 2 + j] * w[j];
                    }
                }
                for j in (0..m).rev() {
                    w[j] /= self.l[j * (j + 1) / 2 + j];
                    for i in 0..j {
                        w[i] -= self.l[j * (j + 1) / 2 + i] * w[j];
                    }
                }
                if pinned {
                    w[n - 1] = 0.0;
                    let mean = w.iter().sum::<f64>() / n as f64;
                    for wc in w.iter_mut() {
                        *wc -= mean;
                    }
                }
                Ok(w.into_iter().map(|v| v as Scalar).collect())
            }
            CoarseMethod::JacobiFallback => {
                let (l, lr) = fallback_level.ok_or_else(|| {
                    Error::Config(
                        "amg::coarse: a fallback solve needs the level and \
                         its row map; None is refused"
                            .to_string(),
                    )
                })?;
                if l.n_cells != n || lr.n_cells != n {
                    return Err(Error::Config(format!(
                        "amg::coarse: the operator has {}/{} cells but the \
                         setup was built for {n} cells",
                        l.n_cells, lr.n_cells
                    )));
                }
                let smoother = self.fallback.as_ref().ok_or_else(|| {
                    Error::Config(
                        "amg::coarse: a fallback setup without a smoother \
                         is refused"
                            .to_string(),
                    )
                })?;
                let mut x = vec![0 as Scalar; n];
                smooth_level_host(l, lr, smoother, &mut x, b, FALLBACK_SWEEPS)?;
                Ok(x)
            }
        }
    }
}

/// The operator a solve runs on (SPEC-LIT §141.4): a coarse level
/// through its row map, or the fine matrix itself, whose coupled faces
/// [`GpuSmoother::smooth_fine`] applies through `ldu_ops`.
pub enum CoarseOperator<'a> {
    /// A coarse level and its row map.
    Level {
        level: &'a GpuLevel,
        rows: &'a GpuLevelRows,
    },
    /// The finest level, as the solve receives it.
    Fine {
        k: &'a LduKernels,
        mesh: &'a GpuMesh,
        a: &'a GpuLduMatrix,
    },
}

impl CoarseOperator<'_> {
    /// The cells the operator acts on.
    fn cells(&self) -> usize {
        match self {
            Self::Level { level, .. } => level.n_cells,
            Self::Fine { a, .. } => a.n_cells,
        }
    }
}

/// The coarse solve on the device (SPEC-LIT §141.4): the packed factor
/// and the one-block scratch `w` uploaded or allocated once, the
/// resolved `amgCoarseSolve`, and, for the `-permissive` substitution,
/// the smoother of §141.3. Setup uploads and allocates, so it runs
/// outside a capture; [`Self::solve`] only launches.
pub struct GpuCoarse {
    /// The packed factor, f64 in both builds (`ofacc`).
    l: DevBuf<Acc>,
    /// The one block's work vector, `max(n, 1)` long.
    w: DevBuf<Acc>,
    kernel: CudaFunction,
    fallback: Option<GpuSmoother>,
    n: usize,
    order: usize,
    method: CoarseMethod,
    sigma: f64,
}

/// The one-block configuration of (141.4.6): grid 1, [`COARSE_BLOCK`]
/// threads.
fn one_block() -> LaunchConfig {
    LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (COARSE_BLOCK, 1, 1),
        shared_mem_bytes: 0,
    }
}

impl GpuCoarse {
    /// Upload the factor, allocate the scratch and resolve the kernel,
    /// once.
    pub fn new(gpu: &Gpu, s: &CoarseSetup) -> Result<Self> {
        let k = KernelSet::new(gpu, crate::kernels::AMG)?;
        let one = [0.0f64];
        Ok(Self {
            l: gpu.upload(if s.l.is_empty() { &one } else { &s.l })?,
            w: gpu.zeros(s.n_cells.max(1))?,
            kernel: k.func("amgCoarseSolve")?,
            fallback: match &s.fallback {
                Some(fb) => Some(GpuSmoother::new(gpu, fb)?),
                None => None,
            },
            n: s.n_cells,
            order: s.order,
            method: s.method,
            sigma: s.sigma,
        })
    }

    /// The method the setup resolved to.
    pub fn method(&self) -> CoarseMethod {
        self.method
    }

    /// The cells the setup was built for.
    pub fn n_cells(&self) -> usize {
        self.n
    }

    /// The solve (141.4.6). Cholesky and pinned are ONE launch of
    /// `amgCoarseSolve`, grid 1, block [`COARSE_BLOCK`] - capturable;
    /// the fallback is `fill_zero(x)` and then
    /// [`FALLBACK_SWEEPS`] sweeps through the smoother - also launches
    /// only. `n == 0` does nothing.
    pub fn solve(
        &mut self,
        gpu: &Gpu,
        op: CoarseOperator<'_>,
        b: &DevBuf<Scalar>,
        x: &mut DevBuf<Scalar>,
    ) -> Result<()> {
        let n = self.n;
        if b.len() < n || x.len() < n {
            return Err(Error::Config(format!(
                "amg::coarse: length mismatch: b has {} and x has {} \
                 entries, the level needs {n}",
                b.len(),
                x.len()
            )));
        }
        let cells = op.cells();
        if cells != n {
            return Err(Error::Config(format!(
                "amg::coarse: the operator has {cells} cells but the \
                 setup was built for {n} cells"
            )));
        }
        if n == 0 {
            return Ok(());
        }
        match self.method {
            CoarseMethod::Cholesky | CoarseMethod::PinnedCholesky => {
                let nl = n as Label;
                let ml = self.order as Label;
                let pinned: Label =
                    if self.method == CoarseMethod::PinnedCholesky {
                        1
                    } else {
                        0
                    };
                let sigma = self.sigma;
                let f = self.kernel.clone();
                unsafe {
                    gpu.stream()
                        .launch_builder(&f)
                        .arg(&mut *x)
                        .arg(b)
                        .arg(&self.l)
                        .arg(&mut self.w)
                        .arg(&nl)
                        .arg(&ml)
                        .arg(&pinned)
                        .arg(&sigma)
                        .launch(one_block())?;
                }
                Ok(())
            }
            CoarseMethod::JacobiFallback => {
                let smoother = self.fallback.as_mut().ok_or_else(|| {
                    Error::Config(
                        "amg::coarse: a fallback setup without a smoother \
                         is refused"
                            .to_string(),
                    )
                })?;
                gpu.fill_zero(x)?;
                match op {
                    CoarseOperator::Level { level, rows } => smoother
                        .smooth_level(gpu, level, rows, x, b, FALLBACK_SWEEPS),
                    CoarseOperator::Fine { k, mesh, a } => {
                        smoother.smooth_fine(gpu, k, mesh, a, x, b, FALLBACK_SWEEPS)
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::capture::{Snapshot, capture_replays_bitwise};

    /// A uniform 6x6x6 box, every patch a wall.
    fn box6() -> crate::mesh::HostMesh {
        let axis = |n: usize| crate::blockgen::GradedAxis {
            lo: 0.0,
            hi: n as Scalar,
            n,
            expansion: 1.0,
            two_sided: false,
        };
        let b = crate::blockgen::BlockSpec {
            x: axis(6),
            y: axis(6),
            z: axis(6),
            windows: Vec::new(),
            patch_name: crate::blockgen::BlockSpec::default().patch_name,
            patch_type: ["wall"; 6].map(String::from),
            cyclic: Vec::new(),
        };
        crate::blockgen::build_mesh(&b).expect("box mesh")
    }

    /// The symmetric level of §141.3.1's gate table on `box6`.
    fn sym6() -> HostLevel {
        let m = box6();
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

    /// `[[1, -1], [-1, 1]]`, the two-cell pure-Neumann level: singular,
    /// so a strict setup pins it.
    fn neu2() -> HostLevel {
        HostLevel {
            n_cells: 2,
            owner: vec![0],
            neighbour: vec![1],
            diag: vec![1 as Scalar; 2],
            upper: vec![-1 as Scalar],
            lower: vec![-1 as Scalar],
        }
    }

    /// Two disconnected two-cell Neumann blocks: singular with a
    /// singular leading block, so the pinned factor meets a zero pivot
    /// and a permissive setup substitutes.
    fn disc4() -> HostLevel {
        HostLevel {
            n_cells: 4,
            owner: vec![0, 2],
            neighbour: vec![1, 3],
            diag: vec![1 as Scalar; 4],
            upper: vec![-1 as Scalar; 2],
            lower: vec![-1 as Scalar; 2],
        }
    }

    /// The registry's gate (`src/capture/registry.rs`, row
    /// `src/amg/coarse.rs`): a regular, a pinned and a fallback solve
    /// capture into a CUDA graph and replay bitwise (SPEC-LIT 81,
    /// 141.4).
    #[test]
    fn coarse_solve_replays_under_capture() {
        let Some(gpu) = Gpu::new(0).ok() else { return };

        let report = capture_replays_bitwise(
            &gpu,
            "the AMG coarse solve (SPEC-LIT 141.4)",
            || {
                let l = sym6();
                let hrows = HostRows::of_level(&l)?;
                let lrows = LevelRows::of_level(&l)?;
                let level = l.upload(&gpu)?;
                let rows = GpuLevelRows::new(&gpu, &lrows)?;
                let reg = GpuCoarse::new(&gpu, &CoarseSetup::new(&hrows, false)?)?;
                let neu = neu2();
                let nrows = HostRows::of_level(&neu)?;
                let nlrows = LevelRows::of_level(&neu)?;
                let nlevel = neu.upload(&gpu)?;
                let nrowsg = GpuLevelRows::new(&gpu, &nlrows)?;
                let pin = GpuCoarse::new(&gpu, &CoarseSetup::new(&nrows, false)?)?;
                let disc = disc4();
                let drows = HostRows::of_level(&disc)?;
                let dlrows = LevelRows::of_level(&disc)?;
                let dlevel = disc.upload(&gpu)?;
                let drowsg = GpuLevelRows::new(&gpu, &dlrows)?;
                let fb = GpuCoarse::new(&gpu, &CoarseSetup::new(&drows, true)?)?;
                let b: Vec<Scalar> = (0..l.n_cells)
                    .map(|c| {
                        (0.1 * c as f64 + ((7 * c + 3) % 11) as f64 / 13.0) as Scalar
                    })
                    .collect();
                let b = gpu.upload(&b)?;
                let x1 = gpu.zeros(l.n_cells)?;
                let x2 = gpu.zeros(2)?;
                let x3 = gpu.zeros(4)?;
                Ok((
                    level, rows, reg, nlevel, nrowsg, dlevel, drowsg, pin, fb,
                    b, x1, x2, x3,
                ))
            },
            |(level, rows, reg, nlevel, nrowsg, dlevel, drowsg, pin, fb, b, x1, x2, x3): &mut (
                GpuLevel,
                GpuLevelRows,
                GpuCoarse,
                GpuLevel,
                GpuLevelRows,
                GpuLevel,
                GpuLevelRows,
                GpuCoarse,
                GpuCoarse,
                DevBuf<Scalar>,
                DevBuf<Scalar>,
                DevBuf<Scalar>,
                DevBuf<Scalar>,
            )| {
                reg.solve(
                    &gpu,
                    CoarseOperator::Level { level, rows },
                    b,
                    x1,
                )?;
                pin.solve(
                    &gpu,
                    CoarseOperator::Level {
                        level: nlevel,
                        rows: nrowsg,
                    },
                    b,
                    x2,
                )?;
                fb.solve(
                    &gpu,
                    CoarseOperator::Level {
                        level: dlevel,
                        rows: drowsg,
                    },
                    b,
                    x3,
                )?;
                Ok(())
            },
            |(_level, _rows, reg, _nlevel, _nrowsg, _dlevel, _drowsg, pin, fb, _b, x1, x2, x3): &(
                GpuLevel,
                GpuLevelRows,
                GpuCoarse,
                GpuLevel,
                GpuLevelRows,
                GpuLevel,
                GpuLevelRows,
                GpuCoarse,
                GpuCoarse,
                DevBuf<Scalar>,
                DevBuf<Scalar>,
                DevBuf<Scalar>,
                DevBuf<Scalar>,
            )| {
                let mut state: Snapshot = Vec::new();
                state.push(("coarse x1", gpu.download(x1)?));
                state.push(("coarse x2", gpu.download(x2)?));
                state.push(("coarse x3", gpu.download(x3)?));
                state.push((
                    "coarse w reg",
                    gpu.download(&reg.w)?.into_iter().map(|v| v as Scalar).collect(),
                ));
                state.push((
                    "coarse w pin",
                    gpu.download(&pin.w)?.into_iter().map(|v| v as Scalar).collect(),
                ));
                state.push((
                    "coarse w fb",
                    gpu.download(&fb.w)?.into_iter().map(|v| v as Scalar).collect(),
                ));
                Ok(state)
            },
        )
        .expect("SPEC-LIT 81.7: the AMG coarse solve must capture and \
                 replay bitwise");
        println!("  AMG coarse: {report}");
    }
}
