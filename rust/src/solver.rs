// meteor-cfd - Copyright (c) 2026 주식회사 메테오시뮬레이션 (Meteo Simulation Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

//! Krylov linear solvers, with every control scalar resident on the device.
//!
//! Written from:
//!   Saad, *Iterative Methods for Sparse Linear Systems*, 2nd ed. (2003),
//!     §6.7 (PCG), §7.4.2 (PBiCGStab), ch. 10 and §12.4 (the multi-colour
//!     reordering that makes an incomplete factorisation parallel; the
//!     factorisation itself is in [`crate::precon`])
//!   van der Vorst, *SIAM J. Sci. Stat. Comput.* 13 (1992) 631-644
//!   Hestenes & Stiefel, *J. Res. Natl. Bur. Stand.* 49 (1952) 409
//!   ofgpu `SPEC-LIT.md` §8; §8.4 is marked *DESIGN* there and is ours.
//!   ofgpu `SPEC-LIT.md` §21 (multi-colour DIC/DILU) and §13.4 (a setting the
//!     solver cannot honour must fail loudly), added when `solvers/<var>/solver`
//!     and `preconditioner` stopped being parsed and discarded.
//! No GPL-licensed source was consulted.
//!
//! # Why every scalar lives on the device
//!
//! A textbook BiCGStab iteration reads like host code: `alpha = rho/(rTilde,
//! v)`, then `s = r - alpha*v`. Done literally on a GPU that is one
//! device-to-host copy and one synchronisation *per inner product*, four
//! times an iteration, and a solve that should be bandwidth-bound becomes
//! latency-bound instead.
//!
//! So none of it happens. `rho`, `alpha`, `omega`, `beta`, the normalisation
//! factor and the residuals are all one-element device buffers. The scalar
//! updates ([`solBetaBicg`](../../cuda/solver.cu), `solDivideScalar`) are
//! one-thread kernels that read those buffers and write them back, and the
//! vector updates take the same pointers and dereference them on the device.
//! An iteration is therefore a pure sequence of launches with no host
//! decision in it, which is the precondition for capturing a whole timestep
//! as a CUDA graph — see [`crate::device::Gpu::capture`].
//!
//! The one remaining host round-trip is the convergence test, and
//! [`SolverControls`] can switch it off two ways:
//!
//! * `check_interval` sets how often the (sticky) device flag is sampled;
//! * `fixed_iters` skips the test altogether and runs exactly `max_iter`
//!   sweeps;
//! * `report_residuals` off skips the end-of-solve read-back as well.
//!
//! `fixed_iters` and `report_residuals = false` together give a solve that
//! touches the host exactly zero times, which
//! `a_fixed_iteration_solve_captures_into_a_cuda_graph` in this file proves
//! by capturing one.
//!
//! # Preconditioners
//!
//! None, Jacobi, and the multi-colour `DIC`/`DILU` of [`crate::precon`]
//! (Saad §12.4, `SPEC-LIT` §21). `DIC` and `DILU` used to be accepted and
//! silently mapped onto Jacobi; they are now run. A preconditioner the
//! workspace cannot build is an error naming what was asked for, per
//! `SPEC-LIT` §13.4 - see [`effective_preconditioner`].
//!
//! # Which Krylov method
//!
//! `solvers/<var>/solver` used to be parsed and never read: every equation
//! got PBiCGStab, including one whose dictionary said `PCG` or `GAMG`.
//! [`solve`] honours the request. `PCG` on an asymmetric matrix is an
//! **error** and not a silent success - CG minimises over a Krylov space using
//! `A`-orthogonality, and on an asymmetric matrix that is not an inner
//! product, so the method does not converge slowly, it converges to the wrong
//! thing or not at all.

use cudarc::driver::{CudaEvent, CudaFunction, LaunchConfig, PinnedHostSlice, PushKernelArg};

use crate::device::{cfg_for, DevBuf, Gpu, BLOCK};
use crate::error::{Error, Result};
use crate::ldu::GpuLduMatrix;
use crate::mesh::GpuMesh;
use crate::precon::MultiColour;
use crate::{Label, Scalar};

/// Read from `system/fvSolution`; defined in [`crate::io::case`] because that
/// is where they are parsed, re-exported here because this is where they are
/// obeyed.
pub use crate::io::case::{LinearSolverKind, Preconditioner, SolverControls};

// ==========================================================================
//  Launch geometry
// ==========================================================================

/// Most partials a stage-one reduction may produce.
///
/// The reductions are grid-strided, so this caps the partial count instead of
/// letting it grow with the problem. Two things follow: the scratch buffer is
/// a fixed 1024 elements whatever the mesh, and one stage-two block can always
/// finish the job in a single launch.
const MAX_REDUCE_BLOCKS: usize = 1024;

/// The `eps` of `SPEC-LIT` §8.4.
///
/// It exists only to keep `res/norm` defined when the operator spans nothing
/// at all. The smallest normal number is the right choice because it cannot
/// perturb a real norm, and it cannot produce a spurious huge ratio either:
/// `norm == 0` forces `b == A·x_ref == A·psi`, so the numerator is exactly
/// zero at the same time.
const NORM_EPS: Scalar = Scalar::MIN_POSITIVE;

/// Grid for a reduction over `n` items, and the number of partials it writes.
fn reduce_geometry(n: usize) -> (LaunchConfig, usize) {
    let blocks = n.div_ceil(BLOCK as usize).clamp(1, MAX_REDUCE_BLOCKS);
    (
        LaunchConfig {
            grid_dim: (blocks as u32, 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        },
        blocks,
    )
}

/// How many partial sums [`device_sum`] and its siblings produce for `n`
/// values - i.e. how long a caller's `partials` buffer has to be.
///
/// Exposed because a module that owns its own reduction buffer (SPEC-LIT
/// §47.8's interface-flux total is one) must size it the same way, and
/// because the fact that this is a **pure function of `n`** is exactly what
/// makes the reduction order-independent and therefore bitwise reproducible.
pub fn reduce_partitions(n: usize) -> usize {
    reduce_geometry(n).1
}

/// One block: the second stage of every reduction.
fn one_block() -> LaunchConfig {
    LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// One thread: every scalar update.
fn one_thread() -> LaunchConfig {
    LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    }
}

fn to_label(n: usize) -> Result<Label> {
    Label::try_from(n)
        .map_err(|_| Error::Config(format!("solver: {n} does not fit in a label")))
}

// ==========================================================================
//  Kernels
// ==========================================================================

/// Every entry point in `cuda/solver.cu`, resolved once.
///
/// Public fields, because `src/pressure/mod.rs` builds a residual out of
/// [`Self::sub`] directly rather than going through a wrapper that would only
/// exist for it.
pub struct SolverKernels {
    // reductions, stage one
    pub sum1: CudaFunction,
    pub sum_mag1: CudaFunction,
    pub dot1: CudaFunction,
    pub dot2_1: CudaFunction,
    pub max_mag1: CudaFunction,
    pub norm_factor1: CudaFunction,
    pub sym_defect1: CudaFunction,
    /// SPEC-LIT §48.3: the coupled half of the same question.
    pub coupled_sym_defect1: CudaFunction,
    // reductions, stage two
    pub sum2: CudaFunction,
    pub sum2_pair: CudaFunction,
    pub max2: CudaFunction,
    pub max2_pair: CudaFunction,
    // vectors
    pub amul: CudaFunction,
    pub copy: CudaFunction,
    pub sub: CudaFunction,
    pub broadcast_scaled: CudaFunction,
    pub invert_diag: CudaFunction,
    pub precond_jacobi: CudaFunction,
    pub p_update: CudaFunction,
    pub s_update: CudaFunction,
    pub x_update: CudaFunction,
    pub r_update: CudaFunction,
    pub axpy: CudaFunction,
    pub axmy: CudaFunction,
    pub p_update_cg: CudaFunction,
    // scalars
    pub set_scalar: CudaFunction,
    pub copy_scalar: CudaFunction,
    pub divide_scalar: CudaFunction,
    pub beta_bicg: CudaFunction,
    pub convergence_test: CudaFunction,
    pub pack_report: CudaFunction,
}

impl SolverKernels {
    pub fn new(gpu: &Gpu) -> Result<Self> {
        let k = crate::device::KernelSet::new(gpu, crate::kernels::SOLVER)?;
        Ok(Self {
            sum1: k.func("solSumStage1")?,
            sum_mag1: k.func("solSumMagStage1")?,
            dot1: k.func("solDotStage1")?,
            dot2_1: k.func("solDot2Stage1")?,
            max_mag1: k.func("solMaxMagStage1")?,
            norm_factor1: k.func("solNormFactorStage1")?,
            sym_defect1: k.func("solSymDefectStage1")?,
            coupled_sym_defect1: k.func("solCoupledSymDefectStage1")?,

            sum2: k.func("solSumStage2")?,
            sum2_pair: k.func("solSum2Stage2")?,
            max2: k.func("solMaxStage2")?,
            max2_pair: k.func("solMax2Stage2")?,

            amul: k.func("solAmul")?,
            copy: k.func("solCopy")?,
            sub: k.func("solSub")?,
            broadcast_scaled: k.func("solBroadcastScaled")?,
            invert_diag: k.func("solInvertDiag")?,
            precond_jacobi: k.func("solPrecondJacobi")?,
            p_update: k.func("solPUpdate")?,
            s_update: k.func("solSUpdate")?,
            x_update: k.func("solXUpdate")?,
            r_update: k.func("solRUpdate")?,
            axpy: k.func("solAxpy")?,
            axmy: k.func("solAxmy")?,
            p_update_cg: k.func("solPUpdateCg")?,

            set_scalar: k.func("solSetScalar")?,
            copy_scalar: k.func("solCopyScalar")?,
            divide_scalar: k.func("solDivideScalar")?,
            beta_bicg: k.func("solBetaBicg")?,
            convergence_test: k.func("solConvergenceTest")?,
            pack_report: k.func("solPackReport")?,
        })
    }
}

// ==========================================================================
//  Workspace
// ==========================================================================

/// Everything a solve needs that is not the matrix or the solution.
///
/// Allocated once and reused: a time loop is not allowed to allocate, and a
/// CUDA graph cannot capture an allocation at all. Reusing it must not change
/// the answer, which `a_reused_workspace_gives_the_same_answer_twice` checks
/// bit for bit.
pub struct SolverWorkspace {
    /// Cells this workspace was sized for.
    pub n: usize,

    // ---- Krylov vectors --------------------------------------------------
    /// Residual `b - A·psi`, updated by the recurrence inside the loop.
    pub r: DevBuf<Scalar>,
    /// The shadow residual `rTilde` of van der Vorst (1992); the initial `r`.
    pub r0: DevBuf<Scalar>,
    pub p: DevBuf<Scalar>,
    pub v: DevBuf<Scalar>,
    pub s: DevBuf<Scalar>,
    pub t: DevBuf<Scalar>,
    /// `M^-1 p` (and, in PCG, `z = M^-1 r`).
    pub p_hat: DevBuf<Scalar>,
    /// `M^-1 s`.
    pub s_hat: DevBuf<Scalar>,

    // ---- shared scratch --------------------------------------------------
    /// `A·psi`. Also what [`device_norm_factor`] leaves behind.
    pub apsi: DevBuf<Scalar>,
    /// General scratch; the constant field `x_ref` during normalisation.
    pub tmp: DevBuf<Scalar>,
    /// General scratch; `A·x_ref` during normalisation.
    pub y: DevBuf<Scalar>,
    /// `1/diag(A)` for Jacobi, or `1/Dt` for the multi-colour incomplete
    /// factorisation. Rebuilt at each solve; exactly one of the two
    /// preconditioners is ever live, so they share the array.
    pub r_diag: DevBuf<Scalar>,

    /// The colouring and kernels `DIC`/`DILU` need, `SPEC-LIT` §21.
    ///
    /// Built by [`SolverWorkspace::for_mesh`] and `None` for a workspace made
    /// with [`SolverWorkspace::new`], which is not given a mesh and so has no
    /// graph to colour. A `DIC`/`DILU` request against such a workspace is an
    /// error rather than a quiet downgrade - see [`effective_preconditioner`].
    pub multicolour: Option<MultiColour>,

    // ---- reduction scratch -----------------------------------------------
    pub partials: DevBuf<Scalar>,
    /// Second partial array, for the fused `(t,s)`/`(t,t)` reduction.
    pub partials_b: DevBuf<Scalar>,

    // ---- device control scalars -----------------------------------------
    pub rho: DevBuf<Scalar>,
    pub rho_old: DevBuf<Scalar>,
    pub alpha: DevBuf<Scalar>,
    pub omega: DevBuf<Scalar>,
    pub beta: DevBuf<Scalar>,
    /// Numerator scratch: `(t,s)`.
    pub num: DevBuf<Scalar>,
    /// Denominator scratch: `(rTilde,v)`, then `(t,t)`, then `(p,q)`.
    pub den: DevBuf<Scalar>,
    /// `mean(psi)`, the `x_ref` of `SPEC-LIT` §8.4.
    pub x_ref: DevBuf<Scalar>,

    /// `SPEC-LIT` §8.4 normalisation factor. UNSCALED residuals are divided
    /// by this to give the reported number.
    pub norm_factor: DevBuf<Scalar>,
    /// `sum|b - A·psi|` before the first sweep. Unscaled.
    pub initial_res: DevBuf<Scalar>,
    /// `sum|b - A·psi|` at the end. Unscaled.
    pub final_res: DevBuf<Scalar>,

    /// Sticky convergence flag, written by the device, sampled by the host.
    pub flag: DevBuf<Label>,
    /// `[initial_res, final_res, norm_factor]`, so reporting costs one copy.
    pub report: DevBuf<Scalar>,

    /// Landing pad for [`Self::flag`]. Page-locked so the copy is a plain DMA
    /// with no staging buffer behind it.
    flag_host: PinnedHostSlice<Label>,

    /// The wait that makes that copy safe to read.
    ///
    /// A device-to-host copy into PAGE-LOCKED memory is genuinely
    /// asynchronous - unlike the pageable case, where the driver blocks until
    /// the transfer lands - so something has to wait for it. `cudarc` would
    /// normally record an event for us, but only when it is managing stream
    /// synchronisation, and [`crate::device::Gpu::new`] deliberately turns
    /// that off. So the event is ours: recorded after the copy, waited on
    /// before the read. Created with the default flags, i.e. spin rather than
    /// block, because this wait is on the critical path of the iteration and
    /// `check_interval` already exists to make it rare.
    flag_event: CudaEvent,
}

impl SolverWorkspace {
    /// A workspace for this mesh, including the colouring the multi-colour
    /// `DIC`/`DILU` of `SPEC-LIT` §21 needs.
    ///
    /// The colouring is built here rather than on first use because a time
    /// loop is not allowed to allocate and a CUDA graph cannot capture an
    /// allocation at all. It costs five device-to-host copies and an O(cells)
    /// greedy pass, once.
    pub fn for_mesh(gpu: &Gpu, m: &GpuMesh) -> Result<Self> {
        let mut w = Self::new(gpu, m.n_cells)?;
        if m.n_cells > 0 {
            w.multicolour = Some(MultiColour::new(gpu, m)?);
        }
        Ok(w)
    }

    pub fn new(gpu: &Gpu, n_cells: usize) -> Result<Self> {
        to_label(n_cells)?;

        // A zero-length device allocation is an error, not an empty buffer,
        // so degenerate meshes get one element they never read.
        let n = n_cells.max(1);

        // SAFETY: the single non-launch `unsafe` in this crate, and it is here
        // because cudarc offers no safe route to page-locked host memory.
        // `alloc_pinned` is unsafe for exactly one reason - it hands back
        // uninitialised memory - and the very next statement overwrites the
        // one element it contains. The allocation is freed by
        // `PinnedHostSlice`'s own `Drop`, and the slice is never aliased: it
        // is private to this struct and only ever written by `read_flag`.
        let mut flag_host: PinnedHostSlice<Label> =
            unsafe { gpu.ctx().alloc_pinned::<Label>(1)? };
        flag_host.as_mut_slice()?[0] = 0;

        Ok(Self {
            n: n_cells,

            r: gpu.zeros(n)?,
            r0: gpu.zeros(n)?,
            p: gpu.zeros(n)?,
            v: gpu.zeros(n)?,
            s: gpu.zeros(n)?,
            t: gpu.zeros(n)?,
            p_hat: gpu.zeros(n)?,
            s_hat: gpu.zeros(n)?,

            apsi: gpu.zeros(n)?,
            tmp: gpu.zeros(n)?,
            y: gpu.zeros(n)?,
            r_diag: gpu.zeros(n)?,
            multicolour: None,

            partials: gpu.zeros(MAX_REDUCE_BLOCKS)?,
            partials_b: gpu.zeros(MAX_REDUCE_BLOCKS)?,

            rho: gpu.zeros(1)?,
            rho_old: gpu.zeros(1)?,
            alpha: gpu.zeros(1)?,
            omega: gpu.zeros(1)?,
            beta: gpu.zeros(1)?,
            num: gpu.zeros(1)?,
            den: gpu.zeros(1)?,
            x_ref: gpu.zeros(1)?,

            norm_factor: gpu.zeros(1)?,
            initial_res: gpu.zeros(1)?,
            final_res: gpu.zeros(1)?,

            flag: gpu.zeros(1)?,
            report: gpu.zeros(3)?,

            flag_host,
            flag_event: gpu.ctx().new_event(None)?,
        })
    }
}

/// What a solve did.
///
/// The residuals are the *scaled* ones - `sum|b - A·psi| / normFactor`, the
/// `SPEC-LIT` §8.4 measure - and are left at zero when `report_residuals` is
/// off, because in that mode the host is never told.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct SolverPerformance {
    pub initial_residual: Scalar,
    pub final_residual: Scalar,
    pub n_iterations: usize,
    /// True only when the stopping criterion was *observed* to be met. A
    /// `fixed_iters` solve with `report_residuals` off never looks, so it
    /// reports `false` rather than guessing.
    pub converged: bool,
}

// ==========================================================================
//  Reductions
//
//  Each is two launches: n values -> at most MAX_REDUCE_BLOCKS partials ->
//  one DEVICE scalar. The result never touches the host.
// ==========================================================================

/// `out = sum(x)`.
pub fn device_sum(
    gpu: &Gpu,
    k: &SolverKernels,
    out: &mut DevBuf<Scalar>,
    x: &DevBuf<Scalar>,
    partials: &mut DevBuf<Scalar>,
    n: usize,
) -> Result<()> {
    // A zero-block grid is an illegal launch configuration, so an empty
    // reduction is answered with a memset instead of a kernel.
    if n == 0 {
        return gpu.fill_zero(out);
    }
    let nl = to_label(n)?;
    let (cfg, nparts) = reduce_geometry(n);

    unsafe {
        gpu.stream()
            .launch_builder(&k.sum1)
            .arg(&mut *partials)
            .arg(x)
            .arg(&nl)
            .launch(cfg)?;
    }
    finish_sum(gpu, k, out, partials, nparts, 0.0)
}

/// `out = sum|x|` - the residual measure of `SPEC-LIT` §8.4.
pub fn device_sum_mag(
    gpu: &Gpu,
    k: &SolverKernels,
    out: &mut DevBuf<Scalar>,
    x: &DevBuf<Scalar>,
    partials: &mut DevBuf<Scalar>,
    n: usize,
) -> Result<()> {
    if n == 0 {
        return gpu.fill_zero(out);
    }
    let nl = to_label(n)?;
    let (cfg, nparts) = reduce_geometry(n);

    unsafe {
        gpu.stream()
            .launch_builder(&k.sum_mag1)
            .arg(&mut *partials)
            .arg(x)
            .arg(&nl)
            .launch(cfg)?;
    }
    finish_sum(gpu, k, out, partials, nparts, 0.0)
}

/// `out = (a,b)`.
pub fn device_dot(
    gpu: &Gpu,
    k: &SolverKernels,
    out: &mut DevBuf<Scalar>,
    a: &DevBuf<Scalar>,
    b: &DevBuf<Scalar>,
    partials: &mut DevBuf<Scalar>,
    n: usize,
) -> Result<()> {
    if n == 0 {
        return gpu.fill_zero(out);
    }
    let nl = to_label(n)?;
    let (cfg, nparts) = reduce_geometry(n);

    unsafe {
        gpu.stream()
            .launch_builder(&k.dot1)
            .arg(&mut *partials)
            .arg(a)
            .arg(b)
            .arg(&nl)
            .launch(cfg)?;
    }
    finish_sum(gpu, k, out, partials, nparts, 0.0)
}

/// `ab = (a,b)` and `aa = (a,a)` in one pass over `a`.
///
/// BiCGStab wants `(t,s)` and `(t,t)` at the same moment; doing them together
/// saves two launches and one re-read of `t` every iteration.
#[allow(clippy::too_many_arguments)]
pub fn device_dot2(
    gpu: &Gpu,
    k: &SolverKernels,
    ab: &mut DevBuf<Scalar>,
    aa: &mut DevBuf<Scalar>,
    a: &DevBuf<Scalar>,
    b: &DevBuf<Scalar>,
    partials_ab: &mut DevBuf<Scalar>,
    partials_aa: &mut DevBuf<Scalar>,
    n: usize,
) -> Result<()> {
    if n == 0 {
        gpu.fill_zero(ab)?;
        return gpu.fill_zero(aa);
    }
    let nl = to_label(n)?;
    let (cfg, nparts) = reduce_geometry(n);
    let np = to_label(nparts)?;

    unsafe {
        gpu.stream()
            .launch_builder(&k.dot2_1)
            .arg(&mut *partials_ab)
            .arg(&mut *partials_aa)
            .arg(a)
            .arg(b)
            .arg(&nl)
            .launch(cfg)?;

        gpu.stream()
            .launch_builder(&k.sum2_pair)
            .arg(&mut *ab)
            .arg(&mut *aa)
            .arg(&*partials_ab)
            .arg(&*partials_aa)
            .arg(&np)
            .launch(one_block())?;
    }
    Ok(())
}

/// `out = max|x|`.
///
/// The maximum is taken over magnitudes because that is the only form a norm
/// ever wants; a signed maximum would need a different identity element and
/// has no caller here.
pub fn device_max_mag(
    gpu: &Gpu,
    k: &SolverKernels,
    out: &mut DevBuf<Scalar>,
    x: &DevBuf<Scalar>,
    partials: &mut DevBuf<Scalar>,
    n: usize,
) -> Result<()> {
    if n == 0 {
        return gpu.fill_zero(out);
    }
    let nl = to_label(n)?;
    let (cfg, nparts) = reduce_geometry(n);
    let np = to_label(nparts)?;

    unsafe {
        gpu.stream()
            .launch_builder(&k.max_mag1)
            .arg(&mut *partials)
            .arg(x)
            .arg(&nl)
            .launch(cfg)?;

        gpu.stream()
            .launch_builder(&k.max2)
            .arg(&mut *out)
            .arg(&*partials)
            .arg(&np)
            .launch(one_block())?;
    }
    Ok(())
}

fn finish_sum(
    gpu: &Gpu,
    k: &SolverKernels,
    out: &mut DevBuf<Scalar>,
    partials: &DevBuf<Scalar>,
    nparts: usize,
    offset: Scalar,
) -> Result<()> {
    let np = to_label(nparts)?;
    unsafe {
        gpu.stream()
            .launch_builder(&k.sum2)
            .arg(&mut *out)
            .arg(partials)
            .arg(&np)
            .arg(&offset)
            .launch(one_block())?;
    }
    Ok(())
}

// ==========================================================================
//  Matrix and vector primitives
// ==========================================================================

/// `y = A·psi`, in the LDU form `src/ldu.rs` fixes.
///
/// A gather over the merged row map - one thread per cell walking its own
/// faces - so there are no atomics on `f64`, the summation order per row is
/// fixed and the product is bitwise reproducible. Cyclic boundary faces are
/// the only boundary faces that reach the matrix; everything else has already
/// been folded into `diag` and `source`.
///
/// SPEC-LIT §70: the map is ordered by the GLOBAL face id, so the order is a
/// property of the mesh rather than of how the mesh was cut up. This is the
/// product every Krylov iteration calls; `ldu_ops::amul` is the other
/// implementation of the same row sum and walks the same map.
pub fn amul(
    gpu: &Gpu,
    k: &SolverKernels,
    y: &mut DevBuf<Scalar>,
    psi: &DevBuf<Scalar>,
    a: &GpuLduMatrix,
    m: &GpuMesh,
) -> Result<()> {
    let n = a.n_cells;
    if n == 0 {
        return Ok(());
    }
    if m.n_cells != n {
        return Err(Error::Config(format!(
            "amul: matrix has {n} cells, mesh has {}",
            m.n_cells
        )));
    }
    let nl = to_label(n)?;

    unsafe {
        gpu.stream()
            .launch_builder(&k.amul)
            .arg(&mut *y)
            .arg(psi)
            .arg(&a.diag)
            .arg(&a.upper)
            .arg(&a.lower)
            .arg(&m.owner)
            .arg(&m.neighbour)
            .arg(&m.rf_offset)
            .arg(&m.rf_face)
            .arg(&m.rf_flags)
            .arg(&a.boundary_coeffs)
            .arg(&m.b_nbr_cell)
            .arg(&nl)
            .launch(cfg_for(n))?;
    }
    Ok(())
}

fn vec_copy(
    gpu: &Gpu,
    k: &SolverKernels,
    dst: &mut DevBuf<Scalar>,
    src: &DevBuf<Scalar>,
    n: usize,
) -> Result<()> {
    if n == 0 {
        return Ok(());
    }
    let nl = to_label(n)?;
    unsafe {
        gpu.stream()
            .launch_builder(&k.copy)
            .arg(&mut *dst)
            .arg(src)
            .arg(&nl)
            .launch(cfg_for(n))?;
    }
    Ok(())
}

/// `out = a - b`.
pub fn vec_sub(
    gpu: &Gpu,
    k: &SolverKernels,
    out: &mut DevBuf<Scalar>,
    a: &DevBuf<Scalar>,
    b: &DevBuf<Scalar>,
    n: usize,
) -> Result<()> {
    if n == 0 {
        return Ok(());
    }
    let nl = to_label(n)?;
    unsafe {
        gpu.stream()
            .launch_builder(&k.sub)
            .arg(&mut *out)
            .arg(a)
            .arg(b)
            .arg(&nl)
            .launch(cfg_for(n))?;
    }
    Ok(())
}

fn set_scalar(
    gpu: &Gpu,
    k: &SolverKernels,
    dst: &mut DevBuf<Scalar>,
    value: Scalar,
) -> Result<()> {
    unsafe {
        gpu.stream()
            .launch_builder(&k.set_scalar)
            .arg(&mut *dst)
            .arg(&value)
            .launch(one_thread())?;
    }
    Ok(())
}

fn copy_scalar(
    gpu: &Gpu,
    k: &SolverKernels,
    dst: &mut DevBuf<Scalar>,
    src: &DevBuf<Scalar>,
) -> Result<()> {
    unsafe {
        gpu.stream()
            .launch_builder(&k.copy_scalar)
            .arg(&mut *dst)
            .arg(src)
            .launch(one_thread())?;
    }
    Ok(())
}

/// `q = num/den`, both operands and the result device-resident. Guarded
/// against a zero denominator on the device; see `cuda/solver.cu`.
fn divide_scalar(
    gpu: &Gpu,
    k: &SolverKernels,
    q: &mut DevBuf<Scalar>,
    num: &DevBuf<Scalar>,
    den: &DevBuf<Scalar>,
) -> Result<()> {
    unsafe {
        gpu.stream()
            .launch_builder(&k.divide_scalar)
            .arg(&mut *q)
            .arg(num)
            .arg(den)
            .launch(one_thread())?;
    }
    Ok(())
}

// ==========================================================================
//  Matrix symmetry
// ==========================================================================

/// Is `upper == lower` on every internal face, to round-off?
///
/// Two device reductions and one small copy back. It exists because
/// `SPEC-LIT` §8.2 restricts PCG to symmetric positive definite systems and
/// §13.4 says a request the solver cannot honour must fail loudly. Called only
/// where the answer changes what happens - [`solve`] when `PCG` or `DIC` was
/// asked for - so a run that uses neither pays nothing and stays capturable
/// into a CUDA graph.
///
/// Uses `w.partials`/`w.partials_b` and `w.num`/`w.den` as scratch and touches
/// nothing else, so it is safe to call before a solve begins.
pub fn matrix_is_symmetric(
    gpu: &Gpu,
    k: &SolverKernels,
    w: &mut SolverWorkspace,
    a: &GpuLduMatrix,
    m: &GpuMesh,
) -> Result<bool> {
    Ok(symmetry_defects(gpu, k, w, a, m)?.is_symmetric())
}

/// The two ways an LDU matrix can fail to be symmetric, measured separately -
/// SPEC-LIT §48.3.
///
/// Separately, because a failure has to name which one it was: "upper != lower"
/// is a bug in a face term and "the two coupled coefficients differ" is a bug
/// in a boundary term, and they are found in completely different places.
#[derive(Debug, Clone, Copy, Default)]
pub struct SymmetryDefects {
    /// `max |upper - lower|` over the internal faces.
    pub face: Scalar,
    /// `max |upper|, |lower|`, the scale `face` is judged against.
    pub face_scale: Scalar,
    /// `max |boundary_coeffs[bf] - boundary_coeffs[pair(bf)]|` over the
    /// COUPLED boundary faces - the half `matrix_is_symmetric` was blind to
    /// before §48.3.
    pub coupled: Scalar,
    pub coupled_scale: Scalar,
}

impl SymmetryDefects {
    /// The same threshold `pressure::probe` uses on the host, and for the same
    /// reason: "these two floats came out of the same arithmetic", scaled off
    /// the working epsilon so a single-precision build means the same thing by
    /// it rather than rejecting every matrix.
    pub fn tolerance() -> Scalar {
        1.0e3 * Scalar::EPSILON
    }

    pub fn face_is_symmetric(&self) -> bool {
        self.face_scale == 0.0 || self.face <= Self::tolerance() * self.face_scale
    }

    pub fn coupled_is_symmetric(&self) -> bool {
        self.coupled_scale == 0.0 || self.coupled <= Self::tolerance() * self.coupled_scale
    }

    pub fn is_symmetric(&self) -> bool {
        self.face_is_symmetric() && self.coupled_is_symmetric()
    }

    /// Which half failed, for the diagnostic a refusal carries.
    pub fn what_failed(&self) -> &'static str {
        match (self.face_is_symmetric(), self.coupled_is_symmetric()) {
            (false, false) => "both the face coefficients and the coupled boundary pair",
            (false, true) => "the face coefficients (upper != lower)",
            (true, false) => {
                "the COUPLED BOUNDARY pair - the two faces of one couple carry \
                 different coefficients, so A(P,Q) != A(Q,P)"
            }
            (true, true) => "nothing",
        }
    }
}

/// Measure both halves of [`matrix_is_symmetric`]'s question.
pub fn symmetry_defects(
    gpu: &Gpu,
    k: &SolverKernels,
    w: &mut SolverWorkspace,
    a: &GpuLduMatrix,
    m: &GpuMesh,
) -> Result<SymmetryDefects> {
    let mut out = SymmetryDefects::default();

    let nf = a.n_internal_faces;
    if nf > 0 {
        let nl = to_label(nf)?;
        let (cfg, nparts) = reduce_geometry(nf);
        let np = to_label(nparts)?;

        unsafe {
            gpu.stream()
                .launch_builder(&k.sym_defect1)
                .arg(&mut w.partials)
                .arg(&mut w.partials_b)
                .arg(&a.upper)
                .arg(&a.lower)
                .arg(&nl)
                .launch(cfg)?;

            gpu.stream()
                .launch_builder(&k.max2_pair)
                .arg(&mut w.num)
                .arg(&mut w.den)
                .arg(&w.partials)
                .arg(&w.partials_b)
                .arg(&np)
                .launch(one_block())?;
        }

        out.face = gpu.download(&w.num)?.first().copied().unwrap_or(0.0);
        out.face_scale = gpu.download(&w.den)?.first().copied().unwrap_or(0.0);
    }

    // SPEC-LIT §48.3. `b_nbr_face` is `-1` everywhere on a mesh with no
    // coupled patch, so this stage measures zero there and the result is
    // exactly what it was before §48.3 - which is the "no false positives"
    // half of §48.4.
    let nbf = a.n_boundary_faces;
    if nbf > 0 && m.n_boundary_faces == nbf {
        let nl = to_label(nbf)?;
        let (cfg, nparts) = reduce_geometry(nbf);
        let np = to_label(nparts)?;

        unsafe {
            gpu.stream()
                .launch_builder(&k.coupled_sym_defect1)
                .arg(&mut w.partials)
                .arg(&mut w.partials_b)
                .arg(&a.boundary_coeffs)
                .arg(&m.b_nbr_cell)
                .arg(&m.b_nbr_face)
                .arg(&nl)
                .launch(cfg)?;

            gpu.stream()
                .launch_builder(&k.max2_pair)
                .arg(&mut w.num)
                .arg(&mut w.den)
                .arg(&w.partials)
                .arg(&w.partials_b)
                .arg(&np)
                .launch(one_block())?;
        }

        out.coupled = gpu.download(&w.num)?.first().copied().unwrap_or(0.0);
        out.coupled_scale = gpu.download(&w.den)?.first().copied().unwrap_or(0.0);
    }

    Ok(out)
}

// ==========================================================================
//  Preconditioning
// ==========================================================================

/// What actually runs, given what the case asked for.
///
/// `SPEC-LIT` §13.4, applied to preconditioners:
///
/// ```text
/// recognised and implemented   -> use it
/// recognised, not implemented  -> Error naming the setting
/// ```
///
/// `DIC` and `DILU` are implemented (`SPEC-LIT` §21) and need a colouring,
/// which only a workspace built by [`SolverWorkspace::for_mesh`] has. Asking
/// for one against a mesh-free workspace is the second line of that table, so
/// it is an error naming the setting rather than a silent Jacobi - which is
/// exactly what this function used to do for every non-`none` request.
pub fn effective_preconditioner(
    requested: Preconditioner,
    w: &SolverWorkspace,
) -> Result<Preconditioner> {
    match requested {
        Preconditioner::None | Preconditioner::Diagonal => Ok(requested),
        Preconditioner::Dic | Preconditioner::Dilu => {
            if w.multicolour.is_some() {
                Ok(requested)
            } else {
                // Recognised, implemented, but not available on THIS
                // workspace. Same rule, same escape hatch, same wording as
                // every other setting - SPEC-LIT 13.4.
                crate::io::contract::unsupported(
                    "solvers/<var>/preconditioner",
                    requested.name(),
                    &["none", "diagonal"],
                    "diagonal (Jacobi): this solver workspace was built \
                     without a mesh, so there is no LDU graph to colour",
                    Preconditioner::Diagonal,
                )
            }
        }
    }
}

/// Build whichever preconditioner `precon` names.
///
/// Jacobi is one kernel; the incomplete factorisations are one kernel per
/// colour, in ascending colour order (`SPEC-LIT` §21 step 1).
pub fn build_preconditioner(
    gpu: &Gpu,
    k: &SolverKernels,
    w: &mut SolverWorkspace,
    a: &GpuLduMatrix,
    m: &GpuMesh,
    precon: Preconditioner,
) -> Result<()> {
    let n = a.n_cells;
    if n == 0 || precon == Preconditioner::None {
        return Ok(());
    }

    if precon == Preconditioner::Diagonal {
        let nl = to_label(n)?;
        unsafe {
            gpu.stream()
                .launch_builder(&k.invert_diag)
                .arg(&mut w.r_diag)
                .arg(&a.diag)
                .arg(&nl)
                .launch(cfg_for(n))?;
        }
        return Ok(());
    }

    // Disjoint borrows: the factorisation writes `r_diag` and reads the
    // colouring, and they are different fields of the same workspace.
    let SolverWorkspace { r_diag, multicolour, .. } = w;
    let Some(mc) = multicolour.as_ref() else {
        return Err(Error::Config(
            "build_preconditioner: DIC/DILU was selected without a colouring; \
             effective_preconditioner should have rejected this already"
                .to_string(),
        ));
    };
    mc.factorise(gpu, r_diag, a, m, precon == Preconditioner::Dic)
}

/// `y = M^-1 x`. With no preconditioner this is a copy, which keeps the two
/// solvers branch-free at the cost of one bandwidth-bound pass.
#[allow(clippy::too_many_arguments)]
fn precondition_parts(
    gpu: &Gpu,
    k: &SolverKernels,
    y: &mut DevBuf<Scalar>,
    x: &DevBuf<Scalar>,
    r_diag: &DevBuf<Scalar>,
    multicolour: Option<&MultiColour>,
    a: &GpuLduMatrix,
    m: &GpuMesh,
    precon: Preconditioner,
    n: usize,
) -> Result<()> {
    if n == 0 {
        return Ok(());
    }
    match precon {
        Preconditioner::None => vec_copy(gpu, k, y, x, n),

        Preconditioner::Diagonal => {
            let nl = to_label(n)?;
            unsafe {
                gpu.stream()
                    .launch_builder(&k.precond_jacobi)
                    .arg(&mut *y)
                    .arg(x)
                    .arg(r_diag)
                    .arg(&nl)
                    .launch(cfg_for(n))?;
            }
            Ok(())
        }

        Preconditioner::Dic | Preconditioner::Dilu => {
            let Some(mc) = multicolour else {
                return Err(Error::Config(
                    "precondition: DIC/DILU was selected without a colouring"
                        .to_string(),
                ));
            };
            // The sweeps run in place, so `x` is copied in first. `y` and `x`
            // are never the same buffer at any call site in this file.
            vec_copy(gpu, k, y, x, n)?;
            mc.apply(gpu, y, r_diag, a, m)
        }
    }
}

/// Which of the workspace's three preconditioned vectors a call is filling.
///
/// `precondition` needs `&mut` on one workspace vector and `&` on another, and
/// the borrow checker will not split a `&mut SolverWorkspace` for us. Naming
/// the pair as an enum and doing the split here keeps that ugliness in one
/// place instead of at five call sites.
#[derive(Clone, Copy)]
enum PreconTarget {
    /// `pHat = M^-1 p`   (BiCGStab)
    PHat,
    /// `sHat = M^-1 s`   (BiCGStab)
    SHat,
    /// `z = M^-1 r`, held in `pHat`   (PCG)
    PHatFromR,
}

#[allow(clippy::too_many_arguments)]
fn precondition_ws(
    gpu: &Gpu,
    k: &SolverKernels,
    w: &mut SolverWorkspace,
    target: PreconTarget,
    a: &GpuLduMatrix,
    m: &GpuMesh,
    precon: Preconditioner,
    n: usize,
) -> Result<()> {
    // Split the workspace into the destination, the source, and the read-only
    // rest that `precondition` needs (`r_diag` and the colouring).
    let SolverWorkspace {
        r,
        p,
        s,
        p_hat,
        s_hat,
        r_diag,
        multicolour,
        ..
    } = w;

    let (dst, src): (&mut DevBuf<Scalar>, &DevBuf<Scalar>) = match target {
        PreconTarget::PHat => (p_hat, p),
        PreconTarget::SHat => (s_hat, s),
        PreconTarget::PHatFromR => (p_hat, r),
    };

    precondition_parts(gpu, k, dst, src, r_diag, multicolour.as_ref(), a, m, precon, n)
}

// ==========================================================================
//  Residual normalisation - SPEC-LIT section 8.4, *DESIGN*
// ==========================================================================

/// Compute the `SPEC-LIT` §8.4 normalisation factor into `w.norm_factor`.
///
/// ***DESIGN.*** This normalisation is ours, not something the literature
/// prescribes. A bare `sum|b - A·psi|` is meaningless across meshes and
/// scalings, so it is measured against the range the operator spans on this
/// particular problem:
///
/// ```text
/// x_ref = mean(psi)
/// norm  = sum|A·psi - A·x_ref| + sum|b - A·x_ref| + eps
/// res   = sum|b - A·psi| / norm
/// ```
///
/// `A·x_ref` is the operator applied to the *constant* field `x_ref`, which
/// on a pure-Neumann system is nearly the null vector - exactly the case
/// where an absolute residual says nothing.
///
/// Clobbers `w.apsi` (left holding `A·psi`), `w.tmp` and `w.y`. It does not
/// touch `w.r`, which is what lets `pressure::residual_norm` form the
/// residual first and normalise it afterwards.
pub fn device_norm_factor(
    gpu: &Gpu,
    k: &SolverKernels,
    w: &mut SolverWorkspace,
    psi: &DevBuf<Scalar>,
    a: &GpuLduMatrix,
    m: &GpuMesh,
) -> Result<()> {
    let n = a.n_cells;
    if n == 0 {
        return gpu.fill_zero(&mut w.norm_factor);
    }
    check_workspace(w, n)?;
    let nl = to_label(n)?;

    // x_ref = mean(psi). 1/n is a property of the mesh, not of the solution,
    // so forming it on the host breaks no rule.
    device_sum(gpu, k, &mut w.x_ref, psi, &mut w.partials, n)?;
    let inv_n = 1.0 / (n as Scalar);
    unsafe {
        gpu.stream()
            .launch_builder(&k.broadcast_scaled)
            .arg(&mut w.tmp)
            .arg(&w.x_ref)
            .arg(&inv_n)
            .arg(&nl)
            .launch(cfg_for(n))?;
    }

    amul(gpu, k, &mut w.y, &w.tmp, a, m)?;
    amul(gpu, k, &mut w.apsi, psi, a, m)?;

    let (cfg, nparts) = reduce_geometry(n);
    unsafe {
        gpu.stream()
            .launch_builder(&k.norm_factor1)
            .arg(&mut w.partials)
            .arg(&w.apsi)
            .arg(&a.source)
            .arg(&w.y)
            .arg(&nl)
            .launch(cfg)?;
    }
    finish_sum(gpu, k, &mut w.norm_factor, &w.partials, nparts, NORM_EPS)
}

// ==========================================================================
//  Convergence
// ==========================================================================

/// Ask the device whether the solve is finished, and record it in the sticky
/// flag. No host arithmetic: the tolerance is multiplied by the norm factor on
/// the device rather than the residual being divided by it.
#[allow(clippy::too_many_arguments)]
fn convergence_test(
    gpu: &Gpu,
    k: &SolverKernels,
    flag: &mut DevBuf<Label>,
    res: &DevBuf<Scalar>,
    res0: &DevBuf<Scalar>,
    norm_factor: &DevBuf<Scalar>,
    ctrl: &SolverControls,
    iter: Label,
) -> Result<()> {
    let tol = ctrl.tolerance;
    let rel = ctrl.rel_tol;
    let min_iter = ctrl.min_iter;
    unsafe {
        gpu.stream()
            .launch_builder(&k.convergence_test)
            .arg(&mut *flag)
            .arg(res)
            .arg(res0)
            .arg(norm_factor)
            .arg(&tol)
            .arg(&rel)
            .arg(&iter)
            .arg(&min_iter)
            .launch(one_thread())?;
    }
    Ok(())
}

/// The only host round-trip in a solve, and the only one `fixed_iters`
/// removes. Page-locked destination, and the wait is on the copy's own event
/// rather than on the whole stream.
///
/// The explicit `record`/`synchronize` is not optional: a D2H copy into pinned
/// memory returns immediately, and without the wait this reads whatever was in
/// the landing pad last time - which is a *plausible* value, so the bug shows
/// up as a solve that stops one check too early or too late rather than as a
/// crash. See the note on [`SolverWorkspace::flag_event`].
fn read_flag(
    gpu: &Gpu,
    flag: &DevBuf<Label>,
    host: &mut PinnedHostSlice<Label>,
    done: &CudaEvent,
) -> Result<bool> {
    gpu.stream().memcpy_dtoh(flag, host)?;
    done.record(gpu.stream())?;
    done.synchronize()?;
    Ok(host.as_slice()?.first().copied().unwrap_or(0) != 0)
}

fn check_workspace(w: &SolverWorkspace, n: usize) -> Result<()> {
    if w.n < n {
        return Err(Error::Config(format!(
            "solver: workspace is sized for {} cells, the system has {n}",
            w.n
        )));
    }
    Ok(())
}

/// Turn the three device scalars into the reported pair, in one copy.
fn collect_report(
    gpu: &Gpu,
    k: &SolverKernels,
    w: &mut SolverWorkspace,
    perf: &mut SolverPerformance,
) -> Result<()> {
    unsafe {
        gpu.stream()
            .launch_builder(&k.pack_report)
            .arg(&mut w.report)
            .arg(&w.initial_res)
            .arg(&w.final_res)
            .arg(&w.norm_factor)
            .launch(one_thread())?;
    }
    let v = gpu.download(&w.report)?;
    let r0 = v.first().copied().unwrap_or(0.0);
    let rf = v.get(1).copied().unwrap_or(0.0);
    let nf = v.get(2).copied().unwrap_or(1.0);

    let inv = if nf > 0.0 { 1.0 / nf } else { 1.0 };
    perf.initial_residual = r0 * inv;
    perf.final_residual = rf * inv;
    Ok(())
}

// ==========================================================================
//  Preconditioned BiCGStab
// ==========================================================================

/// Preconditioned BiCGStab.
///
/// van der Vorst, *SIAM J. Sci. Stat. Comput.* 13 (1992) 631-644, in the form
/// Saad gives as Algorithm 7.7 (§7.4.2):
///
/// ```text
/// r  = b - A·x ;  rTilde = r ;  rho = alpha = omega = 1 ;  p = v = 0
/// repeat
///     rho    = (rTilde, r)
///     beta   = (rho/rho_old)·(alpha/omega)
///     p      = r + beta·(p - omega·v)
///     pHat   = M^-1 p
///     v      = A·pHat
///     alpha  = rho / (rTilde, v)
///     s      = r - alpha·v
///     sHat   = M^-1 s
///     t      = A·sHat
///     omega  = (t,s)/(t,t)
///     x     += alpha·pHat + omega·sHat
///     r      = s - omega·t
/// ```
///
/// Handles the asymmetric systems convection produces, which is why it is the
/// default for everything and the fallback for the pressure equation.
///
/// Every scalar above is a device buffer. The `s`-norm early exit of the
/// textbook algorithm is deliberately absent: it would need a host decision
/// mid-iteration, and it is unnecessary because `s == 0` gives `t == 0`,
/// hence a guarded `omega = 0`, hence `r = s = 0` and an exact answer half an
/// iteration later.
///
/// The in-loop test uses the *recurrence* residual, which is what the
/// algorithm carries anyway; the residual finally reported is recomputed as
/// the true `b - A·x` so that a solve which stagnated cannot claim otherwise.
pub fn solve_pbicgstab(
    gpu: &Gpu,
    k: &SolverKernels,
    psi: &mut DevBuf<Scalar>,
    a: &GpuLduMatrix,
    m: &GpuMesh,
    w: &mut SolverWorkspace,
    ctrl: &SolverControls,
) -> Result<SolverPerformance> {
    let n = a.n_cells;
    let mut perf = SolverPerformance {
        converged: true,
        ..Default::default()
    };
    if n == 0 {
        return Ok(perf);
    }
    perf.converged = false;

    check_workspace(w, n)?;
    if psi.len() < n {
        return Err(Error::Config(format!(
            "solve_pbicgstab: psi holds {} values, the system has {n} cells",
            psi.len()
        )));
    }

    let precon = effective_preconditioner(ctrl.precon, w)?;
    build_preconditioner(gpu, k, w, a, m, precon)?;

    // ---- r = b - A·psi, and the normalisation the residual is measured in.
    // device_norm_factor leaves A·psi in w.apsi, so the residual is one
    // subtraction rather than a second matrix product.
    device_norm_factor(gpu, k, w, &*psi, a, m)?;
    vec_sub(gpu, k, &mut w.r, &a.source, &w.apsi, n)?;
    vec_copy(gpu, k, &mut w.r0, &w.r, n)?;

    device_sum_mag(gpu, k, &mut w.initial_res, &w.r, &mut w.partials, n)?;
    // Report honestly if the loop never runs.
    copy_scalar(gpu, k, &mut w.final_res, &w.initial_res)?;

    gpu.fill_zero(&mut w.p)?;
    gpu.fill_zero(&mut w.v)?;
    gpu.fill_zero(&mut w.flag)?;
    set_scalar(gpu, k, &mut w.rho_old, 1.0)?;
    set_scalar(gpu, k, &mut w.alpha, 1.0)?;
    set_scalar(gpu, k, &mut w.omega, 1.0)?;

    let max_iter = ctrl.max_iter.max(0) as usize;
    let interval = ctrl.check_interval.max(1) as usize;
    let checking = !ctrl.fixed_iters;

    // An already-converged system must not be iterated: that is where a
    // pressure equation spends most of a steady run.
    if checking {
        convergence_test(
            gpu, k, &mut w.flag, &w.initial_res, &w.initial_res, &w.norm_factor, ctrl, 0,
        )?;
        perf.converged = read_flag(gpu, &w.flag, &mut w.flag_host, &w.flag_event)?;
    }

    if !perf.converged {
        for it in 0..max_iter {
            let iters = it + 1;

            device_dot(gpu, k, &mut w.rho, &w.r0, &w.r, &mut w.partials, n)?;

            // beta = (rho/rho_old)·(alpha/omega)
            unsafe {
                gpu.stream()
                    .launch_builder(&k.beta_bicg)
                    .arg(&mut w.beta)
                    .arg(&w.rho)
                    .arg(&w.rho_old)
                    .arg(&w.alpha)
                    .arg(&w.omega)
                    .launch(one_thread())?;
            }

            bicg_p_update(gpu, k, w, n)?;
            precondition_ws(gpu, k, w, PreconTarget::PHat, a, m, precon, n)?;
            amul(gpu, k, &mut w.v, &w.p_hat, a, m)?;

            device_dot(gpu, k, &mut w.den, &w.r0, &w.v, &mut w.partials, n)?;
            divide_scalar(gpu, k, &mut w.alpha, &w.rho, &w.den)?;

            bicg_s_update(gpu, k, w, n)?;
            precondition_ws(gpu, k, w, PreconTarget::SHat, a, m, precon, n)?;
            amul(gpu, k, &mut w.t, &w.s_hat, a, m)?;

            // (t,s) and (t,t) in one pass, then omega = (t,s)/(t,t).
            device_dot2(
                gpu,
                k,
                &mut w.num,
                &mut w.den,
                &w.t,
                &w.s,
                &mut w.partials,
                &mut w.partials_b,
                n,
            )?;
            divide_scalar(gpu, k, &mut w.omega, &w.num, &w.den)?;

            bicg_x_update(gpu, k, psi, w, n)?;
            bicg_r_update(gpu, k, w, n)?;
            copy_scalar(gpu, k, &mut w.rho_old, &w.rho)?;

            perf.n_iterations = iters;

            if checking && iters % interval == 0 {
                device_sum_mag(gpu, k, &mut w.final_res, &w.r, &mut w.partials, n)?;
                let itl = to_label(iters)?;
                convergence_test(
                    gpu,
                    k,
                    &mut w.flag,
                    &w.final_res,
                    &w.initial_res,
                    &w.norm_factor,
                    ctrl,
                    itl,
                )?;
                if read_flag(gpu, &w.flag, &mut w.flag_host, &w.flag_event)? {
                    perf.converged = true;
                    break;
                }
            }
        }
    }

    finish_solve(gpu, k, psi, a, m, w, ctrl, &mut perf)?;
    Ok(perf)
}

fn bicg_p_update(
    gpu: &Gpu,
    k: &SolverKernels,
    w: &mut SolverWorkspace,
    n: usize,
) -> Result<()> {
    let nl = to_label(n)?;
    unsafe {
        gpu.stream()
            .launch_builder(&k.p_update)
            .arg(&mut w.p)
            .arg(&w.r)
            .arg(&w.v)
            .arg(&w.beta)
            .arg(&w.omega)
            .arg(&nl)
            .launch(cfg_for(n))?;
    }
    Ok(())
}

fn bicg_s_update(
    gpu: &Gpu,
    k: &SolverKernels,
    w: &mut SolverWorkspace,
    n: usize,
) -> Result<()> {
    let nl = to_label(n)?;
    unsafe {
        gpu.stream()
            .launch_builder(&k.s_update)
            .arg(&mut w.s)
            .arg(&w.r)
            .arg(&w.v)
            .arg(&w.alpha)
            .arg(&nl)
            .launch(cfg_for(n))?;
    }
    Ok(())
}

fn bicg_x_update(
    gpu: &Gpu,
    k: &SolverKernels,
    psi: &mut DevBuf<Scalar>,
    w: &SolverWorkspace,
    n: usize,
) -> Result<()> {
    let nl = to_label(n)?;
    unsafe {
        gpu.stream()
            .launch_builder(&k.x_update)
            .arg(&mut *psi)
            .arg(&w.p_hat)
            .arg(&w.s_hat)
            .arg(&w.alpha)
            .arg(&w.omega)
            .arg(&nl)
            .launch(cfg_for(n))?;
    }
    Ok(())
}

fn bicg_r_update(
    gpu: &Gpu,
    k: &SolverKernels,
    w: &mut SolverWorkspace,
    n: usize,
) -> Result<()> {
    let nl = to_label(n)?;
    unsafe {
        gpu.stream()
            .launch_builder(&k.r_update)
            .arg(&mut w.r)
            .arg(&w.s)
            .arg(&w.t)
            .arg(&w.omega)
            .arg(&nl)
            .launch(cfg_for(n))?;
    }
    Ok(())
}

// ==========================================================================
//  Preconditioned conjugate gradient
// ==========================================================================

/// Preconditioned conjugate gradient.
///
/// Hestenes & Stiefel (1952); Saad §6.7, Algorithm 6.18:
///
/// ```text
/// r = b - A·x ;  z = M^-1 r ;  p = z ;  rho = (r,z)
/// repeat
///     q      = A·p
///     alpha  = rho / (p,q)
///     x     += alpha·p
///     r     -= alpha·q
///     z      = M^-1 r
///     rho'   = (r,z)
///     beta   = rho'/rho
///     p      = z + beta·p
///     rho    = rho'
/// ```
///
/// **Symmetric positive definite only.** CG minimises over a Krylov space
/// using `A`-orthogonality, and on an asymmetric matrix that quantity is not
/// an inner product: the method does not converge slowly, it converges to the
/// wrong thing or not at all. The pressure equation is the system this is for
/// (`SPEC-LIT` §8.2); anything carrying convection must use
/// [`solve_pbicgstab`]. Nothing here checks the symmetry - it costs a pass
/// over the faces and the caller knows the answer already.
pub fn solve_pcg(
    gpu: &Gpu,
    k: &SolverKernels,
    psi: &mut DevBuf<Scalar>,
    a: &GpuLduMatrix,
    m: &GpuMesh,
    w: &mut SolverWorkspace,
    ctrl: &SolverControls,
) -> Result<SolverPerformance> {
    let n = a.n_cells;
    let mut perf = SolverPerformance {
        converged: true,
        ..Default::default()
    };
    if n == 0 {
        return Ok(perf);
    }
    perf.converged = false;

    check_workspace(w, n)?;
    if psi.len() < n {
        return Err(Error::Config(format!(
            "solve_pcg: psi holds {} values, the system has {n} cells",
            psi.len()
        )));
    }

    let precon = effective_preconditioner(ctrl.precon, w)?;
    build_preconditioner(gpu, k, w, a, m, precon)?;

    device_norm_factor(gpu, k, w, &*psi, a, m)?;
    vec_sub(gpu, k, &mut w.r, &a.source, &w.apsi, n)?;

    device_sum_mag(gpu, k, &mut w.initial_res, &w.r, &mut w.partials, n)?;
    copy_scalar(gpu, k, &mut w.final_res, &w.initial_res)?;

    gpu.fill_zero(&mut w.flag)?;

    // z = M^-1 r ; p = z ; rho = (r,z)
    precondition_ws(gpu, k, w, PreconTarget::PHatFromR, a, m, precon, n)?;
    vec_copy(gpu, k, &mut w.p, &w.p_hat, n)?;
    device_dot(gpu, k, &mut w.rho, &w.r, &w.p_hat, &mut w.partials, n)?;

    let max_iter = ctrl.max_iter.max(0) as usize;
    let interval = ctrl.check_interval.max(1) as usize;
    let checking = !ctrl.fixed_iters;

    if checking {
        convergence_test(
            gpu, k, &mut w.flag, &w.initial_res, &w.initial_res, &w.norm_factor, ctrl, 0,
        )?;
        perf.converged = read_flag(gpu, &w.flag, &mut w.flag_host, &w.flag_event)?;
    }

    if !perf.converged {
        for it in 0..max_iter {
            let iters = it + 1;

            amul(gpu, k, &mut w.v, &w.p, a, m)?;
            device_dot(gpu, k, &mut w.den, &w.p, &w.v, &mut w.partials, n)?;
            divide_scalar(gpu, k, &mut w.alpha, &w.rho, &w.den)?;

            cg_axpy(gpu, k, psi, &w.p, &w.alpha, n)?;
            cg_axmy(gpu, k, w, n)?;

            perf.n_iterations = iters;

            if checking && iters % interval == 0 {
                device_sum_mag(gpu, k, &mut w.final_res, &w.r, &mut w.partials, n)?;
                let itl = to_label(iters)?;
                convergence_test(
                    gpu,
                    k,
                    &mut w.flag,
                    &w.final_res,
                    &w.initial_res,
                    &w.norm_factor,
                    ctrl,
                    itl,
                )?;
                if read_flag(gpu, &w.flag, &mut w.flag_host, &w.flag_event)? {
                    perf.converged = true;
                    break;
                }
            }

            // z = M^-1 r ; beta = (r,z)/rho ; p = z + beta·p
            precondition_ws(gpu, k, w, PreconTarget::PHatFromR, a, m, precon, n)?;
            device_dot(gpu, k, &mut w.num, &w.r, &w.p_hat, &mut w.partials, n)?;
            divide_scalar(gpu, k, &mut w.beta, &w.num, &w.rho)?;
            copy_scalar(gpu, k, &mut w.rho, &w.num)?;
            cg_p_update(gpu, k, w, n)?;
        }
    }

    finish_solve(gpu, k, psi, a, m, w, ctrl, &mut perf)?;
    Ok(perf)
}

fn cg_axpy(
    gpu: &Gpu,
    k: &SolverKernels,
    y: &mut DevBuf<Scalar>,
    x: &DevBuf<Scalar>,
    scale: &DevBuf<Scalar>,
    n: usize,
) -> Result<()> {
    let nl = to_label(n)?;
    unsafe {
        gpu.stream()
            .launch_builder(&k.axpy)
            .arg(&mut *y)
            .arg(x)
            .arg(scale)
            .arg(&nl)
            .launch(cfg_for(n))?;
    }
    Ok(())
}

/// `r -= alpha·q`.
fn cg_axmy(gpu: &Gpu, k: &SolverKernels, w: &mut SolverWorkspace, n: usize) -> Result<()> {
    let nl = to_label(n)?;
    unsafe {
        gpu.stream()
            .launch_builder(&k.axmy)
            .arg(&mut w.r)
            .arg(&w.v)
            .arg(&w.alpha)
            .arg(&nl)
            .launch(cfg_for(n))?;
    }
    Ok(())
}

/// `p = z + beta·p`.
fn cg_p_update(gpu: &Gpu, k: &SolverKernels, w: &mut SolverWorkspace, n: usize) -> Result<()> {
    let nl = to_label(n)?;
    unsafe {
        gpu.stream()
            .launch_builder(&k.p_update_cg)
            .arg(&mut w.p)
            .arg(&w.p_hat)
            .arg(&w.beta)
            .arg(&nl)
            .launch(cfg_for(n))?;
    }
    Ok(())
}

// ==========================================================================
//  Shared epilogue
// ==========================================================================

/// The end of both solvers.
///
/// With `report_residuals` on, the residual is recomputed as the true
/// `b - A·psi` rather than taken from the recurrence: a Krylov recurrence
/// residual drifts from the real one as the solve stagnates, and a solver
/// that reports the drifted number is a solver that hides its own failure.
/// Costs one extra matrix product per solve, and nothing at all when the
/// residuals are not wanted.
#[allow(clippy::too_many_arguments)]
fn finish_solve(
    gpu: &Gpu,
    k: &SolverKernels,
    psi: &mut DevBuf<Scalar>,
    a: &GpuLduMatrix,
    m: &GpuMesh,
    w: &mut SolverWorkspace,
    ctrl: &SolverControls,
    perf: &mut SolverPerformance,
) -> Result<()> {
    if !ctrl.report_residuals {
        return Ok(());
    }
    let n = a.n_cells;

    amul(gpu, k, &mut w.apsi, &*psi, a, m)?;
    vec_sub(gpu, k, &mut w.tmp, &a.source, &w.apsi, n)?;
    device_sum_mag(gpu, k, &mut w.final_res, &w.tmp, &mut w.partials, n)?;

    collect_report(gpu, k, w, perf)?;

    // In fixed-iteration mode nothing sampled the device flag, but the
    // numbers are now on the host anyway, so the same criterion can be
    // applied here rather than reporting "unknown" as "not converged".
    if ctrl.fixed_iters {
        let abs = perf.final_residual <= ctrl.tolerance;
        let rel = ctrl.rel_tol > 0.0 && perf.final_residual <= ctrl.rel_tol * perf.initial_residual;
        perf.converged = abs || rel;
    }
    Ok(())
}

// ==========================================================================
//  Solver selection - SPEC-LIT section 13.4 applied to `solvers/<var>/solver`
// ==========================================================================

/// Solve `A·psi = b` with the method the case asked for.
///
/// ```text
/// PBiCGStab -> solve_pbicgstab
/// PCG       -> solve_pcg, after checking the matrix really is symmetric
/// GAMG      -> Error: it is a pressure BACKEND here, not a Krylov method
/// ```
///
/// **PCG on an asymmetric matrix is an error.** Conjugate gradients minimise
/// over a Krylov space using `A`-orthogonality, and on an asymmetric matrix
/// that quantity is not an inner product: the method does not converge slowly,
/// it converges to the wrong thing or not at all. It *often appears* to work,
/// which is exactly why it has to be refused rather than left to the user to
/// notice. `SPEC-LIT` §8.2 and §13.4.
///
/// `DIC` is likewise refused on an asymmetric matrix: it is the Cholesky form
/// (`SPEC-LIT` §21) and there is no Cholesky factor of an asymmetric matrix to
/// approximate. `DILU` covers that case and the message says so.
///
/// The symmetry check costs one pass over the faces and one small copy back,
/// and runs ONLY when `PCG` or `DIC` was requested - so a PBiCGStab + Jacobi
/// or PBiCGStab + DILU solve is untouched and still captures into a CUDA graph.
pub fn solve(
    gpu: &Gpu,
    k: &SolverKernels,
    psi: &mut DevBuf<Scalar>,
    a: &GpuLduMatrix,
    m: &GpuMesh,
    w: &mut SolverWorkspace,
    ctrl: &SolverControls,
) -> Result<SolverPerformance> {
    let needs_symmetry =
        ctrl.solver == LinearSolverKind::PCG || ctrl.precon == Preconditioner::Dic;

    if needs_symmetry {
        let d = symmetry_defects(gpu, k, w, a, m)?;
        if !d.is_symmetric() {
            if ctrl.solver == LinearSolverKind::PCG {
                return Err(Error::Config(format!(
                    "solver PCG was requested for an asymmetric system: {} \
                     disagree. Conjugate gradients are defined only for \
                     symmetric positive definite matrices (SPEC-LIT 8.2); on \
                     this one they are not guaranteed to converge at all. Use \
                     PBiCGStab.",
                    d.what_failed()
                )));
            }
            return Err(Error::Config(format!(
                "preconditioner DIC was requested for an asymmetric matrix: {} \
                 disagree. DIC is the incomplete CHOLESKY factorisation \
                 (SPEC-LIT 21) and an asymmetric matrix has no Cholesky factor \
                 to approximate. Use DILU, which is the asymmetric case of the \
                 same algorithm.",
                d.what_failed()
            )));
        }
    }

    match ctrl.solver {
        LinearSolverKind::PBiCGStab => solve_pbicgstab(gpu, k, psi, a, m, w, ctrl),
        LinearSolverKind::PCG => solve_pcg(gpu, k, psi, a, m, w, ctrl),
        LinearSolverKind::Gamg => Err(Error::Config(
            "solver GAMG was requested. Algebraic multigrid is not a Krylov \
             method and is not reimplemented here (SPEC-LIT 8.3): it reaches \
             ofgpu only as the AMGX pressure backend, which the pressure \
             equation selects through crate::pressure. For any other equation \
             use PBiCGStab, or PCG if the system is symmetric."
                .to_string(),
        )),
    }
}

// ==========================================================================
//  Tests
// ==========================================================================

// The reference arithmetic in these tests is `f64` whatever the crate solves
// in, so device values are cast on the way in and out. With the default
// feature set `Scalar` is already `f64` and clippy calls the cast redundant;
// under `--features single` it is `f32` and the cast is the whole point.
#[cfg(test)]
#[allow(clippy::unnecessary_cast)]
mod tests {
    use super::*;

    use crate::mesh::{HostMesh, PatchInfo, PatchKind};
    use crate::types::Vec3;

    // ----------------------------------------------------------------------
    //  How close is close, given what the crate solves in
    //
    //  Every reference below is computed in `f64` on the host, because that is
    //  the only way to have an answer that does not come from the thing being
    //  tested. The device works in [`Scalar`], which the `single` feature
    //  switches to `f32`, so the thresholds have to move with it - otherwise a
    //  single-precision build would fail on arithmetic that is behaving
    //  perfectly.
    // ----------------------------------------------------------------------

    /// Absolute agreement demanded between a converged solve and the dense
    /// direct solve of the same system.
    #[cfg(feature = "single")]
    const SLACK: f64 = 2e-3;
    #[cfg(not(feature = "single"))]
    const SLACK: f64 = 1e-10;

    /// Relative agreement demanded of a single device operation - one matrix
    /// product, one normalisation factor - against the same formula on the
    /// host.
    #[cfg(feature = "single")]
    const ROUNDOFF: f64 = 1e-5;
    #[cfg(not(feature = "single"))]
    const ROUNDOFF: f64 = 1e-13;

    /// Relative agreement demanded of a reduction over 300k values, where the
    /// accumulated round-off is what is being measured.
    #[cfg(feature = "single")]
    const RED_SLACK: f64 = 1e-4;
    #[cfg(not(feature = "single"))]
    const RED_SLACK: f64 = 1e-12;

    /// The `tolerance` a solve is asked to reach. Below the precision floor it
    /// would simply never be met.
    #[cfg(feature = "single")]
    const SOLVE_TOL: Scalar = 1e-6;
    #[cfg(not(feature = "single"))]
    const SOLVE_TOL: Scalar = 1e-14;

    /// A tolerance an already-converged field passes on the first look.
    #[cfg(feature = "single")]
    const LOOSE_TOL: Scalar = 1e-4;
    #[cfg(not(feature = "single"))]
    const LOOSE_TOL: Scalar = 1e-8;

    // ----------------------------------------------------------------------
    //  A dense system, and a mesh whose LDU structure can hold it
    // ----------------------------------------------------------------------

    /// Deterministic pseudo-random values in `[-1, 1)`. No dependency, and the
    /// same numbers every run so a failure can be reproduced.
    fn noise(n: usize, seed: u64) -> Vec<f64> {
        let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (s >> 33) as f64 / (1u64 << 31) as f64 - 1.0
            })
            .collect()
    }

    /// The face list of a *complete* graph on `n` cells, minus the pair
    /// `(0, n-1)`, which is instead coupled through a cyclic patch.
    ///
    /// A complete graph is the densest LDU matrix that exists, so the solver
    /// is exercised against a genuinely dense reference rather than against a
    /// nearly diagonal one where almost any iteration would look convergent.
    /// Faces come out in ascending `(owner, neighbour)` order, which is the
    /// upper-triangular ordering every gather kernel assumes.
    fn face_pairs(n: usize) -> Vec<(usize, usize)> {
        let mut f = Vec::new();
        for i in 0..n {
            for j in (i + 1)..n {
                if !(i == 0 && j == n - 1) {
                    f.push((i, j));
                }
            }
        }
        f
    }

    /// A mesh carrying that structure. The geometry is never read by the
    /// linear solver, so it is filled with placeholders of the right length;
    /// what matters is the addressing and the cyclic couple.
    fn dense_mesh(n: usize) -> HostMesh {
        let faces = face_pairs(n);
        let nif = faces.len();

        let mut hm = HostMesh {
            n_cells: n,
            n_internal_faces: nif,
            // Two faces, one on each side of a cyclic couple joining the first
            // cell to the last.
            n_boundary_faces: 2,
            n_points: 0,

            owner: faces.iter().map(|&(o, _)| o as Label).collect(),
            neighbour: faces.iter().map(|&(_, m)| m as Label).collect(),

            v: vec![1.0; n],
            c: vec![Vec3::default(); n],

            sf: vec![Vec3::default(); nif],
            mag_sf: vec![1.0; nif],
            cf: vec![Vec3::default(); nif],
            weights: vec![0.5; nif],
            delta_coeffs: vec![1.0; nif],
            non_orth_corr: vec![Vec3::default(); nif],

            b_face_cells: vec![0, (n - 1) as Label],
            b_sf: vec![Vec3::default(); 2],
            b_mag_sf: vec![1.0; 2],
            b_cf: vec![Vec3::default(); 2],
            b_delta_coeffs: vec![1.0; 2],
            b_y: vec![1.0; 2],
            b_nbr_cell: vec![(n - 1) as Label, 0],
            b_weights: vec![0.5; 2],
            b_kind: vec![PatchKind::Cyclic as Label; 2],
            b_patch: vec![0, 1],

            patches: vec![
                PatchInfo {
                    name: "cyc_lo".into(),
                    type_name: "cyclic".into(),
                    kind: PatchKind::Cyclic,
                    start: 0,
                    size: 1,
                    nbr_patch: Some(1),
                },
                PatchInfo {
                    name: "cyc_hi".into(),
                    type_name: "cyclic".into(),
                    kind: PatchKind::Cyclic,
                    start: 1,
                    size: 1,
                    nbr_patch: Some(0),
                },
            ],

            ..HostMesh::default()
        };
        hm.build_cell_face_maps();
        hm
    }

    /// A dense `n x n` matrix, and the LDU/cyclic arrays that represent it
    /// exactly on [`dense_mesh`].
    struct Dense {
        n: usize,
        a: Vec<f64>,
        diag: Vec<Scalar>,
        upper: Vec<Scalar>,
        lower: Vec<Scalar>,
        /// `boundary_coeffs` of the two cyclic faces.
        bnd: Vec<Scalar>,
    }

    impl Dense {
        fn at(&self, i: usize, j: usize) -> f64 {
            self.a[i * self.n + j]
        }

        /// `symmetric` gives an SPD matrix (symmetric, strictly diagonally
        /// dominant with a positive diagonal, hence positive definite by
        /// Gershgorin); otherwise an asymmetric one that is still dominant
        /// enough for BiCGStab to be well behaved.
        fn build(n: usize, seed: u64, symmetric: bool) -> Self {
            let off = noise(n * n, seed);
            let mut a = vec![0.0; n * n];

            for i in 0..n {
                for j in 0..n {
                    if i != j {
                        a[i * n + j] = if symmetric && j < i {
                            off[j * n + i]
                        } else {
                            off[i * n + j]
                        };
                    }
                }
            }
            for i in 0..n {
                let s: f64 = (0..n).filter(|&j| j != i).map(|j| a[i * n + j].abs()).sum();
                a[i * n + i] = s + 1.0 + 0.5 * (i as f64 % 3.0);
            }
            // A symmetric matrix must stay symmetric after the diagonal is
            // set; the row sums differ, so use the same rule both ways.
            if symmetric {
                for i in 0..n {
                    let s: f64 = (0..n).filter(|&j| j != i).map(|j| a[i * n + j].abs()).sum();
                    a[i * n + i] = s + 1.0;
                }
            }

            let faces = face_pairs(n);
            let diag = (0..n).map(|i| a[i * n + i] as Scalar).collect();
            let upper = faces.iter().map(|&(o, m)| a[o * n + m] as Scalar).collect();
            let lower = faces.iter().map(|&(o, m)| a[m * n + o] as Scalar).collect();
            // A(0, n-1) = -boundaryCoeffs[0], A(n-1, 0) = -boundaryCoeffs[1].
            let bnd = vec![
                -a[n - 1] as Scalar,
                -a[(n - 1) * n] as Scalar,
            ];

            Self { n, a, diag, upper, lower, bnd }
        }

        fn matvec(&self, x: &[f64]) -> Vec<f64> {
            (0..self.n)
                .map(|i| (0..self.n).map(|j| self.at(i, j) * x[j]).sum())
                .collect()
        }

        /// Gaussian elimination with partial pivoting - the direct solve the
        /// iterative answer is checked against. Written out here rather than
        /// taken from anywhere, so the comparison is against arithmetic and
        /// not against another implementation.
        fn direct_solve(&self, b: &[f64]) -> Vec<f64> {
            let n = self.n;
            let mut m = self.a.clone();
            let mut x = b.to_vec();

            for col in 0..n {
                let mut piv = col;
                for row in (col + 1)..n {
                    if m[row * n + col].abs() > m[piv * n + col].abs() {
                        piv = row;
                    }
                }
                if piv != col {
                    for j in 0..n {
                        m.swap(col * n + j, piv * n + j);
                    }
                    x.swap(col, piv);
                }
                let d = m[col * n + col];
                assert!(d.abs() > 0.0, "singular test matrix");

                for row in (col + 1)..n {
                    let f = m[row * n + col] / d;
                    if f == 0.0 {
                        continue;
                    }
                    for j in col..n {
                        m[row * n + j] -= f * m[col * n + j];
                    }
                    x[row] -= f * x[col];
                }
            }
            for col in (0..n).rev() {
                let mut s = x[col];
                for j in (col + 1)..n {
                    s -= m[col * n + j] * x[j];
                }
                x[col] = s / m[col * n + col];
            }
            x
        }
    }

    /// Everything a test needs on the device. `None` when there is no GPU, so
    /// the suite still passes on a machine without one.
    struct Rig {
        gpu: Gpu,
        k: SolverKernels,
        m: GpuMesh,
        a: GpuLduMatrix,
        dense: Dense,
        exact: Vec<f64>,
        b: Vec<f64>,
    }

    fn rig(n: usize, seed: u64, symmetric: bool) -> Option<Rig> {
        let hm = dense_mesh(n);
        let dense = Dense::build(n, seed, symmetric);

        let exact: Vec<f64> = noise(n, seed + 91).iter().map(|v| 1.0 + 2.0 * v).collect();
        let b = dense.matvec(&exact);

        let gpu = Gpu::new(0).ok()?;
        let k = SolverKernels::new(&gpu).ok()?;
        let m = GpuMesh::upload(&gpu, &hm).ok()?;

        let mut a = GpuLduMatrix::new(&gpu, &m).ok()?;
        a.zero(&gpu).ok()?;
        gpu.write(&mut a.diag, &dense.diag).ok()?;
        gpu.write(&mut a.upper, &dense.upper).ok()?;
        gpu.write(&mut a.lower, &dense.lower).ok()?;
        gpu.write(&mut a.boundary_coeffs, &dense.bnd).ok()?;
        let src: Vec<Scalar> = b.iter().map(|v| *v as Scalar).collect();
        gpu.write(&mut a.source, &src).ok()?;

        Some(Rig { gpu, k, m, a, dense, exact, b })
    }

    fn tight() -> SolverControls {
        SolverControls {
            solver: LinearSolverKind::PBiCGStab,
            tolerance: SOLVE_TOL,
            rel_tol: 0.0,
            max_iter: 500,
            min_iter: 0,
            precon: Preconditioner::Diagonal,
            check_interval: 1,
            fixed_iters: false,
            report_residuals: true,
        }
    }

    fn max_diff(a: &[f64], b: &[f64]) -> f64 {
        a.iter().zip(b).fold(0.0f64, |m, (x, y)| m.max((x - y).abs()))
    }

    // ----------------------------------------------------------------------
    //  Reductions
    // ----------------------------------------------------------------------

    /// A zero-length reduction must not launch a grid of zero blocks, which is
    /// an illegal configuration rather than a no-op. It must still leave a
    /// defined answer behind.
    #[test]
    fn a_zero_length_reduction_writes_zero_without_launching() {
        let Some(gpu) = Gpu::new(0).ok() else { return };
        let k = SolverKernels::new(&gpu).expect("solver kernels");

        let x: DevBuf<Scalar> = gpu.zeros(4).expect("x");
        let mut partials: DevBuf<Scalar> = gpu.zeros(MAX_REDUCE_BLOCKS).expect("partials");
        let mut partials_b: DevBuf<Scalar> = gpu.zeros(MAX_REDUCE_BLOCKS).expect("partials");

        // Seed the outputs with something non-zero so "wrote zero" is a real
        // observation rather than the allocation showing through.
        let mut out: DevBuf<Scalar> = gpu.upload(&[7.0 as Scalar]).expect("out");
        let mut out_b: DevBuf<Scalar> = gpu.upload(&[7.0 as Scalar]).expect("out");

        device_sum(&gpu, &k, &mut out, &x, &mut partials, 0).expect("sum");
        assert_eq!(gpu.download(&out).expect("dl")[0], 0.0);

        device_sum_mag(&gpu, &k, &mut out, &x, &mut partials, 0).expect("sum_mag");
        assert_eq!(gpu.download(&out).expect("dl")[0], 0.0);

        device_dot(&gpu, &k, &mut out, &x, &x, &mut partials, 0).expect("dot");
        assert_eq!(gpu.download(&out).expect("dl")[0], 0.0);

        device_max_mag(&gpu, &k, &mut out, &x, &mut partials, 0).expect("max");
        assert_eq!(gpu.download(&out).expect("dl")[0], 0.0);

        device_dot2(
            &gpu, &k, &mut out, &mut out_b, &x, &x, &mut partials, &mut partials_b, 0,
        )
        .expect("dot2");
        assert_eq!(gpu.download(&out).expect("dl")[0], 0.0);
        assert_eq!(gpu.download(&out_b).expect("dl")[0], 0.0);

        gpu.sync().expect("sync");
    }

    /// The two-stage reduction against the same arithmetic done on the host,
    /// at a length that needs the grid-stride path (more elements than the
    /// 1024-block cap times 256 threads).
    #[test]
    fn the_reductions_agree_with_the_host() {
        let Some(gpu) = Gpu::new(0).ok() else { return };
        let k = SolverKernels::new(&gpu).expect("solver kernels");

        let n = 300_003;
        let hx: Vec<Scalar> = noise(n, 5).iter().map(|v| (v * 3.0) as Scalar).collect();
        let hy: Vec<Scalar> = noise(n, 6).iter().map(|v| (v * 0.5) as Scalar).collect();

        let x = gpu.upload(&hx).expect("x");
        let y = gpu.upload(&hy).expect("y");
        let mut partials: DevBuf<Scalar> = gpu.zeros(MAX_REDUCE_BLOCKS).expect("p");
        let mut partials_b: DevBuf<Scalar> = gpu.zeros(MAX_REDUCE_BLOCKS).expect("p");
        let mut out: DevBuf<Scalar> = gpu.zeros(1).expect("out");
        let mut out_b: DevBuf<Scalar> = gpu.zeros(1).expect("out");

        let host_sum: f64 = hx.iter().map(|v| *v as f64).sum();
        let host_mag: f64 = hx.iter().map(|v| (*v as f64).abs()).sum();
        let host_dot: f64 = hx.iter().zip(&hy).map(|(a, b)| *a as f64 * *b as f64).sum();
        let host_max = hx.iter().fold(0.0f64, |m, v| m.max((*v as f64).abs()));

        let rel = |got: f64, want: f64| (got - want).abs() / want.abs().max(1.0);

        device_sum(&gpu, &k, &mut out, &x, &mut partials, n).expect("sum");
        let got = gpu.download(&out).expect("dl")[0] as f64;
        assert!(rel(got, host_sum) < RED_SLACK, "sum {got} vs {host_sum}");

        device_sum_mag(&gpu, &k, &mut out, &x, &mut partials, n).expect("mag");
        let got = gpu.download(&out).expect("dl")[0] as f64;
        assert!(rel(got, host_mag) < RED_SLACK, "sum|x| {got} vs {host_mag}");

        device_dot(&gpu, &k, &mut out, &x, &y, &mut partials, n).expect("dot");
        let got = gpu.download(&out).expect("dl")[0] as f64;
        assert!(rel(got, host_dot) < RED_SLACK, "dot {got} vs {host_dot}");

        device_max_mag(&gpu, &k, &mut out, &x, &mut partials, n).expect("max");
        let got = gpu.download(&out).expect("dl")[0] as f64;
        assert_eq!(got, host_max, "max|x|");

        // The fused pair must agree with the two separate reductions.
        device_dot2(
            &gpu, &k, &mut out, &mut out_b, &x, &y, &mut partials, &mut partials_b, n,
        )
        .expect("dot2");
        let xy = gpu.download(&out).expect("dl")[0] as f64;
        let xx = gpu.download(&out_b).expect("dl")[0] as f64;
        let host_xx: f64 = hx.iter().map(|v| (*v as f64) * (*v as f64)).sum();
        assert!(rel(xy, host_dot) < RED_SLACK, "dot2 (x,y) {xy} vs {host_dot}");
        assert!(rel(xx, host_xx) < RED_SLACK, "dot2 (x,x) {xx} vs {host_xx}");
    }

    // ----------------------------------------------------------------------
    //  The matrix product the solvers are built on
    // ----------------------------------------------------------------------

    /// `amul` against a dense host product, cyclic coupling included. If this
    /// is wrong then every solver result below is meaningless.
    #[test]
    fn amul_reproduces_the_dense_product() {
        let Some(r) = rig(19, 31, false) else { return };

        let hx: Vec<f64> = noise(19, 77).iter().map(|v| 1.0 + v).collect();
        let dx: Vec<Scalar> = hx.iter().map(|v| *v as Scalar).collect();
        let x = r.gpu.upload(&dx).expect("x");
        let mut y: DevBuf<Scalar> = r.gpu.zeros(19).expect("y");

        amul(&r.gpu, &r.k, &mut y, &x, &r.a, &r.m).expect("amul");
        let got: Vec<f64> = r
            .gpu
            .download(&y)
            .expect("dl")
            .iter()
            .map(|v| *v as f64)
            .collect();

        let want = r.dense.matvec(&hx);
        let err = max_diff(&got, &want);
        let scale = want.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        assert!(err / scale < ROUNDOFF, "amul off by {err:.3e}");
    }

    // ----------------------------------------------------------------------
    //  The solvers, against a dense direct solve
    // ----------------------------------------------------------------------

    #[test]
    fn pcg_matches_a_dense_direct_solve_on_an_spd_system() {
        let n = 23;
        let Some(r) = rig(n, 1234, true) else { return };

        let direct = r.dense.direct_solve(&r.b);
        // The direct solve must itself be right, or it certifies nothing.
        assert!(
            max_diff(&direct, &r.exact) < 1e-10,
            "the test's own direct solve is wrong"
        );

        let mut psi: DevBuf<Scalar> = r.gpu.zeros(n).expect("psi");
        let mut w = SolverWorkspace::for_mesh(&r.gpu, &r.m).expect("workspace");

        let perf = solve_pcg(&r.gpu, &r.k, &mut psi, &r.a, &r.m, &mut w, &tight())
            .expect("pcg");

        let got: Vec<f64> = r
            .gpu
            .download(&psi)
            .expect("dl")
            .iter()
            .map(|v| *v as f64)
            .collect();

        assert!(perf.converged, "pcg did not converge: {perf:?}");
        let err = max_diff(&got, &direct);
        assert!(
            err < SLACK,
            "pcg differs from the dense direct solve by {err:.3e} after \
             {} iterations (residual {:.3e})",
            perf.n_iterations,
            perf.final_residual
        );
        // A dense SPD system of order n is solved by CG in at most n steps in
        // exact arithmetic; anything much beyond that means the recurrence is
        // wrong rather than merely slow.
        assert!(
            perf.n_iterations <= 2 * n,
            "pcg took {} iterations on a {n}-cell system",
            perf.n_iterations
        );
    }

    #[test]
    fn pbicgstab_matches_a_dense_direct_solve_on_an_spd_system() {
        let n = 23;
        let Some(r) = rig(n, 1234, true) else { return };

        let direct = r.dense.direct_solve(&r.b);
        let mut psi: DevBuf<Scalar> = r.gpu.zeros(n).expect("psi");
        let mut w = SolverWorkspace::for_mesh(&r.gpu, &r.m).expect("workspace");

        let perf = solve_pbicgstab(&r.gpu, &r.k, &mut psi, &r.a, &r.m, &mut w, &tight())
            .expect("pbicgstab");

        let got: Vec<f64> = r
            .gpu
            .download(&psi)
            .expect("dl")
            .iter()
            .map(|v| *v as f64)
            .collect();

        assert!(perf.converged, "pbicgstab did not converge: {perf:?}");
        let err = max_diff(&got, &direct);
        assert!(err < SLACK, "pbicgstab differs by {err:.3e}");
    }

    /// The asymmetric path, which is the whole reason BiCGStab is here: on
    /// this matrix `upper != lower` on every face and on the cyclic couple,
    /// so a solver that quietly assumed symmetry would fail.
    #[test]
    fn pbicgstab_matches_a_dense_direct_solve_on_an_asymmetric_system() {
        let n = 29;
        let Some(r) = rig(n, 4321, false) else { return };

        // Confirm the system really is asymmetric before drawing conclusions.
        let asym = (0..n)
            .flat_map(|i| (0..n).map(move |j| (i, j)))
            .fold(0.0f64, |m, (i, j)| {
                m.max((r.dense.at(i, j) - r.dense.at(j, i)).abs())
            });
        assert!(asym > 0.1, "the asymmetric fixture is symmetric");

        let direct = r.dense.direct_solve(&r.b);
        let mut psi: DevBuf<Scalar> = r.gpu.zeros(n).expect("psi");
        let mut w = SolverWorkspace::for_mesh(&r.gpu, &r.m).expect("workspace");

        let perf = solve_pbicgstab(&r.gpu, &r.k, &mut psi, &r.a, &r.m, &mut w, &tight())
            .expect("pbicgstab");

        let got: Vec<f64> = r
            .gpu
            .download(&psi)
            .expect("dl")
            .iter()
            .map(|v| *v as f64)
            .collect();

        assert!(perf.converged, "pbicgstab did not converge: {perf:?}");
        let err = max_diff(&got, &direct);
        assert!(err < SLACK, "pbicgstab differs by {err:.3e}");
    }

    /// Without a preconditioner the same system must come out at the same
    /// answer, only more slowly. This is what makes `Preconditioner::None` a
    /// usable control rather than an untested branch.
    #[test]
    fn an_unpreconditioned_solve_reaches_the_same_answer() {
        let n = 23;
        let Some(r) = rig(n, 1234, false) else { return };

        let direct = r.dense.direct_solve(&r.b);
        let mut psi: DevBuf<Scalar> = r.gpu.zeros(n).expect("psi");
        let mut w = SolverWorkspace::for_mesh(&r.gpu, &r.m).expect("workspace");

        let ctrl = SolverControls { precon: Preconditioner::None, ..tight() };
        let perf = solve_pbicgstab(&r.gpu, &r.k, &mut psi, &r.a, &r.m, &mut w, &ctrl)
            .expect("pbicgstab");

        let got: Vec<f64> = r
            .gpu
            .download(&psi)
            .expect("dl")
            .iter()
            .map(|v| *v as f64)
            .collect();

        assert!(perf.converged, "unpreconditioned solve did not converge");
        assert!(max_diff(&got, &direct) < SLACK);
    }

    /// A workspace is allocated once and reused for the whole run, so a second
    /// solve through it must not see anything the first left behind. Bitwise
    /// equality, not "close": the gather is deterministic, so anything else is
    /// state leaking between solves.
    #[test]
    fn a_reused_workspace_gives_the_same_answer_twice() {
        let n = 29;
        let Some(r) = rig(n, 99, false) else { return };
        let mut w = SolverWorkspace::for_mesh(&r.gpu, &r.m).expect("workspace");

        let mut first: DevBuf<Scalar> = r.gpu.zeros(n).expect("psi");
        let p1 = solve_pbicgstab(&r.gpu, &r.k, &mut first, &r.a, &r.m, &mut w, &tight())
            .expect("first");
        let a1 = r.gpu.download(&first).expect("dl");

        let mut second: DevBuf<Scalar> = r.gpu.zeros(n).expect("psi");
        let p2 = solve_pbicgstab(&r.gpu, &r.k, &mut second, &r.a, &r.m, &mut w, &tight())
            .expect("second");
        let a2 = r.gpu.download(&second).expect("dl");

        assert_eq!(a1, a2, "a reused workspace changed the answer");
        assert_eq!(p1.n_iterations, p2.n_iterations);
        assert_eq!(p1.initial_residual, p2.initial_residual);
        assert_eq!(p1.final_residual, p2.final_residual);

        // And the same again for PCG on the same workspace, which must not be
        // disturbed by BiCGStab having used it.
        let Some(rs) = rig(n, 99, true) else { return };
        let mut w2 = SolverWorkspace::for_mesh(&rs.gpu, &rs.m).expect("workspace");
        let mut c1: DevBuf<Scalar> = rs.gpu.zeros(n).expect("psi");
        let mut c2: DevBuf<Scalar> = rs.gpu.zeros(n).expect("psi");
        solve_pcg(&rs.gpu, &rs.k, &mut c1, &rs.a, &rs.m, &mut w2, &tight()).expect("cg1");
        solve_pcg(&rs.gpu, &rs.k, &mut c2, &rs.a, &rs.m, &mut w2, &tight()).expect("cg2");
        assert_eq!(
            rs.gpu.download(&c1).expect("dl"),
            rs.gpu.download(&c2).expect("dl")
        );
    }

    /// A system that is already solved must cost zero iterations. That path
    /// runs on every steady-state pressure equation near convergence, and a
    /// solver that iterated anyway would waste most of a run.
    #[test]
    fn an_already_solved_system_takes_no_iterations() {
        let n = 17;
        let Some(r) = rig(n, 7, true) else { return };

        let start: Vec<Scalar> = r.exact.iter().map(|v| *v as Scalar).collect();
        let mut psi = r.gpu.upload(&start).expect("psi");
        let mut w = SolverWorkspace::for_mesh(&r.gpu, &r.m).expect("workspace");

        let ctrl = SolverControls { tolerance: LOOSE_TOL, ..tight() };
        let perf = solve_pbicgstab(&r.gpu, &r.k, &mut psi, &r.a, &r.m, &mut w, &ctrl)
            .expect("solve");

        assert_eq!(perf.n_iterations, 0, "iterated an already-solved system");
        assert!(perf.converged);
    }

    /// The `SPEC-LIT` §8.4 normalisation, against the same formula evaluated
    /// on the host. It is our own design, so nothing but the specification
    /// text defines it - which makes an independent evaluation the only real
    /// check available.
    #[test]
    fn the_norm_factor_follows_the_specification() {
        let n = 19;
        let Some(r) = rig(n, 2024, false) else { return };

        let hpsi: Vec<f64> = noise(n, 55).iter().map(|v| 0.5 + v).collect();
        let dpsi: Vec<Scalar> = hpsi.iter().map(|v| *v as Scalar).collect();
        let psi = r.gpu.upload(&dpsi).expect("psi");
        let mut w = SolverWorkspace::for_mesh(&r.gpu, &r.m).expect("workspace");

        device_norm_factor(&r.gpu, &r.k, &mut w, &psi, &r.a, &r.m).expect("norm");
        let got = r.gpu.download(&w.norm_factor).expect("dl")[0] as f64;

        // x_ref = mean(psi); norm = sum|A psi - A x_ref| + sum|b - A x_ref|
        let x_ref = hpsi.iter().sum::<f64>() / n as f64;
        let a_psi = r.dense.matvec(&hpsi);
        let a_ref = r.dense.matvec(&vec![x_ref; n]);
        let want: f64 = a_psi
            .iter()
            .zip(&a_ref)
            .map(|(p, q)| (p - q).abs())
            .sum::<f64>()
            + r.b
                .iter()
                .zip(&a_ref)
                .map(|(p, q)| (p - q).abs())
                .sum::<f64>();

        assert!(
            (got - want).abs() / want < ROUNDOFF,
            "norm factor {got:.15e} vs specification {want:.15e}"
        );
    }

    /// The claim the whole file is built around: with `fixed_iters` set and
    /// `report_residuals` off, a solve makes no host round-trip at all - so
    /// the CUDA graph capture, which fails outright on any synchronisation or
    /// read-back inside it, succeeds.
    #[test]
    fn a_fixed_iteration_solve_captures_into_a_cuda_graph() {
        let n = 29;
        let Some(r) = rig(n, 606, false) else { return };

        let mut psi: DevBuf<Scalar> = r.gpu.zeros(n).expect("psi");
        let mut w = SolverWorkspace::for_mesh(&r.gpu, &r.m).expect("workspace");

        let ctrl = SolverControls {
            max_iter: 40,
            fixed_iters: true,
            report_residuals: false,
            ..tight()
        };

        // Warm the answer up outside the capture, so what the graph replays is
        // an ordinary steady-state iteration rather than the first one.
        solve_pbicgstab(&r.gpu, &r.k, &mut psi, &r.a, &r.m, &mut w, &ctrl).expect("warm");

        let captured = r
            .gpu
            .capture(|_| {
                solve_pbicgstab(&r.gpu, &r.k, &mut psi, &r.a, &r.m, &mut w, &ctrl).map(|_| ())
            })
            .expect("capture failed - the solve made a host round-trip");

        let mut graph = captured.expect("capture produced no work");
        graph.upload().expect("upload");
        graph.launch().expect("launch");
        r.gpu.sync().expect("sync");

        let got: Vec<f64> = r
            .gpu
            .download(&psi)
            .expect("dl")
            .iter()
            .map(|v| *v as f64)
            .collect();
        let direct = r.dense.direct_solve(&r.b);
        let err = max_diff(&got, &direct);
        assert!(err < SLACK, "the replayed graph solved to {err:.3e}");
    }

    /// `fixed_iters` runs exactly the sweeps it was told to and reports them.
    #[test]
    fn fixed_iterations_run_exactly_the_requested_sweeps() {
        let n = 29;
        let Some(r) = rig(n, 11, false) else { return };

        let mut psi: DevBuf<Scalar> = r.gpu.zeros(n).expect("psi");
        let mut w = SolverWorkspace::for_mesh(&r.gpu, &r.m).expect("workspace");

        let ctrl = SolverControls {
            max_iter: 3,
            fixed_iters: true,
            report_residuals: true,
            ..tight()
        };
        let perf = solve_pbicgstab(&r.gpu, &r.k, &mut psi, &r.a, &r.m, &mut w, &ctrl)
            .expect("solve");

        assert_eq!(perf.n_iterations, 3);
        assert!(
            perf.final_residual < perf.initial_residual,
            "three sweeps did not reduce the residual: {perf:?}"
        );
    }

    /// A larger `check_interval` may overshoot, but never by more than
    /// `check_interval - 1` sweeps, and never changes the answer.
    #[test]
    fn a_wider_check_interval_only_overshoots() {
        let n = 29;
        let Some(r) = rig(n, 313, false) else { return };
        let mut w = SolverWorkspace::for_mesh(&r.gpu, &r.m).expect("workspace");

        let mut a1: DevBuf<Scalar> = r.gpu.zeros(n).expect("psi");
        let every = SolverControls { check_interval: 1, ..tight() };
        let p1 = solve_pbicgstab(&r.gpu, &r.k, &mut a1, &r.a, &r.m, &mut w, &every)
            .expect("every");

        let mut a5: DevBuf<Scalar> = r.gpu.zeros(n).expect("psi");
        let every5 = SolverControls { check_interval: 5, ..tight() };
        let p5 = solve_pbicgstab(&r.gpu, &r.k, &mut a5, &r.a, &r.m, &mut w, &every5)
            .expect("every 5");

        assert!(p1.converged && p5.converged);
        assert!(
            p5.n_iterations >= p1.n_iterations && p5.n_iterations < p1.n_iterations + 5,
            "interval 5 took {} sweeps against {}",
            p5.n_iterations,
            p1.n_iterations
        );

        let direct = r.dense.direct_solve(&r.b);
        for buf in [&a1, &a5] {
            let got: Vec<f64> = r
                .gpu
                .download(buf)
                .expect("dl")
                .iter()
                .map(|v| *v as f64)
                .collect();
            assert!(max_diff(&got, &direct) < SLACK);
        }
    }
    // ======================================================================
    //  SPEC-LIT 21 / 22: multi-colour DIC and DILU
    //
    //  On a STRUCTURED mesh, not on `dense_mesh`: a complete graph needs one
    //  colour per cell, which is a legitimate colouring and a useless one to
    //  measure an iteration count against. A hex mesh needs two, which is what
    //  SPEC-LIT 21 says and what these tests rely on.
    // ======================================================================

    use crate::precon::{Adjacency, Colouring, MultiColour};

    /// A structured mesh plus an LDU matrix that is strictly diagonally
    /// dominant with a positive diagonal - hence, when symmetric, positive
    /// definite by Gershgorin, which is what PCG needs.
    struct Structured {
        gpu: Gpu,
        k: SolverKernels,
        hm: HostMesh,
        m: GpuMesh,
        a: GpuLduMatrix,
        exact: Vec<f64>,
        diag: Vec<f64>,
        upper: Vec<f64>,
        lower: Vec<f64>,
    }

    fn structured(n: [usize; 3], symmetric: bool) -> Option<Structured> {
        let d = Vec3::new(1.0 / n[0] as Scalar, 1.0 / n[1] as Scalar, 1.0 / n[2] as Scalar);
        let (mut hm, pts, faces) = crate::mesh::topology::tests::box_mesh(n, d);
        for p in hm.patches.iter_mut() {
            p.kind = PatchKind::Generic;
            p.type_name = "patch".to_string();
        }
        hm.build_cell_face_maps();
        hm.compute_geometry(&pts, &faces).ok()?;

        let nc = hm.n_cells;
        let nf = hm.n_internal_faces;

        // Off-diagonals that vary from face to face, so the factorisation is
        // not accidentally uniform and cannot pass by symmetry alone.
        let upper: Vec<f64> = (0..nf)
            .map(|f| -(1.0 + 0.4 * ((f as f64) * 0.7).sin()))
            .collect();
        let lower: Vec<f64> = if symmetric {
            upper.clone()
        } else {
            (0..nf)
                .map(|f| -(1.0 + 0.4 * ((f as f64) * 1.3).cos()))
                .collect()
        };

        // diag[i] = 1 + 1.05 * sum_j |A(i,j)|: strictly dominant, positive.
        let mut row = vec![0.0f64; nc];
        for f in 0..nf {
            row[hm.owner[f] as usize] += upper[f].abs();
            row[hm.neighbour[f] as usize] += lower[f].abs();
        }
        let diag: Vec<f64> = row.iter().map(|s| 1.0 + 1.05 * s).collect();

        let exact: Vec<f64> = noise(nc, 4242).iter().map(|v| 1.0 + 2.0 * v).collect();

        // b = A x, formed here so the answer is known independently of the
        // solver that is about to be tested.
        let mut b = vec![0.0f64; nc];
        for c in 0..nc {
            b[c] = diag[c] * exact[c];
        }
        for f in 0..nf {
            let (o, nb) = (hm.owner[f] as usize, hm.neighbour[f] as usize);
            b[o] += upper[f] * exact[nb];
            b[nb] += lower[f] * exact[o];
        }

        let gpu = Gpu::new(0).ok()?;
        let k = SolverKernels::new(&gpu).ok()?;
        let m = GpuMesh::upload(&gpu, &hm).ok()?;

        let mut a = GpuLduMatrix::new(&gpu, &m).ok()?;
        a.zero(&gpu).ok()?;
        gpu.write(&mut a.diag, &diag.iter().map(|v| *v as Scalar).collect::<Vec<_>>())
            .ok()?;
        gpu.write(&mut a.upper, &upper.iter().map(|v| *v as Scalar).collect::<Vec<_>>())
            .ok()?;
        gpu.write(&mut a.lower, &lower.iter().map(|v| *v as Scalar).collect::<Vec<_>>())
            .ok()?;
        gpu.write(&mut a.source, &b.iter().map(|v| *v as Scalar).collect::<Vec<_>>())
            .ok()?;

        Some(Structured { gpu, k, hm, m, a, exact, diag, upper, lower })
    }

    /// The factorisation and the two sweeps, on the host, from the definition.
    ///
    /// Deliberately written as a scalar loop over cells in colour order - the
    /// opposite of what the device does, which is one launch per colour with
    /// no order inside it. The two agreeing is the statement that the
    /// per-colour schedule really does reproduce the sequential elimination.
    fn host_dilu(st: &Structured, col: &Colouring, x: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let hm = &st.hm;
        let nc = hm.n_cells;

        let coeff = |_c: usize, j: usize| -> (usize, f64, f64) {
            let f = hm.cf_face[j] as usize;
            let owner = hm.cf_own[j] != 0;
            let nbr = if owner { hm.neighbour[f] } else { hm.owner[f] } as usize;
            // A(c, nbr) and A(nbr, c).
            let (a_cn, a_nc) = if owner {
                (st.upper[f], st.lower[f])
            } else {
                (st.lower[f], st.upper[f])
            };
            (nbr, a_cn, a_nc)
        };

        // Cells in COLOUR order, which is the ordering the factorisation is
        // defined against. Index order would be wrong and not obviously so: a
        // neighbour of a lower colour may have a higher index, and its rD
        // would still be zero when this cell read it.
        let mut r_d = vec![0.0f64; nc];
        for colour in 0..col.n_colours {
            for i in col.offsets[colour]..col.offsets[colour + 1] {
                let c = col.cells[i] as usize;
                let mut d = st.diag[c];
                for j in hm.cf_offset[c] as usize..hm.cf_offset[c + 1] as usize {
                    let (nbr, a_cn, a_nc) = coeff(c, j);
                    if col.colour[nbr] < col.colour[c] {
                        d -= a_cn * a_nc * r_d[nbr];
                    }
                }
                r_d[c] = 1.0 / d;
            }
        }

        let mut y = x.to_vec();
        for colour in 0..col.n_colours {
            for i in col.offsets[colour]..col.offsets[colour + 1] {
                let c = col.cells[i] as usize;
                let mut acc = 0.0;
                for j in hm.cf_offset[c] as usize..hm.cf_offset[c + 1] as usize {
                    let (nbr, a_cn, _) = coeff(c, j);
                    if col.colour[nbr] < col.colour[c] {
                        acc += a_cn * y[nbr];
                    }
                }
                y[c] = r_d[c] * (y[c] - acc);
            }
        }
        for colour in (0..col.n_colours).rev() {
            for i in col.offsets[colour]..col.offsets[colour + 1] {
                let c = col.cells[i] as usize;
                let mut acc = 0.0;
                for j in hm.cf_offset[c] as usize..hm.cf_offset[c + 1] as usize {
                    let (nbr, a_cn, _) = coeff(c, j);
                    if col.colour[nbr] > col.colour[c] {
                        acc += a_cn * y[nbr];
                    }
                }
                y[c] -= r_d[c] * acc;
            }
        }

        (r_d, y)
    }

    /// The device factorisation and sweeps against the host mirror above.
    #[test]
    fn the_multicolour_sweeps_are_the_incomplete_factorisation() {
        for symmetric in [true, false] {
            let Some(st) = structured([5, 4, 3], symmetric) else { return };
            let g = Adjacency::of(&st.hm);
            let col = Colouring::greedy(&g);
            let mc = MultiColour::from_colouring(&st.gpu, &g, &col).expect("colouring");

            let nc = st.hm.n_cells;
            let x: Vec<f64> = noise(nc, 77).iter().map(|v| 1.0 + v).collect();

            let mut r_diag: DevBuf<Scalar> = st.gpu.zeros(nc).expect("rD");
            mc.factorise(&st.gpu, &mut r_diag, &st.a, &st.m, symmetric)
                .expect("factorise");
            let mut y = st
                .gpu
                .upload(&x.iter().map(|v| *v as Scalar).collect::<Vec<_>>())
                .expect("y");
            mc.apply(&st.gpu, &mut y, &r_diag, &st.a, &st.m).expect("apply");
            st.gpu.sync().expect("sync");

            let (want_rd, want_y) = host_dilu(&st, &col, &x);
            let got_rd: Vec<f64> = st
                .gpu
                .download(&r_diag)
                .expect("dl")
                .iter()
                .map(|v| *v as f64)
                .collect();
            let got_y: Vec<f64> = st
                .gpu
                .download(&y)
                .expect("dl")
                .iter()
                .map(|v| *v as f64)
                .collect();

            assert!(
                max_diff(&got_rd, &want_rd) <= ROUNDOFF * 10.0,
                "symmetric={symmetric}: rD differs by {}",
                max_diff(&got_rd, &want_rd)
            );
            assert!(
                max_diff(&got_y, &want_y) <= ROUNDOFF * 10.0,
                "symmetric={symmetric}: M^-1 x differs by {}",
                max_diff(&got_y, &want_y)
            );
        }
    }

    fn solve_with(
        st: &mut Structured,
        precon: Preconditioner,
        kind: LinearSolverKind,
    ) -> (Vec<f64>, usize) {
        let nc = st.hm.n_cells;
        let mut w = SolverWorkspace::for_mesh(&st.gpu, &st.m).expect("workspace");
        let mut x: DevBuf<Scalar> = st.gpu.zeros(nc).expect("x");
        let ctrl = SolverControls {
            solver: kind,
            tolerance: SOLVE_TOL,
            rel_tol: 0.0,
            max_iter: 2000,
            precon,
            ..SolverControls::default()
        };
        let perf = solve(&st.gpu, &st.k, &mut x, &st.a, &st.m, &mut w, &ctrl).expect("solve");
        assert!(perf.converged, "{precon:?} did not converge");
        let got: Vec<f64> = st
            .gpu
            .download(&x)
            .expect("dl")
            .iter()
            .map(|v| *v as f64)
            .collect();
        (got, perf.n_iterations)
    }

    /// `SPEC-LIT` §22: multi-colour DIC vs unpreconditioned - same answer,
    /// fewer iterations.
    #[test]
    fn dic_reaches_the_same_answer_in_fewer_iterations() {
        let Some(mut st) = structured([8, 7, 6], true) else { return };

        let (none, it_none) = solve_with(&mut st, Preconditioner::None, LinearSolverKind::PCG);
        let (dic, it_dic) = solve_with(&mut st, Preconditioner::Dic, LinearSolverKind::PCG);

        assert!(
            max_diff(&none, &st.exact) < SLACK * 1e3,
            "the unpreconditioned solve did not reach the known answer"
        );
        assert!(
            max_diff(&none, &dic) < SLACK * 1e3,
            "DIC and no preconditioner disagree by {}",
            max_diff(&none, &dic)
        );
        println!(
            "multi-colour DIC: {it_dic} iterations against {it_none}              unpreconditioned, {} colours",
            SolverWorkspace::for_mesh(&st.gpu, &st.m)
                .expect("workspace")
                .multicolour
                .as_ref()
                .map(|m| m.n_colours())
                .unwrap_or(0)
        );
        assert!(
            it_dic < it_none,
            "DIC took {it_dic} iterations against {it_none} unpreconditioned"
        );
    }

    /// The same for DILU on the asymmetric system BiCGStab is for.
    #[test]
    fn dilu_reaches_the_same_answer_in_fewer_iterations() {
        let Some(mut st) = structured([8, 7, 6], false) else { return };

        let (none, it_none) =
            solve_with(&mut st, Preconditioner::None, LinearSolverKind::PBiCGStab);
        let (dilu, it_dilu) =
            solve_with(&mut st, Preconditioner::Dilu, LinearSolverKind::PBiCGStab);

        assert!(max_diff(&none, &st.exact) < SLACK * 1e3);
        assert!(
            max_diff(&none, &dilu) < SLACK * 1e3,
            "DILU and no preconditioner disagree by {}",
            max_diff(&none, &dilu)
        );
        println!("multi-colour DILU: {it_dilu} iterations against {it_none} unpreconditioned");
        assert!(
            it_dilu < it_none,
            "DILU took {it_dilu} iterations against {it_none} unpreconditioned"
        );
    }

    /// `SPEC-LIT` §21/§22: the iteration count must not change when the colour
    /// ordering changes.
    ///
    /// Two things are being asked, and they are different:
    ///
    /// * shuffling the cells INSIDE each colour changes only which thread
    ///   handles which cell. Because no two cells of a colour are neighbours,
    ///   that must be **bitwise** identical - this is the schedule-independence
    ///   §21 is built for, and it is what makes the whole thing usable on a GPU.
    /// * reversing the colour LABELS is a different elimination ordering and so
    ///   a different (equally valid) factorisation. Its iteration count is
    ///   asserted to be the same only to within the one iteration a Krylov
    ///   method may legitimately differ by, and the ANSWER must be the same.
    #[test]
    fn the_colour_ordering_does_not_change_the_iteration_count() {
        let Some(st) = structured([8, 7, 6], false) else { return };
        let nc = st.hm.n_cells;
        let g = Adjacency::of(&st.hm);
        let base = Colouring::greedy(&g);

        let run = |col: &Colouring| -> (Vec<f64>, usize) {
            let mut w = SolverWorkspace::for_mesh(&st.gpu, &st.m).expect("workspace");
            w.multicolour =
                Some(MultiColour::from_colouring(&st.gpu, &g, col).expect("colouring"));
            let mut x: DevBuf<Scalar> = st.gpu.zeros(nc).expect("x");
            let ctrl = SolverControls {
                solver: LinearSolverKind::PBiCGStab,
                tolerance: SOLVE_TOL,
                rel_tol: 0.0,
                max_iter: 2000,
                precon: Preconditioner::Dilu,
                ..SolverControls::default()
            };
            let perf =
                solve(&st.gpu, &st.k, &mut x, &st.a, &st.m, &mut w, &ctrl).expect("solve");
            assert!(perf.converged);
            let got: Vec<f64> = st
                .gpu
                .download(&x)
                .expect("dl")
                .iter()
                .map(|v| *v as f64)
                .collect();
            (got, perf.n_iterations)
        };

        let (x0, it0) = run(&base);
        let (x1, it1) = run(&base.with_shuffled_cells_within_colours(0xC0FFEE));
        assert_eq!(
            it0, it1,
            "shuffling the cells within a colour changed the iteration count \
             from {it0} to {it1}; the sweep is not schedule-independent"
        );
        assert_eq!(
            x0, x1,
            "shuffling the cells within a colour changed the answer"
        );

        println!(
            "{} colours; base {it0} iterations, cells shuffled within colours {it1}",
            base.n_colours
        );
        let (x2, it2) = run(&base.with_reversed_colour_order());
        println!("colour order reversed: {it2} iterations");
        assert!(
            max_diff(&x0, &x2) < SLACK * 1e3,
            "reversing the colour order changed the answer by {}",
            max_diff(&x0, &x2)
        );
        assert!(
            it2.abs_diff(it0) <= 1,
            "reversing the colour order moved the iteration count from {it0} \
             to {it2}"
        );
    }

    // ----------------------------------------------------------------------
    //  SPEC-LIT 13.4: the request must be honoured or refused
    // ----------------------------------------------------------------------

    #[test]
    fn symmetry_is_detected_both_ways() {
        let Some(sym) = structured([4, 3, 3], true) else { return };
        let mut w = SolverWorkspace::for_mesh(&sym.gpu, &sym.m).expect("workspace");
        assert!(matrix_is_symmetric(&sym.gpu, &sym.k, &mut w, &sym.a, &sym.m).expect("check"));

        let Some(asym) = structured([4, 3, 3], false) else { return };
        let mut w = SolverWorkspace::for_mesh(&asym.gpu, &asym.m).expect("workspace");
        assert!(
            !matrix_is_symmetric(&asym.gpu, &asym.k, &mut w, &asym.a, &asym.m).expect("check")
        );
    }

    /// SPEC-LIT §48.3, the "no false positives" half. On an UNCOUPLED mesh the
    /// coupled stage measures exactly zero, so the verdict and the reported
    /// face defect are what they were before §48.3 extended the check.
    #[test]
    fn the_coupled_stage_is_silent_on_an_uncoupled_mesh() {
        let Some(sym) = structured([4, 3, 3], true) else { return };
        let mut w = SolverWorkspace::for_mesh(&sym.gpu, &sym.m).expect("workspace");
        let d = symmetry_defects(&sym.gpu, &sym.k, &mut w, &sym.a, &sym.m).expect("defects");
        assert_eq!(d.coupled, 0.0, "an uncoupled mesh has no coupled pair");
        assert_eq!(d.coupled_scale, 0.0);
        assert!(d.coupled_is_symmetric());
        assert!(d.is_symmetric());
        assert_eq!(d.what_failed(), "nothing");
    }

    /// SPEC-LIT §48.3, the half that was a blind spot. A matrix whose two
    /// coupled boundary coefficients differ is asymmetric, and used to be
    /// reported symmetric because nothing looked at them.
    #[test]
    fn unequal_coupled_boundary_coefficients_are_asymmetric() {
        use crate::ldu_ops::tests as ldut;

        let Some(gpu) = crate::Gpu::new(0).ok() else { return };
        let hm = ldut::chain_mesh();
        let m = crate::mesh::GpuMesh::upload(&gpu, &hm).expect("upload");
        let k = SolverKernels::new(&gpu).expect("kernels");
        let mut w = SolverWorkspace::for_mesh(&gpu, &m).expect("workspace");
        let mut a = GpuLduMatrix::new(&gpu, &m).expect("matrix");

        // A symmetric face part, so only the coupled pair is in question.
        let up = vec![-1.0 as Scalar; hm.n_internal_faces];
        gpu.write(&mut a.upper, &up).expect("upper");
        gpu.write(&mut a.lower, &up).expect("lower");

        // The two faces of the couple, equal first.
        let mut bc = vec![0.0 as Scalar; hm.n_boundary_faces];
        let mut pair = None;
        for bf in 0..hm.n_boundary_faces {
            if hm.b_nbr_face[bf] >= 0 {
                pair = Some((bf, hm.b_nbr_face[bf] as usize));
                break;
            }
        }
        let (i, j) = pair.expect("the chain mesh has a cyclic pair");
        bc[i] = -3.0;
        bc[j] = -3.0;
        gpu.write(&mut a.boundary_coeffs, &bc).expect("bc");
        let d = symmetry_defects(&gpu, &k, &mut w, &a, &m).expect("defects");
        assert!(d.is_symmetric(), "equal coupled coefficients are symmetric");

        // Now make them differ. Nothing else about the matrix changes.
        bc[j] = -3.5;
        gpu.write(&mut a.boundary_coeffs, &bc).expect("bc");
        let d = symmetry_defects(&gpu, &k, &mut w, &a, &m).expect("defects");
        assert!(
            d.face_is_symmetric(),
            "upper and lower are still equal: face defect {}",
            d.face
        );
        assert!(
            !d.coupled_is_symmetric(),
            "A(P,Q) = -3 and A(Q,P) = -3.5 is not a symmetric matrix; defect {}",
            d.coupled
        );
        assert!(!d.is_symmetric());
        assert!(
            d.what_failed().contains("COUPLED"),
            "the message must name which half failed: {}",
            d.what_failed()
        );
    }

    /// The point of the whole exercise: `solver PCG;` on an asymmetric system
    /// used to run PBiCGStab and say nothing. Now it is refused.
    #[test]
    fn pcg_on_an_asymmetric_matrix_is_an_error() {
        let Some(st) = structured([4, 3, 3], false) else { return };
        let mut w = SolverWorkspace::for_mesh(&st.gpu, &st.m).expect("workspace");
        let mut x: DevBuf<Scalar> = st.gpu.zeros(st.hm.n_cells).expect("x");
        let ctrl = SolverControls {
            solver: LinearSolverKind::PCG,
            ..SolverControls::default()
        };
        let e = solve(&st.gpu, &st.k, &mut x, &st.a, &st.m, &mut w, &ctrl)
            .expect_err("PCG on an asymmetric matrix must be refused")
            .to_string();
        assert!(e.contains("PCG"), "{e}");
        assert!(e.contains("PBiCGStab"), "{e}");
    }

    #[test]
    fn dic_on_an_asymmetric_matrix_is_an_error_that_names_dilu() {
        let Some(st) = structured([4, 3, 3], false) else { return };
        let mut w = SolverWorkspace::for_mesh(&st.gpu, &st.m).expect("workspace");
        let mut x: DevBuf<Scalar> = st.gpu.zeros(st.hm.n_cells).expect("x");
        let ctrl = SolverControls {
            solver: LinearSolverKind::PBiCGStab,
            precon: Preconditioner::Dic,
            ..SolverControls::default()
        };
        let e = solve(&st.gpu, &st.k, &mut x, &st.a, &st.m, &mut w, &ctrl)
            .expect_err("DIC on an asymmetric matrix must be refused")
            .to_string();
        assert!(e.contains("DILU"), "{e}");
    }

    #[test]
    fn gamg_is_refused_with_the_reason() {
        let Some(st) = structured([4, 3, 3], true) else { return };
        let mut w = SolverWorkspace::for_mesh(&st.gpu, &st.m).expect("workspace");
        let mut x: DevBuf<Scalar> = st.gpu.zeros(st.hm.n_cells).expect("x");
        let ctrl = SolverControls {
            solver: LinearSolverKind::Gamg,
            ..SolverControls::default()
        };
        let e = solve(&st.gpu, &st.k, &mut x, &st.a, &st.m, &mut w, &ctrl)
            .expect_err("GAMG must be refused outside the pressure backend")
            .to_string();
        assert!(e.contains("GAMG"), "{e}");
        assert!(e.contains("AMGX"), "{e}");
    }

    /// A mesh-free workspace has no graph to colour, so DIC/DILU is an error
    /// naming the setting - not a silent Jacobi, which is what this used to be.
    #[test]
    fn dic_without_a_colouring_is_an_error_and_not_a_silent_jacobi() {
        let _guard = crate::io::contract::permissive_test_guard();
        let Some(gpu) = Gpu::new(0).ok() else { return };
        let w = SolverWorkspace::new(&gpu, 16).expect("workspace");
        crate::io::contract::set_permissive(false);
        let e = effective_preconditioner(Preconditioner::Dilu, &w)
            .expect_err("must be refused")
            .to_string();
        assert!(e.contains("DILU"), "{e}");

        // And a workspace that HAS one honours the request rather than
        // downgrading it.
        assert_eq!(
            effective_preconditioner(Preconditioner::Diagonal, &w).expect("jacobi"),
            Preconditioner::Diagonal
        );
    }
}
