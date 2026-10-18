// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

//! The block-coupled solve of SPEC-LIT §109.5: §8.1's BiCGStab run over the
//! block matrix of §109 as ONE Krylov system of `3 n_cells` unknowns, with
//! §21's multi-colour no-fill factorisation written with `3x3` blocks and a
//! block-Jacobi comparison.
//!
//! The flat layout (109.8): every Krylov vector is a `DevBuf<Scalar>` of
//! length `3 n_cells`, cell-major - `x[3 c + i]` is component `i` of cell
//! `c` - which is byte for byte a `DevBuf<Vec3>` of `n_cells`, so the scalar
//! solver's kernels, reductions and fused updates run on it unchanged with
//! `n = 3 n_cells` and the block product launches the SAME `solidBlockAmul`
//! kernel the typed operator launches, over flat buffers. The product is
//! §109.1's ordinary block product, a row acting on a vector:
//! `(A x)_c = diag[c] x_c + sum_{f: owner[f] = c} upper[f] x_neighbour[f]
//! + sum_{f: neighbour[f] = c} lower[f] x_owner[f]`, with
//! `(T v)_i = sum_j T_ij v_j` for a block `T`.
//!
//! One Krylov method: §8.1's PBiCGStab, the loop the scalar systems run,
//! over this length - the matrix is symmetric (§109.3), so §8.2's conjugate
//! gradient would be lawful too, and it is a later choice, not a correction.
//! Two preconditioners: §21's multi-colour no-fill factorisation with a `3x3`
//! block where §21 has a number, every product kept in its order because
//! blocks do not commute ([`BlockPrecon::Dilu`]), and the `3x3` diagonal
//! inverse ([`BlockPrecon::Jacobi`], §8.3's `M = diag(A)`).
//!
//! `numerics.solver` and `precon` name the SCALAR systems' method and
//! preconditioner and are NOT read here: the block solve always runs
//! PBiCGStab with the [`BlockPrecon`] it is given, and a `GAMG` request is
//! refused by name.
//!
//! The §8.4 residual normalisation is applied literally to the flat vector
//! (109.11): `x_ref` is the mean of ALL `3 n_cells` entries broadcast to
//! every entry - a constant vector field, still in the null space of a
//! traction-only operator, which is the property §8.4 asks of it.
//!
//! Written from:
//!   Saad, *Iterative Methods for Sparse Linear Systems*, 2nd ed. (2003),
//!     ch. 10 and ch. 12 - the no-fill incomplete factorisation and the
//!     multi-colour ordering that makes it parallel
//!   van der Vorst, *SIAM J. Sci. Stat. Comput.* 13 (1992) 631-644 - the
//!     BiCGStab loop, run unchanged over the flat system
//!   P. Cardiff, Ž. Tuković, H. Jasak, A. Ivanković, *Comput. Struct.* 175
//!     (2016) 100-122, DOI 10.1016/j.compstruc.2016.07.004 - the IDEA of
//!     solving the three displacement components in one matrix; cited for
//!     that and for nothing else
//!   ofgpu `SPEC-LIT.md` §8 (the solvers, their residual normalisation),
//!     §21 (the colouring and the per-colour sweep order), §109 (the block
//!     matrix this solves)
//!
//! The scalar solver's fused updates are CALLED, not copied: `src/solver.rs`
//! gains visibility only. No GPL-licensed source was consulted.

use crate::error::{Error, Result};
use crate::mesh::GpuMesh;
use crate::precon::MultiColour;
use crate::solid::block::GpuBlockLdu;
use crate::solver::{
    bicg_p_update, bicg_r_update, bicg_rho_and_beta, bicg_s_update, bicg_x_update,
    check_workspace, collect_report, convergence_test, copy_scalar, device_sum, device_sum_mag,
    dot2_then_divide, dot_then_divide, finish_sum, read_flag, refuse_round_trip_in_capture,
    set_scalar, sum_mag_then_test, to_label, vec_copy, vec_sub, NORM_EPS, SolverControls,
    SolverKernels, SolverPerformance, SolverWorkspace,
};
use crate::{io::case::LinearSolverKind, Label, Scalar, Tensor};

use crate::device::{cfg_for, DevBuf, Gpu, KernelSet};
use cudarc::driver::{CudaFunction, PushKernelArg};

// ==========================================================================
//  The preconditioner choice of SPEC-LIT §109.5.
// ==========================================================================

/// Which preconditioner the block solve runs (§109.5): the multi-colour
/// no-fill `3x3`-block factorisation, or the `3x3` diagonal inverse.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlockPrecon {
    /// `rD_v = diag[v]^-1`, `y_v = rD_v x_v` (109.10).
    Jacobi,
    /// §21's factorisation with every scalar replaced by its `3x3` block,
    /// every product kept in its order (109.9).
    Dilu,
}

/// `cuda/solidblock.cu`'s solve-side entry points, resolved once: the block
/// product of §109.1 and the five kernels of (109.9)/(109.10).
pub struct CoupledKernels {
    amul: CudaFunction,
    factor: CudaFunction,
    forward: CudaFunction,
    backward: CudaFunction,
    invert_diag: CudaFunction,
    jacobi: CudaFunction,
}

impl CoupledKernels {
    pub fn new(gpu: &Gpu) -> Result<Self> {
        let k = KernelSet::new(gpu, crate::kernels::SOLIDBLOCK)?;
        Ok(Self {
            amul: k.func("solidBlockAmul")?,
            factor: k.func("solidBlockFactorColour")?,
            forward: k.func("solidBlockForwardColour")?,
            backward: k.func("solidBlockBackwardColour")?,
            invert_diag: k.func("solidBlockInvertDiag")?,
            jacobi: k.func("solidBlockJacobi")?,
        })
    }
}

/// The flat Krylov vectors of the block system (109.8) and the scratch the
/// scalar loop shares with them, sized `3 n_cells` by
/// [`SolverWorkspace::new`], with the colouring already built - a time loop
/// and a CUDA graph may not allocate, so everything here is made once.
pub struct CoupledWorkspace {
    pub n_cells: usize,
    /// The scalar workspace, `SolverWorkspace::new(gpu, 3 * n_cells)`.
    pub w: SolverWorkspace,
    /// The unknown, `[3 n_cells]` (109.8).
    pub x: DevBuf<Scalar>,
    /// The flat copy of `a.source`, loaded by the caller before a solve.
    pub b: DevBuf<Scalar>,
    /// `[n_cells]`, `rD` of (109.9)/(109.10).
    pub r_diag: DevBuf<Tensor>,
    /// The colouring the `Dilu` sweeps walk.
    pub colouring: MultiColour,
}

impl CoupledWorkspace {
    /// Colour the mesh and size everything for it. Setup only.
    pub fn for_mesh(gpu: &Gpu, m: &GpuMesh) -> Result<Self> {
        let colouring = MultiColour::new(gpu, m)?;
        Self::with_colouring(gpu, m.n_cells, colouring)
    }

    /// The same, from a colouring the caller made - the schedule-independence
    /// test hands a shuffled one.
    pub fn with_colouring(gpu: &Gpu, n_cells: usize, colouring: MultiColour) -> Result<Self> {
        let n = n_cells * 3;
        Ok(Self {
            n_cells,
            w: SolverWorkspace::new(gpu, n)?,
            x: gpu.zeros(n)?,
            b: gpu.zeros(n)?,
            r_diag: gpu.zeros(n_cells.max(1))?,
            colouring,
        })
    }
}

// ==========================================================================
//  The primitives
// ==========================================================================

/// (109.2) on the flat layout: `y = A x`, launching the SAME
/// `solidBlockAmul` kernel [`crate::solid::block::BlockOperator::amul`]
/// launches, with flat `y`/`x` of length `3 n_cells`. One thread per cell
/// gathers its row over `cf_offset/cf_face/cf_own` in CSR order, so the
/// summation order per row - and therefore the product, bit for bit - is the
/// typed one.
pub fn block_amul(
    gpu: &Gpu,
    ck: &CoupledKernels,
    y: &mut DevBuf<Scalar>,
    x: &DevBuf<Scalar>,
    a: &GpuBlockLdu,
    m: &GpuMesh,
) -> Result<()> {
    let n = a.n_cells;
    if m.n_cells != n {
        return Err(Error::Config(format!(
            "block amul: the matrix has {n} cells, the mesh has {}",
            m.n_cells
        )));
    }
    if y.len() < n * 3 || x.len() < n * 3 {
        return Err(Error::Config(format!(
            "block amul: y has {} and x has {} elements, the system has {}",
            y.len(),
            x.len(),
            n * 3
        )));
    }
    if n == 0 {
        return Ok(());
    }
    let nl = to_label(n)?;
    unsafe {
        gpu.stream()
            .launch_builder(&ck.amul)
            .arg(&mut *y)
            .arg(x)
            .arg(&a.diag)
            .arg(&a.upper)
            .arg(&a.lower)
            .arg(&m.owner)
            .arg(&m.neighbour)
            .arg(&m.cf_offset)
            .arg(&m.cf_face)
            .arg(&m.cf_own)
            .arg(&nl)
            .launch(cfg_for(n))?;
    }
    Ok(())
}

/// The shape checks both [`build_block_preconditioner`] and
/// [`solve_block_pbicgstab`] run before anything launches.
fn check_shapes(a: &GpuBlockLdu, m: &GpuMesh, cw: &CoupledWorkspace) -> Result<()> {
    if a.n_cells != m.n_cells {
        return Err(Error::Config(format!(
            "block solve: the matrix has {} cells, the mesh has {}",
            a.n_cells, m.n_cells
        )));
    }
    if cw.n_cells != a.n_cells {
        return Err(Error::Config(format!(
            "block solve: the workspace is sized for {} cells, the matrix \
             has {}",
            cw.n_cells, a.n_cells
        )));
    }
    Ok(())
}

/// Refuse `GAMG` by name: the block solve is PBiCGStab, and GAMG is the
/// pressure backend here - `crate::solver::solve`'s own reason, quoted.
fn refuse_gamg(ctrl: &SolverControls) -> Result<()> {
    if ctrl.solver == LinearSolverKind::Gamg {
        return Err(Error::Config(
            "block solve: GAMG was requested. The block-coupled solve of \
             SPEC-LIT 109.5 is always PBiCGStab - numerics.solver names the \
             scalar systems' method and is not read here. The scalar side's \
             own refusal stands: solver GAMG was requested. Algebraic \
             multigrid is not a Krylov method and is not reimplemented here \
             (SPEC-LIT 8.3): it reaches ofgpu only as the AMGX pressure \
             backend, which the pressure equation selects through \
             crate::pressure. For any other equation use PBiCGStab, or PCG \
             if the system is symmetric."
                .to_string(),
        ));
    }
    Ok(())
}

/// Build `cw.r_diag` for the chosen preconditioner: (109.9)'s factorisation
/// colour by colour in ascending order, or (109.10)'s diagonal inverse in
/// one launch. Setup only - `solve_block_pbicgstab` calls it at the top of
/// every solve, as the scalar solver builds its own.
pub fn build_block_preconditioner(
    gpu: &Gpu,
    ck: &CoupledKernels,
    cw: &mut CoupledWorkspace,
    a: &GpuBlockLdu,
    m: &GpuMesh,
    precon: BlockPrecon,
) -> Result<()> {
    check_shapes(a, m, cw)?;
    let n = a.n_cells;
    if n == 0 {
        return Ok(());
    }
    match precon {
        BlockPrecon::Dilu => {
            let nc = cw.colouring.colouring.n_colours;
            for c in 0..nc {
                let lo = cw.colouring.colouring.offsets[c];
                let count = cw.colouring.colouring.offsets[c + 1] - lo;
                if count == 0 {
                    continue;
                }
                let start = lo as Label;
                let count_l = count as Label;
                unsafe {
                    gpu.stream()
                        .launch_builder(&ck.factor)
                        .arg(&mut cw.r_diag)
                        .arg(&a.diag)
                        .arg(&a.upper)
                        .arg(&a.lower)
                        .arg(&cw.colouring.colouring.colour)
                        .arg(&cw.colouring.colouring.cells)
                        .arg(&m.owner)
                        .arg(&m.neighbour)
                        .arg(&m.cf_offset)
                        .arg(&m.cf_face)
                        .arg(&m.cf_own)
                        .arg(&start)
                        .arg(&count_l)
                        .launch(cfg_for(count))?;
                }
            }
        }
        BlockPrecon::Jacobi => {
            let nl = n as Label;
            unsafe {
                gpu.stream()
                    .launch_builder(&ck.invert_diag)
                    .arg(&mut cw.r_diag)
                    .arg(&a.diag)
                    .arg(&nl)
                    .launch(cfg_for(n))?;
            }
        }
    }
    Ok(())
}

/// `y = M^-1 x` (109.9)/(109.10). The workspace's pieces are passed
/// separately so that the mutable borrow of `y` (the caller's `p_hat` or
/// `s_hat`) does not collide with the shared `r_diag` and colouring.
/// `Dilu` copies `x` into `y` and sweeps in place, exactly
/// [`crate::precon::MultiColour::apply`]'s contract; `Jacobi` is one launch
/// reading `x` and writing `y`.
#[allow(clippy::too_many_arguments)]
pub fn precondition(
    gpu: &Gpu,
    k: &SolverKernels,
    ck: &CoupledKernels,
    y: &mut DevBuf<Scalar>,
    x: &DevBuf<Scalar>,
    cw_r_diag: &DevBuf<Tensor>,
    colouring: &MultiColour,
    a: &GpuBlockLdu,
    m: &GpuMesh,
    precon: BlockPrecon,
) -> Result<()> {
    let n = a.n_cells;
    if m.n_cells != n {
        return Err(Error::Config(format!(
            "block preconditioner: the matrix has {n} cells, the mesh has {}",
            m.n_cells
        )));
    }
    if y.len() < n * 3 || x.len() < n * 3 {
        return Err(Error::Config(format!(
            "block preconditioner: y has {} and x has {} elements, the \
             system has {}",
            y.len(),
            x.len(),
            n * 3
        )));
    }
    if n == 0 {
        return Ok(());
    }
    match precon {
        BlockPrecon::Dilu => {
            vec_copy(gpu, k, y, x, n * 3)?;
            let nc = colouring.colouring.n_colours;
            for c in 0..nc {
                sweep(gpu, &ck.forward, y, cw_r_diag, colouring, a, m, c)?;
            }
            for c in (0..nc).rev() {
                sweep(gpu, &ck.backward, y, cw_r_diag, colouring, a, m, c)?;
            }
        }
        BlockPrecon::Jacobi => {
            let nl = n as Label;
            unsafe {
                gpu.stream()
                    .launch_builder(&ck.jacobi)
                    .arg(&mut *y)
                    .arg(x)
                    .arg(cw_r_diag)
                    .arg(&nl)
                    .launch(cfg_for(n))?;
            }
        }
    }
    Ok(())
}

/// One colour's forward or backward sweep - [`crate::precon::MultiColour`]'s
/// private `sweep`, written out over the block kernels: `(start, count)`
/// from the colouring's own host `offsets`.
#[allow(clippy::too_many_arguments)]
fn sweep(
    gpu: &Gpu,
    f: &CudaFunction,
    y: &mut DevBuf<Scalar>,
    r_diag: &DevBuf<Tensor>,
    colouring: &MultiColour,
    a: &GpuBlockLdu,
    m: &GpuMesh,
    colour: usize,
) -> Result<()> {
    let lo = colouring.colouring.offsets[colour];
    let count = colouring.colouring.offsets[colour + 1] - lo;
    if count == 0 {
        return Ok(());
    }
    let start = lo as Label;
    let count_l = count as Label;
    unsafe {
        gpu.stream()
            .launch_builder(f)
            .arg(&mut *y)
            .arg(r_diag)
            .arg(&a.upper)
            .arg(&a.lower)
            .arg(&colouring.colouring.colour)
            .arg(&colouring.colouring.cells)
            .arg(&m.owner)
            .arg(&m.neighbour)
            .arg(&m.cf_offset)
            .arg(&m.cf_face)
            .arg(&m.cf_own)
            .arg(&start)
            .arg(&count_l)
            .launch(cfg_for(count))?;
    }
    Ok(())
}

// ==========================================================================
//  The solve
// ==========================================================================

/// The flat copy of `a.source` into `cw.b` - `k.copy` (`solCopy`) pushing the
/// `DevBuf<Vec3>`'s device pointer as the source of a `3 n_cells` flat copy.
/// [`solve_block_pbicgstab`] does this at the top of every solve.
pub(crate) fn copy_source_to_b(
    gpu: &Gpu,
    k: &SolverKernels,
    cw: &mut CoupledWorkspace,
    a: &GpuBlockLdu,
) -> Result<()> {
    let n = a.n_cells * 3;
    if n == 0 {
        return Ok(());
    }
    let nl = to_label(n)?;
    unsafe {
        gpu.stream()
            .launch_builder(&k.copy)
            .arg(&mut cw.b)
            .arg(&a.source)
            .arg(&nl)
            .launch(cfg_for(n))?;
    }
    Ok(())
}

/// (109.11) - `crate::solver::device_norm_factor` transcribed with
/// `n = 3 n_cells`: `x_ref` is the mean of ALL flat entries broadcast to
/// every entry, the block product of it and of `x` feed `norm_factor1`, and
/// `finish_sum(.., NORM_EPS)` lands the factor. Needs `cw.b` loaded; leaves
/// `A x` in `cw.w.apsi`.
pub fn block_norm_factor(
    gpu: &Gpu,
    k: &SolverKernels,
    ck: &CoupledKernels,
    cw: &mut CoupledWorkspace,
    a: &GpuBlockLdu,
    m: &GpuMesh,
) -> Result<()> {
    let n = a.n_cells * 3;
    if n == 0 {
        return gpu.fill_zero(&mut cw.w.norm_factor);
    }
    check_workspace(&cw.w, n)?;
    let nl = to_label(n)?;

    // x_ref = mean(x) over the FLAT vector. 1/(3n) is a property of the
    // system, not of the solution, so forming it on the host breaks no rule.
    device_sum(gpu, k, &mut cw.w.x_ref, &cw.x, &mut cw.w.partials, n)?;
    let inv_n = 1.0 / (n as Scalar);
    unsafe {
        gpu.stream()
            .launch_builder(&k.broadcast_scaled)
            .arg(&mut cw.w.tmp)
            .arg(&cw.w.x_ref)
            .arg(&inv_n)
            .arg(&nl)
            .launch(cfg_for(n))?;
    }

    block_amul(gpu, ck, &mut cw.w.y, &cw.w.tmp, a, m)?;
    block_amul(gpu, ck, &mut cw.w.apsi, &cw.x, a, m)?;

    let (cfg, nparts) = crate::solver::reduce_geometry(n);
    unsafe {
        gpu.stream()
            .launch_builder(&k.norm_factor1)
            .arg(&mut cw.w.partials)
            .arg(&cw.w.apsi)
            .arg(&cw.b)
            .arg(&cw.w.y)
            .arg(&nl)
            .launch(cfg)?;
    }
    finish_sum(gpu, k, &mut cw.w.norm_factor, &cw.w.partials, nparts, NORM_EPS)
}

/// §8.1's PBiCGStab over the flat `3 n_cells` system (109.8), the loop
/// `crate::solver::solve_pbicgstab` runs, line for line: the same calls in
/// the same order, `n = 3 n_cells`, `psi = cw.x`, `a.source` copied into
/// `cw.b` by the caller, `amul` the block product, the preconditioner
/// [`precondition`]. Builds its own preconditioner, solves `cw.x` in place,
/// allocates nothing. The residuals are (109.11)-normalised; the reported
/// final residual is the recomputed TRUE `b - A x`, as there.
pub fn solve_block_pbicgstab(
    gpu: &Gpu,
    k: &SolverKernels,
    ck: &CoupledKernels,
    cw: &mut CoupledWorkspace,
    a: &GpuBlockLdu,
    m: &GpuMesh,
    ctrl: &SolverControls,
    precon: BlockPrecon,
) -> Result<SolverPerformance> {
    refuse_gamg(ctrl)?;
    check_shapes(a, m, cw)?;
    let n = a.n_cells * 3;
    let mut perf = SolverPerformance {
        converged: true,
        ..Default::default()
    };
    if n == 0 {
        return Ok(perf);
    }
    perf.converged = false;
    refuse_round_trip_in_capture(gpu, ctrl, "solve_block_pbicgstab")?;

    check_workspace(&cw.w, n)?;
    if cw.x.len() < n {
        return Err(Error::Config(format!(
            "solve_block_pbicgstab: x holds {} values, the system has {n}",
            cw.x.len()
        )));
    }

    build_block_preconditioner(gpu, ck, cw, a, m, precon)?;
    copy_source_to_b(gpu, k, cw, a)?;

    // ---- r = b - A·x, and the (109.11) normalisation. block_norm_factor
    // leaves A·x in cw.w.apsi, so the residual is one subtraction.
    block_norm_factor(gpu, k, ck, cw, a, m)?;
    vec_sub(gpu, k, &mut cw.w.r, &cw.b, &cw.w.apsi, n)?;
    vec_copy(gpu, k, &mut cw.w.r0, &cw.w.r, n)?;

    device_sum_mag(gpu, k, &mut cw.w.initial_res, &cw.w.r, &mut cw.w.partials, n)?;
    // Report honestly if the loop never runs.
    copy_scalar(gpu, k, &mut cw.w.final_res, &cw.w.initial_res)?;

    gpu.fill_zero(&mut cw.w.p)?;
    gpu.fill_zero(&mut cw.w.v)?;
    gpu.fill_zero(&mut cw.w.flag)?;
    set_scalar(gpu, k, &mut cw.w.rho_old, 1.0)?;
    set_scalar(gpu, k, &mut cw.w.alpha, 1.0)?;
    set_scalar(gpu, k, &mut cw.w.omega, 1.0)?;

    let max_iter = ctrl.max_iter.max(0) as usize;
    let interval = ctrl.check_interval.max(1) as usize;
    let checking = !ctrl.fixed_iters;

    // An already-converged system must not be iterated.
    if checking {
        convergence_test(
            gpu, k, &mut cw.w.flag, &cw.w.initial_res, &cw.w.initial_res, &cw.w.norm_factor,
            ctrl, 0,
        )?;
        perf.converged = read_flag(gpu, &cw.w.flag, &mut cw.w.flag_host, &cw.w.flag_event)?;
    }

    if !perf.converged {
        for it in 0..max_iter {
            let iters = it + 1;

            // rho = (r0,r); beta = (rho/rho_old)·(alpha/omega); rho_old = rho.
            bicg_rho_and_beta(gpu, k, &mut cw.w, n)?;

            bicg_p_update(gpu, k, &mut cw.w, n)?;
            precondition(gpu, k, ck, &mut cw.w.p_hat, &cw.w.p, &cw.r_diag, &cw.colouring, a, m, precon)?;
            block_amul(gpu, ck, &mut cw.w.v, &cw.w.p_hat, a, m)?;

            // alpha = rho/(r0,v)
            dot_then_divide(
                gpu, k, &mut cw.w.den, &mut cw.w.alpha, &cw.w.rho, &cw.w.r0, &cw.w.v,
                &mut cw.w.partials, n,
            )?;

            bicg_s_update(gpu, k, &mut cw.w, n)?;
            precondition(gpu, k, ck, &mut cw.w.s_hat, &cw.w.s, &cw.r_diag, &cw.colouring, a, m, precon)?;
            block_amul(gpu, ck, &mut cw.w.t, &cw.w.s_hat, a, m)?;

            // (t,s) and (t,t) in one pass, then omega = (t,s)/(t,t).
            dot2_then_divide(
                gpu, k, &mut cw.w.num, &mut cw.w.den, &mut cw.w.omega, &cw.w.t, &cw.w.s,
                &mut cw.w.partials, &mut cw.w.partials_b, n,
            )?;

            bicg_x_update(gpu, k, &mut cw.x, &cw.w, n)?;
            bicg_r_update(gpu, k, &mut cw.w, n)?;

            perf.n_iterations = iters;

            if checking && iters % interval == 0 {
                let itl = to_label(iters)?;
                sum_mag_then_test(
                    gpu, k, &mut cw.w.final_res, &mut cw.w.flag, &cw.w.r, &cw.w.initial_res,
                    &cw.w.norm_factor, &mut cw.w.partials, ctrl, itl, n,
                )?;
                if read_flag(gpu, &cw.w.flag, &mut cw.w.flag_host, &cw.w.flag_event)? {
                    perf.converged = true;
                    break;
                }
            }
        }
    }

    // finish_solve's body, guarded by report_residuals as there: the TRUE
    // b - A x, and the fixed-iteration verdict read off it.
    if ctrl.report_residuals {
        block_amul(gpu, ck, &mut cw.w.apsi, &cw.x, a, m)?;
        vec_sub(gpu, k, &mut cw.w.tmp, &cw.b, &cw.w.apsi, n)?;
        device_sum_mag(gpu, k, &mut cw.w.final_res, &cw.w.tmp, &mut cw.w.partials, n)?;

        collect_report(gpu, k, &mut cw.w, &mut perf)?;

        // In fixed-iteration mode nothing sampled the device flag, but the
        // numbers are on the host now, so the same criterion applies here.
        if ctrl.fixed_iters {
            let abs = perf.final_residual <= ctrl.tolerance;
            let rel = ctrl.rel_tol > 0.0
                && perf.final_residual <= ctrl.rel_tol * perf.initial_residual;
            perf.converged = abs || rel;
        }
    }
    Ok(perf)
}

// ==========================================================================
//  The block map F(u) and the outer loop around it (SPEC-LIT §109.6)
// ==========================================================================

use crate::solid::block::BlockOperator;
use crate::solid::displacement::Displacement;
use crate::solid::motion_ratio_of;
use crate::solid::outer::{
    aitken_omega, anderson_gamma, gmres_fixed_point, observed_contraction, OuterControls,
    OuterReport, Relaxation,
};
use crate::Vec3;

/// Everything one block application needs, built once per region: the
/// operator (§109.1-§109.2), the flat workspace (§109.5), and both kernel
/// sets.
pub struct CoupledSolid {
    pub op: BlockOperator,
    pub cw: CoupledWorkspace,
    pub k: SolverKernels,
    pub ck: CoupledKernels,
}

impl CoupledSolid {
    /// A bonded region is refused by name, through [`BlockOperator::new`]
    /// (§95.8).
    pub fn new(gpu: &Gpu, d: &Displacement<'_>) -> Result<Self> {
        Ok(Self {
            op: BlockOperator::new(gpu, d)?,
            cw: CoupledWorkspace::for_mesh(gpu, d.mesh())?,
            k: SolverKernels::new(gpu)?,
            ck: CoupledKernels::new(gpu)?,
        })
    }
}

/// The flat copy INTO a `DevBuf<Scalar>` workspace vector from a
/// `DevBuf<Vec3>` cell field - `k.copy` (`solCopy`) pushing the typed
/// buffer's device pointer as the source, the launch shape of
/// [`copy_source_to_b`]. A typed buffer is only ever a launch argument,
/// never handed to a typed helper such as [`vec_copy`].
fn copy_vec3_to_flat(
    gpu: &Gpu,
    k: &SolverKernels,
    dst: &mut DevBuf<Scalar>,
    src: &DevBuf<Vec3>,
) -> Result<()> {
    // `n` is the FLAT count: the buffer holds one Vec3 per cell, the
    // kernel copies one Scalar per thread (109.8).
    let n = src.len() * 3;
    if n == 0 {
        return Ok(());
    }
    let nl = to_label(n)?;
    unsafe {
        gpu.stream()
            .launch_builder(&k.copy)
            .arg(dst)
            .arg(src)
            .arg(&nl)
            .launch(cfg_for(n))?;
    }
    Ok(())
}

/// The flat copy OUT of a `DevBuf<Scalar>` workspace vector into a
/// `DevBuf<Vec3>` cell field, the mirror of [`copy_vec3_to_flat`].
fn copy_flat_to_vec3(
    gpu: &Gpu,
    k: &SolverKernels,
    dst: &mut DevBuf<Vec3>,
    src: &DevBuf<Scalar>,
) -> Result<()> {
    // The mirror of [`copy_vec3_to_flat`]: the flat count, not the cell
    // count.
    let n = dst.len() * 3;
    if n == 0 {
        return Ok(());
    }
    let nl = to_label(n)?;
    unsafe {
        gpu.stream()
            .launch_builder(&k.copy)
            .arg(dst)
            .arg(src)
            .arg(&nl)
            .launch(cfg_for(n))?;
    }
    Ok(())
}

/// ONE application of the block map `F(u)` into `out` (§109.6's (109.12)):
/// the boundary values and gradients of the CURRENT `u` (`d.passes`
/// sub-passes), the block operator assembled at that state, the flat copy
/// of `d.u.f` into `cw.x` (the warm start), ONE [`solve_block_pbicgstab`]
/// with `d.ctrl` and [`BlockPrecon::Dilu`], the flat copy of `cw.x` into
/// `out`. `u` is not written; its fixed point is the discrete solution of
/// §109's operator.
pub fn apply_block(
    gpu: &Gpu,
    d: &mut Displacement<'_>,
    cs: &mut CoupledSolid,
    out: &mut DevBuf<Vec3>,
) -> Result<SolverPerformance> {
    if out.len() != d.mesh().n_cells {
        return Err(Error::Config(format!(
            "apply_block: out holds {} elements for {} cells - the output of \
             the block map is one Vec3 per cell",
            out.len(),
            d.mesh().n_cells
        )));
    }
    d.correct_boundary(gpu)?;
    cs.op.assemble(gpu, d)?;
    copy_vec3_to_flat(gpu, &cs.k, &mut cs.cw.x, &d.u.f)?;
    let perf = solve_block_pbicgstab(
        gpu,
        &cs.k,
        &cs.ck,
        &mut cs.cw,
        cs.op.matrix(),
        d.mesh(),
        &d.ctrl,
        BlockPrecon::Dilu,
    )?;
    copy_flat_to_vec3(gpu, &cs.k, out, &cs.cw.x)?;
    Ok(perf)
}

/// §95.3's outer loop around §109.6's block map:
/// [`crate::solid::outer::solve`] transcribed line for line,
/// [`apply_block`] in place of the three scalar solves - the same
/// relaxations, the same stopping quantity `r = F(u) - u`, the same
/// divergence refusal, the same report, and the same closing boundary
/// correction so the gradient a stress read-out takes belongs to the
/// accepted `u`. `predicted_contraction` is the SEGREGATED split's
/// prediction, `(mu + lambda)/(2 mu + lambda)`, kept so the two loops
/// print the same columns; the observed contraction of THIS loop is what
/// a slender body separates them on.
pub fn solve(gpu: &Gpu, d: &mut Displacement<'_>, ctrl: &OuterControls) -> Result<OuterReport> {
    if let Relaxation::Gmres(m) = ctrl.relaxation {
        return solve_gmres(gpu, d, ctrl, m);
    }
    d.passes = ctrl.boundary_passes;
    let n_c = d.mesh().n_cells;

    // Cell volumes, once: motion_ratio needs them; the mesh does not move.
    let vols: Vec<Scalar> = gpu.download(&d.mesh().v)?;
    let mut cs = CoupledSolid::new(gpu, d)?;
    let mut report = OuterReport {
        predicted_contraction: d.material.predicted_contraction(),
        ..Default::default()
    };
    let mut u: Vec<Vec3> = gpu.download(&d.u.f)?;
    let mut next = gpu.zeros(n_c)?;
    let mut r: Vec<Vec3> = vec![Vec3::ZERO; n_c];
    let mut r_prev: Vec<Vec3> = vec![Vec3::ZERO; n_c];
    let mut g_prev: Vec<Vec3> = vec![Vec3::ZERO; n_c];
    let mut d_f: Vec<Vec<Vec3>> = Vec::new();
    let mut d_g: Vec<Vec<Vec3>> = Vec::new();
    let depth = match ctrl.relaxation {
        Relaxation::Anderson(m) => m,
        _ => 0,
    };
    let mut omega = 1.0 as Scalar;
    let mut first = 0.0 as Scalar;
    let mut prev_norm = 0.0 as Scalar;

    for k in 1..=ctrl.max_outer {
        let perf = apply_block(gpu, d, &mut cs, &mut next)?;
        report.linear_iterations += perf.n_iterations;
        let f: Vec<Vec3> = gpu.download(&next)?;
        for c in 0..n_c {
            r[c] = f[c] - u[c];
        }
        let norm = l2(&r);
        report.norms.push(norm);
        report.iterations = k;
        if !norm.is_finite() || (k > 1 && norm > first * 1.0e6) {
            return Err(Error::Diverged {
                iteration: k,
                what: format!(
                    "the block-coupled displacement fixed point grew from {first:.3e} to \
                     {norm:.3e} at nu = {:.2} (a million-fold): the outer loop around \
                     SPEC-LIT §109.6's block map does not contract here; \
                     relaxation = {:?}, boundary_passes = {}",
                    d.material.nu, ctrl.relaxation, ctrl.boundary_passes
                ),
            });
        }

        if k == 1 {
            first = norm;
            omega = 1.0;
        } else {
            report.ratios.push(if prev_norm > 0.0 { norm / prev_norm } else { 0.0 });
            if ctrl.relaxation == Relaxation::Aitken {
                omega = aitken_omega(omega, &r_prev, &r);
            }
            if norm <= first * (10.0 as Scalar).powf(-ctrl.decades) {
                report.converged = true;
                for c in 0..n_c {
                    u[c] += r[c] * omega;
                }
                gpu.write(&mut d.u.f, &u)?;
                break;
            }
            if depth > 0 {
                // The two difference columns of this iteration, then the
                // window trimmed to `depth` - the prototype's order.
                let mut df = vec![Vec3::ZERO; n_c];
                let mut dg = vec![Vec3::ZERO; n_c];
                for c in 0..n_c {
                    df[c] = r[c] - r_prev[c];
                    dg[c] = f[c] - g_prev[c];
                }
                d_f.push(df);
                d_g.push(dg);
                while d_f.len() > depth {
                    d_f.remove(0);
                    d_g.remove(0);
                }
            }
        }
        if ctrl.relaxation == Relaxation::Aitken {
            report.omegas.push(omega);
        }
        if d_f.is_empty() {
            for c in 0..n_c {
                u[c] += r[c] * omega;
            }
        } else {
            let gamma = anderson_gamma(&d_f, &r);
            for c in 0..n_c {
                let mut v = f[c];
                for (j, col) in d_g.iter().enumerate() {
                    v -= col[c] * gamma[j];
                }
                u[c] = v;
            }
            // Not a relaxation factor: the one-norm of the least-squares
            // combination, which is what says whether the columns have gone
            // linearly dependent.
            report.omegas.push(gamma.iter().map(|x| x.abs()).sum());
        }
        gpu.write(&mut d.u.f, &u)?;
        r_prev.copy_from_slice(&r);
        g_prev.copy_from_slice(&f);
        prev_norm = norm;
    }

    // The boundary values and the gradient the caller will read stress from
    // have to belong to the displacement that was just accepted.
    d.correct_boundary(gpu)?;
    report.observed_contraction = observed_contraction(&report.ratios, 10);
    report.motion_ratio = motion_ratio_of(&u, &vols);
    println!(
        "solid coupled: nu={:.3} relaxation={:?} outer={} converged={} observed={:.4} \
         predicted={:.4} omega_last={:.3} linear_iters={} motion_ratio={:.3e}",
        d.material.nu,
        ctrl.relaxation,
        report.iterations,
        report.converged,
        report.observed_contraction,
        report.predicted_contraction,
        report.omegas.last().copied().unwrap_or(omega),
        report.linear_iterations,
        report.motion_ratio
    );
    Ok(report)
}

/// SPEC-LIT §142.2: restarted GMRES(`restart`) on the block map's fixed
/// point, over the loop's whole carried state `x = (u, ub)` - the cell
/// displacement and the boundary values one application starts from
/// (142.5). `Phi(x)` writes `x` into `d.u.f` and `d.u.bf`, runs one
/// [`apply_block`] and reads back `(F(u), ub')`, so `Phi` is (109.12) with
/// the boundary values it carries made explicit, and its fixed point is
/// §142.1's `u*`. The host side is [`crate::solid::outer`]'s
/// `gmres_fixed_point`; the report has the Anderson loop's fields, with
/// `omegas` left empty, and the same closing boundary correction.
fn solve_gmres(
    gpu: &Gpu,
    d: &mut Displacement<'_>,
    ctrl: &OuterControls,
    restart: usize,
) -> Result<OuterReport> {
    d.passes = ctrl.boundary_passes;
    let n_c = d.mesh().n_cells;
    let vols: Vec<Scalar> = gpu.download(&d.mesh().v)?;
    let mut cs = CoupledSolid::new(gpu, d)?;
    let mut report = OuterReport {
        predicted_contraction: d.material.predicted_contraction(),
        ..Default::default()
    };
    let mut x: Vec<Vec3> = gpu.download(&d.u.f)?;
    x.extend(gpu.download(&d.u.bf)?);
    let mut next = gpu.zeros(n_c)?;
    let nu = d.material.nu;
    let passes = ctrl.boundary_passes;
    let cycles = gmres_fixed_point(
        &mut x,
        restart,
        ctrl.decades,
        ctrl.max_outer,
        &mut report,
        |first, norm| {
            format!(
                "the block-coupled displacement fixed point grew from {first:.3e} to \
                 {norm:.3e} at nu = {nu:.2} (a million-fold, or not finite): GMRES({restart}) \
                 on SPEC-LIT §142.2's state does not converge here; boundary_passes = {passes}"
            )
        },
        |state: &[Vec3]| {
            gpu.write(&mut d.u.f, &state[..n_c])?;
            gpu.write(&mut d.u.bf, &state[n_c..])?;
            let perf = apply_block(gpu, d, &mut cs, &mut next)?;
            let mut out: Vec<Vec3> = gpu.download(&next)?;
            out.extend(gpu.download(&d.u.bf)?);
            Ok((out, perf.n_iterations))
        },
    )?;
    gpu.write(&mut d.u.f, &x[..n_c])?;
    gpu.write(&mut d.u.bf, &x[n_c..])?;

    // The boundary values and the gradient the caller will read stress from
    // have to belong to the displacement that was just accepted.
    d.correct_boundary(gpu)?;
    report.observed_contraction = observed_contraction(&report.ratios, 10);
    report.motion_ratio = motion_ratio_of(&x[..n_c], &vols);
    println!(
        "solid coupled: nu={:.3} relaxation={:?} outer={} cycles={} converged={} observed={:.4} \
         predicted={:.4} linear_iters={} motion_ratio={:.3e}",
        d.material.nu,
        ctrl.relaxation,
        report.iterations,
        cycles,
        report.converged,
        report.observed_contraction,
        report.predicted_contraction,
        report.linear_iterations,
        report.motion_ratio
    );
    Ok(report)
}

/// The Euclidean norm of a vector field, sequential over cells - the
/// prototype's own reduction order; [`crate::solid::outer`] keeps it
/// private, so it is copied, four lines.
fn l2(v: &[Vec3]) -> Scalar {
    v.iter().map(|a| a.mag_sqr()).sum::<Scalar>().sqrt()
}

// ==========================================================================
//  Tests. The host references walk the FACE LIST, not the kernels' cf_*
//  lists - the same choice Gate 109-B's scatter-shaped host twin makes.
// ==========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::case::{LinearSolverKind, Preconditioner};
    use crate::mesh::{GpuMesh, HostMesh};
    use crate::precon::{Adjacency, Colouring};
    use crate::reference;
    use crate::solid::bc;
    use crate::solid::block::{dense_from_block, flatten3, unflatten3, BlockOperator, HostBlockLdu};
    use crate::solid::displacement::Displacement;
    use crate::solid::{prototype, Material};
    use crate::Vec3;

    /// One `Scalar`'s bit pattern, the width the `to_bits` pair produces.
    #[cfg(not(feature = "single"))]
    type Bits = u64;
    #[cfg(feature = "single")]
    type Bits = u32;

    const DT: Scalar = 100.0;
    const T_REF: Scalar = 300.0;

    fn gpu() -> Option<Gpu> {
        Gpu::new(0).ok()
    }

    fn upload(gpu: &Gpu, m: &HostMesh) -> GpuMesh {
        GpuMesh::upload(gpu, m).expect("upload")
    }

    fn flat9(v: &[Tensor]) -> Vec<Scalar> {
        v.iter()
            .flat_map(|t| [t.xx, t.xy, t.xz, t.yx, t.yy, t.yz, t.zx, t.zy, t.zz])
            .collect()
    }

    /// `max|got - want| / max|want|` over the whole array.
    fn rel_max(got: &[Scalar], want: &[Scalar]) -> Scalar {
        assert_eq!(got.len(), want.len(), "rel_max: length mismatch");
        let scale = want.iter().fold(0.0 as Scalar, |a, &v| a.max(v.abs()));
        let d = got
            .iter()
            .zip(want)
            .fold(0.0 as Scalar, |a, (&g, &w)| a.max((g - w).abs()));
        d / scale.max(crate::SCALAR_FLOOR)
    }

    /// The controls R5 states: the block solve reads only the tolerance
    /// family, `check_interval`, `fixed_iters` and `report_residuals`.
    fn solid_controls() -> SolverControls {
        SolverControls {
            solver: LinearSolverKind::PBiCGStab,
            precon: Preconditioner::Dilu,
            tolerance: 1e-12,
            rel_tol: 0.0,
            max_iter: 2000,
            min_iter: 0,
            check_interval: 1,
            fixed_iters: false,
            report_residuals: true,
            gamg: crate::io::case::GamgControls::default(),
        }
    }

    /// R3/R5's system: one material, `T = T_REF + DT` everywhere, `u = 0`,
    /// the boundary closed and the block matrix assembled at that state.
    fn block_system(
        gpu: &Gpu,
        hm: &HostMesh,
        gm: &GpuMesh,
        per_patch: &[[bc::CompBc; 3]],
    ) -> Result<BlockOperator> {
        let mat = Material::steel(0.3);
        let n = hm.n_cells;
        let nbf = hm.n_boundary_faces;
        let mut d = Displacement::new(gpu, gm, hm, mat, per_patch, solid_controls())?;
        d.set_temperature(gpu, &vec![T_REF + DT; n], &vec![T_REF + DT; nbf], T_REF)?;
        d.set_displacement(gpu, &vec![Vec3::new(0.0, 0.0, 0.0); n], &vec![Vec3::new(0.0, 0.0, 0.0); nbf])?;
        d.correct_boundary(gpu)?;
        let mut op = BlockOperator::new(gpu, &d)?;
        op.assemble(gpu, &d)?;
        Ok(op)
    }

    /// `y_k = sin(k)`, `k in 0..3 n` - the vector the preconditioner acts on.
    fn sin_vector(n3: usize) -> Vec<Scalar> {
        (0..n3).map(|i| (i as Scalar).sin()).collect()
    }

    // ---- the host 3x3 arithmetic of (109.9)/(109.10), face-list order ----

    fn hmatvec(t: Tensor, v: Vec3) -> Vec3 {
        Vec3::new(
            t.xx * v.x + t.xy * v.y + t.xz * v.z,
            t.yx * v.x + t.yy * v.y + t.yz * v.z,
            t.zx * v.x + t.zy * v.y + t.zz * v.z,
        )
    }

    fn hsubv(a: Vec3, b: Vec3) -> Vec3 {
        Vec3::new(a.x - b.x, a.y - b.y, a.z - b.z)
    }

    fn hmatmul(a: Tensor, b: Tensor) -> Tensor {
        Tensor {
            xx: a.xx * b.xx + a.xy * b.yx + a.xz * b.zx,
            xy: a.xx * b.xy + a.xy * b.yy + a.xz * b.zy,
            xz: a.xx * b.xz + a.xy * b.yz + a.xz * b.zz,
            yx: a.yx * b.xx + a.yy * b.yx + a.yz * b.zx,
            yy: a.yx * b.xy + a.yy * b.yy + a.yz * b.zy,
            yz: a.yx * b.xz + a.yy * b.yz + a.yz * b.zz,
            zx: a.zx * b.xx + a.zy * b.yx + a.zz * b.zx,
            zy: a.zx * b.xy + a.zy * b.yy + a.zz * b.zy,
            zz: a.zx * b.xz + a.zy * b.yz + a.zz * b.zz,
        }
    }

    fn hsubt(a: Tensor, b: Tensor) -> Tensor {
        Tensor {
            xx: a.xx - b.xx, xy: a.xy - b.xy, xz: a.xz - b.xz,
            yx: a.yx - b.yx, yy: a.yy - b.yy, yz: a.yz - b.yz,
            zx: a.zx - b.zx, zy: a.zy - b.zy, zz: a.zz - b.zz,
        }
    }

    fn identity3() -> Tensor {
        Tensor { xx: 1.0, xy: 0.0, xz: 0.0, yx: 0.0, yy: 1.0, yz: 0.0, zx: 0.0, zy: 0.0, zz: 1.0 }
    }

    fn hdet3(t: Tensor) -> Scalar {
        t.xx * (t.yy * t.zz - t.yz * t.zy) - t.xy * (t.yx * t.zz - t.yz * t.zx)
            + t.xz * (t.yx * t.zy - t.yy * t.zx)
    }

    /// `adj(t)/det`: entry `(i,j)` is the cofactor of `(j,i)`.
    fn hadj_over(t: Tensor, det: Scalar) -> Tensor {
        let d = 1.0 / det;
        Tensor {
            xx: (t.yy * t.zz - t.yz * t.zy) * d,
            xy: -(t.xy * t.zz - t.xz * t.zy) * d,
            xz: (t.xy * t.yz - t.xz * t.yy) * d,
            yx: -(t.yx * t.zz - t.yz * t.zx) * d,
            yy: (t.xx * t.zz - t.xz * t.zx) * d,
            yz: -(t.xx * t.yz - t.xz * t.yx) * d,
            zx: (t.yx * t.zy - t.yy * t.zx) * d,
            zy: -(t.xx * t.zy - t.xy * t.zx) * d,
            zz: (t.xx * t.yy - t.xy * t.yx) * d,
        }
    }

    /// `solidBlockInverse3`'s three-way rule on the host.
    fn hinv3(t: Tensor, fallback: Tensor) -> Tensor {
        let det = hdet3(t);
        if det != 0.0 {
            return hadj_over(t, det);
        }
        let fdet = hdet3(fallback);
        if fdet != 0.0 {
            return hadj_over(fallback, fdet);
        }
        identity3()
    }

    /// (109.9)'s factorisation on the host, colour by colour, walking the
    /// FACE list `0..n_internal_faces` - not the kernels' `cf_*` lists.
    fn host_dilu_factor(col: &Colouring, a: &HostBlockLdu, m: &HostMesh) -> Vec<Tensor> {
        let mut rd = vec![Tensor::ZERO; a.n_cells];
        for c in 0..col.n_colours {
            for slot in col.offsets[c]..col.offsets[c + 1] {
                let v = col.cells[slot] as usize;
                let mut dt = a.diag[v];
                for f in 0..m.n_internal_faces {
                    let (nbr, avu, auv) = if m.owner[f] as usize == v {
                        (m.neighbour[f] as usize, a.upper[f], a.lower[f])
                    } else if m.neighbour[f] as usize == v {
                        (m.owner[f] as usize, a.lower[f], a.upper[f])
                    } else {
                        continue;
                    };
                    if col.colour[nbr] < col.colour[v] {
                        dt = hsubt(dt, hmatmul(avu, hmatmul(rd[nbr], auv)));
                    }
                }
                rd[v] = hinv3(dt, a.diag[v]);
            }
        }
        rd
    }

    /// (109.9)'s forward/backward sweeps on the host, the same colour order.
    fn host_dilu_apply(
        col: &Colouring,
        rd: &[Tensor],
        a: &HostBlockLdu,
        m: &HostMesh,
        x: &[Scalar],
    ) -> Vec<Scalar> {
        let mut y: Vec<Vec3> = unflatten3(x);
        for c in 0..col.n_colours {
            for slot in col.offsets[c]..col.offsets[c + 1] {
                let v = col.cells[slot] as usize;
                let mut acc = Vec3::new(0.0, 0.0, 0.0);
                for f in 0..m.n_internal_faces {
                    let (nbr, avu) = if m.owner[f] as usize == v {
                        (m.neighbour[f] as usize, a.upper[f])
                    } else if m.neighbour[f] as usize == v {
                        (m.owner[f] as usize, a.lower[f])
                    } else {
                        continue;
                    };
                    if col.colour[nbr] < col.colour[v] {
                        let q = hmatvec(avu, y[nbr]);
                        acc = Vec3::new(acc.x + q.x, acc.y + q.y, acc.z + q.z);
                    }
                }
                y[v] = hmatvec(rd[v], hsubv(y[v], acc));
            }
        }
        for c in (0..col.n_colours).rev() {
            for slot in col.offsets[c]..col.offsets[c + 1] {
                let v = col.cells[slot] as usize;
                let mut acc = Vec3::new(0.0, 0.0, 0.0);
                for f in 0..m.n_internal_faces {
                    let (nbr, avu) = if m.owner[f] as usize == v {
                        (m.neighbour[f] as usize, a.upper[f])
                    } else if m.neighbour[f] as usize == v {
                        (m.owner[f] as usize, a.lower[f])
                    } else {
                        continue;
                    };
                    if col.colour[nbr] > col.colour[v] {
                        let q = hmatvec(avu, y[nbr]);
                        acc = Vec3::new(acc.x + q.x, acc.y + q.y, acc.z + q.z);
                    }
                }
                let q = hmatvec(rd[v], acc);
                y[v] = Vec3::new(y[v].x - q.x, y[v].y - q.y, y[v].z - q.z);
            }
        }
        flatten3(&y)
    }

    /// (109.10) on the host: the inverse of every `diag[c]`.
    fn host_jacobi(a: &HostBlockLdu) -> Vec<Tensor> {
        (0..a.n_cells)
            .map(|c| hinv3(a.diag[c], identity3()))
            .collect()
    }

    fn host_jacobi_apply(rd: &[Tensor], x: &[Scalar]) -> Vec<Scalar> {
        let y: Vec<Vec3> = unflatten3(x)
            .iter()
            .zip(rd)
            .map(|(&v, &t)| hmatvec(t, v))
            .collect();
        flatten3(&y)
    }

    /// The device half of one comparison: the colouring uploaded, the
    /// preconditioner built, `M^-1 sin` and `rD` downloaded.
    fn device_factor_and_apply(
        gpu: &Gpu,
        k: &SolverKernels,
        ck: &CoupledKernels,
        hm: &HostMesh,
        gm: &GpuMesh,
        a: &GpuBlockLdu,
        col: &Colouring,
        precon: BlockPrecon,
    ) -> Result<(Vec<Scalar>, Vec<Scalar>)> {
        let n = a.n_cells;
        let n3 = n * 3;
        let mc = MultiColour::from_colouring(gpu, &Adjacency::of(hm), col)?;
        let mut cw = CoupledWorkspace::with_colouring(gpu, n, mc)?;
        build_block_preconditioner(gpu, ck, &mut cw, a, gm, precon)?;
        let x = gpu.upload(&sin_vector(n3))?;
        let mut y = gpu.zeros::<Scalar>(n3)?;
        precondition(gpu, k, ck, &mut y, &x, &cw.r_diag, &cw.colouring, a, gm, precon)?;
        let rd = gpu.download(&cw.r_diag)?;
        Ok((flat9(&rd), gpu.download(&y)?))
    }

    /// The three meshes R3 names, in order.
    fn the_three_meshes() -> Vec<(&'static str, HostMesh)> {
        vec![
            ("uniform block(6)", prototype::block(6).expect("block")),
            ("jittered block(6, 0.25)", prototype::jittered_block(6, 0.25).expect("jittered")),
            ("graded block", crate::solid::tests::graded_block()),
        ]
    }

    /// R3's DILU half: the device factorisation+sweep against the host
    /// face-list twin, to 1e-12, on all three meshes.
    #[test]
    fn block_dilu_matches_the_host_factorisation_on_three_meshes() {
        let Some(gpu) = gpu() else { return };
        let k = SolverKernels::new(&gpu).expect("solver kernels");
        let ck = CoupledKernels::new(&gpu).expect("coupled kernels");
        for (name, hm) in the_three_meshes() {
            let gm = upload(&gpu, &hm);
            let op = block_system(&gpu, &hm, &gm, &bc::fixed_minus_x()).expect("system");
            let a_host = HostBlockLdu::download(&gpu, op.matrix()).expect("download");
            let col = Colouring::greedy(&Adjacency::of(&hm));

            let (rd_dev, y_dev) = device_factor_and_apply(
                &gpu, &k, &ck, &hm, &gm, op.matrix(), &col, BlockPrecon::Dilu,
            )
            .expect("device factorise+apply");
            let rd_host = host_dilu_factor(&col, &a_host, &hm);
            let y_host = host_dilu_apply(&col, &rd_host, &a_host, &hm, &sin_vector(hm.n_cells * 3));
            let e = rel_max(&y_dev, &y_host);
            println!("  block-DILU {name}: rel_max(y) = {e:.3e} ({:.2e} on rD)",
                rel_max(&rd_dev, &flat9(&rd_host)));
            assert!((e as f64) <= crate::precision::roundoff_tol(1e-12), "block-DILU {name}: rel_max = {e}");
        }
    }

    /// R3's Jacobi half: the device `3x3` diagonal inverse against the host's.
    #[test]
    fn block_jacobi_matches_the_host_inverse_on_three_meshes() {
        let Some(gpu) = gpu() else { return };
        let k = SolverKernels::new(&gpu).expect("solver kernels");
        let ck = CoupledKernels::new(&gpu).expect("coupled kernels");
        for (name, hm) in the_three_meshes() {
            let gm = upload(&gpu, &hm);
            let op = block_system(&gpu, &hm, &gm, &bc::fixed_minus_x()).expect("system");
            let a_host = HostBlockLdu::download(&gpu, op.matrix()).expect("download");
            let col = Colouring::greedy(&Adjacency::of(&hm));

            let (rd_dev, y_dev) = device_factor_and_apply(
                &gpu, &k, &ck, &hm, &gm, op.matrix(), &col, BlockPrecon::Jacobi,
            )
            .expect("device factorise+apply");
            let rd_host = host_jacobi(&a_host);
            let y_host = host_jacobi_apply(&rd_host, &sin_vector(hm.n_cells * 3));
            let e = rel_max(&y_dev, &y_host);
            println!("  block-Jacobi {name}: rel_max(y) = {e:.3e} ({:.2e} on rD)",
                rel_max(&rd_dev, &flat9(&rd_host)));
            assert!((e as f64) <= crate::precision::roundoff_tol(1e-12), "block-Jacobi {name}: rel_max = {e}");
        }
    }

    /// `solidBlockInverse3` is an inverse, checked by multiplication rather
    /// than against a second transcription of the same formula: (a) block-
    /// Jacobi's `rD` times a full, non-symmetric `diag` block is the identity;
    /// (b) a factorisation whose `Dt` is exactly singular falls back to the
    /// INVERSE of the cell's own `diag` (109.9's safe reciprocal), not to the
    /// block divided by its determinant. `block(2)` is two-coloured and every
    /// colour-1 cell has three colour-0 neighbours, so `diag = I` on colour 0,
    /// `3 I` on colour 1 and `upper = lower = I` make `Dt = 3I - 3I = 0`
    /// exactly and the right answer `I/3`. The bound is loose enough for f32.
    #[test]
    fn the_block_inverse_inverts_a_full_block_and_falls_back_to_the_diagonal() {
        let Some(gpu) = gpu() else { return };
        let k = SolverKernels::new(&gpu).expect("solver kernels");
        let ck = CoupledKernels::new(&gpu).expect("coupled kernels");
        let hm = prototype::block(2).expect("block 2");
        let gm = upload(&gpu, &hm);
        let col = Colouring::greedy(&Adjacency::of(&hm));
        assert_eq!(col.n_colours, 2, "block(2) must be two-coloured");
        let n = hm.n_cells;
        let nif = hm.n_internal_faces;
        let eye = identity3();
        let check = |what: &str, got: Tensor, want: Tensor| {
            let e = rel_max(&flat9(&[got]), &flat9(&[want]));
            assert!(e <= 1e-5, "{what}: rel_max = {e}, got {got:?}");
        };

        // (a) a full block with every off-diagonal entry distinct.
        let full = |c: usize| {
            let s = c as Scalar * 0.01;
            Tensor {
                xx: 4.0 + s, xy: 1.0, xz: 2.0,
                yx: 0.5, yy: 5.0, yz: 1.0 + s,
                zx: 1.5, zy: 0.25, zz: 6.0,
            }
        };
        let ha = HostBlockLdu {
            n_cells: n,
            n_internal_faces: nif,
            diag: (0..n).map(full).collect(),
            upper: vec![Tensor::ZERO; nif],
            lower: vec![Tensor::ZERO; nif],
            source: vec![Vec3::ZERO; n],
        };
        let a = ha.upload(&gpu).expect("upload a");
        let (rd, _) = device_factor_and_apply(&gpu, &k, &ck, &hm, &gm, &a, &col, BlockPrecon::Jacobi)
            .expect("jacobi");
        for c in 0..n {
            let r = Tensor {
                xx: rd[9 * c], xy: rd[9 * c + 1], xz: rd[9 * c + 2],
                yx: rd[9 * c + 3], yy: rd[9 * c + 4], yz: rd[9 * c + 5],
                zx: rd[9 * c + 6], zy: rd[9 * c + 7], zz: rd[9 * c + 8],
            };
            check("rD . diag", hmatmul(r, full(c)), eye);
        }

        // (b) the singular-Dt fallback.
        let three = Tensor { xx: 3.0, yy: 3.0, zz: 3.0, ..Tensor::ZERO };
        let third = Tensor { xx: 1.0 / 3.0, yy: 1.0 / 3.0, zz: 1.0 / 3.0, ..Tensor::ZERO };
        let hb = HostBlockLdu {
            n_cells: n,
            n_internal_faces: nif,
            diag: (0..n).map(|c| if col.colour[c] == 0 { eye } else { three }).collect(),
            upper: vec![eye; nif],
            lower: vec![eye; nif],
            source: vec![Vec3::ZERO; n],
        };
        let b = hb.upload(&gpu).expect("upload b");
        let (rd, _) = device_factor_and_apply(&gpu, &k, &ck, &hm, &gm, &b, &col, BlockPrecon::Dilu)
            .expect("dilu");
        for c in 0..n {
            let r = Tensor {
                xx: rd[9 * c], xy: rd[9 * c + 1], xz: rd[9 * c + 2],
                yx: rd[9 * c + 3], yy: rd[9 * c + 4], yz: rd[9 * c + 5],
                zx: rd[9 * c + 6], zy: rd[9 * c + 7], zz: rd[9 * c + 8],
            };
            let want = if col.colour[c] == 0 { eye } else { third };
            check("singular-Dt fallback", r, want);
        }
    }

    /// R4: the factorisation is schedule-independent TO THE BIT - a shuffled
    /// colouring gives the same `rD` and the same `M^-1 y`; a reversed colour
    /// order is a different, equally lawful factorisation, so only the
    /// (109.11) outcome is compared and both iteration counts are printed.
    #[test]
    fn the_block_factorisation_is_schedule_independent() {
        let Some(gpu) = gpu() else { return };
        let k = SolverKernels::new(&gpu).expect("solver kernels");
        let ck = CoupledKernels::new(&gpu).expect("coupled kernels");

        let hm = prototype::jittered_block(6, 0.25).expect("jittered");
        let gm = upload(&gpu, &hm);
        let op = block_system(&gpu, &hm, &gm, &bc::fixed_minus_x()).expect("system");
        let col = Colouring::greedy(&Adjacency::of(&hm));
        let shuffled = col.with_shuffled_cells_within_colours(7);

        let (rd_a, y_a) = device_factor_and_apply(
            &gpu, &k, &ck, &hm, &gm, op.matrix(), &col, BlockPrecon::Dilu,
        ).expect("unshuffled");
        let (rd_b, y_b) = device_factor_and_apply(
            &gpu, &k, &ck, &hm, &gm, op.matrix(), &shuffled, BlockPrecon::Dilu,
        ).expect("shuffled");

        let rd_a_bits: Vec<Bits> = rd_a.iter().map(|&s| s.to_bits()).collect();
        let rd_b_bits: Vec<Bits> = rd_b.iter().map(|&s| s.to_bits()).collect();
        assert_eq!(rd_a_bits, rd_b_bits, "shuffled rD differs from the unshuffled one");
        let y_a_bits: Vec<Bits> = y_a.iter().map(|&s| s.to_bits()).collect();
        let y_b_bits: Vec<Bits> = y_b.iter().map(|&s| s.to_bits()).collect();
        assert_eq!(y_a_bits, y_b_bits, "shuffled M^-1 y differs from the unshuffled one");

        // The reversed colour order on R5's system: same tolerance, both
        // iteration counts printed. A reversed colouring is still a valid
        // colouring, so the factorisation is lawful - and different.
        let hm5 = prototype::block(6).expect("block");
        let gm5 = upload(&gpu, &hm5);
        let op5 = block_system(&gpu, &hm5, &gm5, &bc::free_expansion()).expect("system 5");
        let col5 = Colouring::greedy(&Adjacency::of(&hm5));
        let rev5 = col5.with_reversed_colour_order();
        let mut ctrl = solid_controls();
        ctrl.tolerance = 1e-5;
        let iters = |name: &str, c: &Colouring| -> usize {
            let mc = MultiColour::from_colouring(&gpu, &Adjacency::of(&hm5), c).expect("mc");
            let mut cw = CoupledWorkspace::with_colouring(&gpu, hm5.n_cells, mc).expect("cw");
            let perf = solve_block_pbicgstab(
                &gpu, &k, &ck, &mut cw, op5.matrix(), &gm5, &ctrl, BlockPrecon::Dilu,
            ).expect("solve");
            println!("  schedule independence: {name} colour order: {} iterations, converged = {}",
                perf.n_iterations, perf.converged);
            assert!(perf.converged, "{name} colour order did not converge at 1e-5");
            perf.n_iterations
        };
        let plain = iters("ascending", &col5);
        let reversed = iters("reversed", &rev5);
        println!("  schedule independence: ascending = {plain}, reversed = {reversed}");
    }

    /// R5: on `block(6)` with free expansion, the block solve's answer is
    /// the dense direct solve's to 1e-10, and block-DILU converges in FEWER
    /// iterations than block-Jacobi at the same tolerance.
    #[test]
    fn the_block_solve_matches_the_dense_direct_solve_and_dilu_beats_jacobi() {
        let Some(gpu) = gpu() else { return };
        let k = SolverKernels::new(&gpu).expect("solver kernels");
        let ck = CoupledKernels::new(&gpu).expect("coupled kernels");
        let hm = prototype::block(6).expect("block");
        let gm = upload(&gpu, &hm);
        assert_eq!(hm.n_cells * 3, 648, "block(6) is 216 cells, 648 unknowns");
        let op = block_system(&gpu, &hm, &gm, &bc::free_expansion()).expect("system");
        let a_host = HostBlockLdu::download(&gpu, op.matrix()).expect("download");

        let solve_with = |precon: BlockPrecon| -> Result<(Vec<Scalar>, SolverPerformance)> {
            let mut cw = CoupledWorkspace::for_mesh(&gpu, &gm)?;
            let perf = solve_block_pbicgstab(
                &gpu, &k, &ck, &mut cw, op.matrix(), &gm, &solid_controls(), precon,
            )?;
            Ok((gpu.download(&cw.x)?, perf))
        };

        let (x_dilu, perf_dilu) = solve_with(BlockPrecon::Dilu).expect("dilu solve");
        let (x_jacobi, perf_jacobi) = solve_with(BlockPrecon::Jacobi).expect("jacobi solve");
        assert!(perf_dilu.converged, "block-DILU did not converge");
        assert!(perf_jacobi.converged, "block-Jacobi did not converge");

        // The direct solve of the dense 3n x 3n expansion.
        let dense = dense_from_block(&a_host, &hm);
        let b_flat = flatten3(&a_host.source);
        let x_dense = reference::solve_dense(dense, &b_flat).expect("dense direct solve");

        let e_dilu = rel_max(&x_dilu, &x_dense);
        let e_jacobi = rel_max(&x_jacobi, &x_dense);
        println!(
            "  block solve on block(6): block-DILU {} iterations, block-Jacobi {} \
             iterations; rel_max vs the dense direct solve: DILU {e_dilu:.3e}, Jacobi {e_jacobi:.3e}",
            perf_dilu.n_iterations, perf_jacobi.n_iterations
        );
        assert!((e_dilu as f64) <= crate::precision::roundoff_tol(1e-10), "block-DILU vs dense: {e_dilu}");
        assert!((e_jacobi as f64) <= crate::precision::roundoff_tol(1e-10), "block-Jacobi vs dense: {e_jacobi}");
        assert!(
            perf_jacobi.n_iterations > perf_dilu.n_iterations,
            "block-Jacobi ({}) did not need more iterations than block-DILU ({})",
            perf_jacobi.n_iterations, perf_dilu.n_iterations
        );
    }

    /// R6: the refusals, by name. `PCG` and `ctrl.precon` are not read and
    /// not refused; a wrong-sized workspace and a wrong-sized mesh are.
    #[test]
    fn the_block_solve_refuses_a_wrong_workspace_and_gamg_by_name() {
        let Some(gpu) = gpu() else { return };
        let k = SolverKernels::new(&gpu).expect("solver kernels");
        let ck = CoupledKernels::new(&gpu).expect("coupled kernels");
        let hm = prototype::block(6).expect("block");
        let gm = upload(&gpu, &hm);
        let op = block_system(&gpu, &hm, &gm, &bc::free_expansion()).expect("system");
        let mut ctrl = solid_controls();

        // A workspace sized for another mesh is refused by name.
        let hm4 = prototype::block(4).expect("block 4");
        let gm4 = upload(&gpu, &hm4);
        let mut cw4 = CoupledWorkspace::for_mesh(&gpu, &gm4).expect("workspace 4");
        let err = solve_block_pbicgstab(
            &gpu, &k, &ck, &mut cw4, op.matrix(), &gm, &ctrl, BlockPrecon::Dilu,
        )
        .expect_err("a workspace for block(4) must be refused");
        assert!(
            err.to_string().contains("workspace"),
            "wrong workspace: {err}"
        );

        // A mesh that is not the matrix's is refused by name.
        let mut cw = CoupledWorkspace::for_mesh(&gpu, &gm).expect("workspace");
        let err = solve_block_pbicgstab(
            &gpu, &k, &ck, &mut cw, op.matrix(), &gm4, &ctrl, BlockPrecon::Dilu,
        )
        .expect_err("a mesh mismatch must be refused");
        assert!(err.to_string().contains("mesh"), "mesh mismatch: {err}");

        // GAMG is refused by name: the block solve is PBiCGStab, GAMG is the
        // pressure backend.
        ctrl.solver = LinearSolverKind::Gamg;
        let err = solve_block_pbicgstab(
            &gpu, &k, &ck, &mut cw, op.matrix(), &gm, &ctrl, BlockPrecon::Dilu,
        )
        .expect_err("GAMG must be refused");
        let msg = err.to_string();
        assert!(
            msg.contains("PBiCGStab") && msg.contains("pressure backend"),
            "GAMG refusal: {err}"
        );

        // PCG in `ctrl.solver` is NOT refused - and not read; the solve runs
        // PBiCGStab with the BlockPrecon it is given.
        let mut ctrl_pcg = solid_controls();
        ctrl_pcg.solver = LinearSolverKind::PCG;
        ctrl_pcg.precon = Preconditioner::Dic;
        let mut cw_pcg = CoupledWorkspace::for_mesh(&gpu, &gm).expect("workspace pcg");
        let perf = solve_block_pbicgstab(
            &gpu, &k, &ck, &mut cw_pcg, op.matrix(), &gm, &ctrl_pcg, BlockPrecon::Dilu,
        )
        .expect("PCG named in numerics must not be refused");
        assert!(perf.converged);
    }

    /// A fixed-iteration solve reports its six iterations and a final
    /// residual that IS the recomputed true (109.11) - compared against a
    /// HOST recomputation from the downloaded `x`, matrix and source. Six
    /// sweeps leave the residual far above round-off (the checked solve needs
    /// eighteen to reach 1e-12), so the relative comparison measures the
    /// report, not two summation orders of a round-off floor.
    #[test]
    // f32 (F32-16): residual report off 1.4e-5 rel in f32
    #[cfg_attr(feature = "single", ignore = "fails at f32: SPEC-LIT 112.3")]
    fn a_fixed_iteration_block_solve_reports_the_recomputed_true_residual() {
        let Some(gpu) = gpu() else { return };
        let k = SolverKernels::new(&gpu).expect("solver kernels");
        let ck = CoupledKernels::new(&gpu).expect("coupled kernels");
        let hm = prototype::block(6).expect("block");
        let gm = upload(&gpu, &hm);
        let op = block_system(&gpu, &hm, &gm, &bc::free_expansion()).expect("system");
        let a_host = HostBlockLdu::download(&gpu, op.matrix()).expect("download");

        let mut ctrl = solid_controls();
        ctrl.fixed_iters = true;
        ctrl.max_iter = 6;
        ctrl.report_residuals = true;

        let mut cw = CoupledWorkspace::for_mesh(&gpu, &gm).expect("workspace");
        let perf = solve_block_pbicgstab(
            &gpu, &k, &ck, &mut cw, op.matrix(), &gm, &ctrl, BlockPrecon::Dilu,
        )
        .expect("solve");
        assert_eq!(perf.n_iterations, 6, "fixed_iters must run exactly max_iter sweeps");
        println!(
            "  fixed-iteration report: initial = {:.6e}, final = {:.6e}",
            perf.initial_residual, perf.final_residual
        );

        // The host recomputation of (109.11). Its NORMALISATION is formed
        // from the state the solve started from (`x0 = 0`, as §8.4 and
        // solve_pbicgstab form it once, before the loop); its numerator from
        // the downloaded final `x`.
        let x = gpu.download(&cw.x).expect("x");
        let x0 = vec![0.0 as Scalar; x.len()];
        let dense = dense_from_block(&a_host, &hm);
        let mv = |v: &[Scalar]| -> Vec<Scalar> {
            (0..v.len())
                .map(|r| {
                    dense[r * v.len()..r * v.len() + v.len()]
                        .iter()
                        .zip(v)
                        .map(|(&a, &b)| a * b)
                        .sum()
                })
                .collect()
        };
        let b_flat = flatten3(&a_host.source);
        let ax = mv(&x);
        let ax0 = mv(&x0);
        let n3 = x0.len();
        let mean: Scalar = x0.iter().sum::<Scalar>() / (n3 as Scalar);
        let x_ref = vec![mean; n3];
        let ax_ref = mv(&x_ref);
        let norm = ax0.iter().zip(&ax_ref).map(|(&a, &b)| (a - b).abs()).sum::<Scalar>()
            + b_flat.iter().zip(&ax_ref).map(|(&a, &b)| (a - b).abs()).sum::<Scalar>()
            + NORM_EPS;
        let num: Scalar = b_flat.iter().zip(&ax).map(|(&a, &b)| (a - b).abs()).sum();
        let res = num / norm;

        println!("  fixed-iteration report: host (109.11) = {res:.6e}");
        assert!(res > 1e-8, "six sweeps must leave the residual above round-off: {res}");
        let e = (perf.final_residual - res).abs() / res;
        assert!(e <= 1e-10, "reported final residual off the host truth: relative {e}");
        // finish_solve's fixed_iters rule, read off the report: 1e-12 is not
        // reached in six sweeps, so the verdict is `false`.
        assert!(!perf.converged, "fixed_iters verdict must read off the report");
    }

    /// R8: the fixed-iteration block solve captures and replays bitwise
    /// under SPEC-LIT §81's protocol. The preconditioner build and the
    /// assembly are setup; the iterate is a zeroed `x` and ONE solve.
    #[test]
    fn the_block_coupled_solve_replays_bitwise() {
        let Some(gpu) = gpu() else { return };
        let k = SolverKernels::new(&gpu).expect("solver kernels");
        let ck = CoupledKernels::new(&gpu).expect("coupled kernels");
        let mat = Material::steel(0.3);
        let hm = prototype::block(6).expect("block");
        let gm = upload(&gpu, &hm);
        let n = hm.n_cells;
        let nbf = hm.n_boundary_faces;

        let mut ctrl = solid_controls();
        ctrl.fixed_iters = true;
        ctrl.max_iter = 4;
        ctrl.report_residuals = false;

        let report = crate::capture::capture_replays_bitwise(
            &gpu,
            "solid block-coupled solve, one fixed-iteration solve",
            || {
                let mut d = Displacement::new(&gpu, &gm, &hm, mat, &bc::fixed_minus_x(), ctrl)?;
                d.set_temperature(&gpu, &vec![T_REF + DT; n], &vec![T_REF + DT; nbf], T_REF)?;
                d.correct_boundary(&gpu)?;
                let mut blk = BlockOperator::new(&gpu, &d)?;
                blk.assemble(&gpu, &d)?;
                let mut cw = CoupledWorkspace::for_mesh(&gpu, &gm)?;
                build_block_preconditioner(&gpu, &ck, &mut cw, blk.matrix(), &gm, BlockPrecon::Dilu)?;
                Ok((blk, cw))
            },
            |(blk, cw)| {
                gpu.fill_zero(&mut cw.x)?;
                solve_block_pbicgstab(
                    &gpu, &k, &ck, cw, blk.matrix(), &gm, &ctrl, BlockPrecon::Dilu,
                )
                .map(|_| ())
            },
            |(_, cw)| {
                Ok(vec![
                    ("x", gpu.download(&cw.x)?),
                    ("r", gpu.download(&cw.w.r)?),
                    ("p", gpu.download(&cw.w.p)?),
                    ("p_hat", gpu.download(&cw.w.p_hat)?),
                    ("s_hat", gpu.download(&cw.w.s_hat)?),
                    ("rho", gpu.download(&cw.w.rho)?),
                    ("alpha", gpu.download(&cw.w.alpha)?),
                    ("omega", gpu.download(&cw.w.omega)?),
                    ("r_diag", flat9(&gpu.download(&cw.r_diag)?)),
                ])
            },
        )
        .expect("SPEC-LIT 81.7: the block-coupled solve must capture and replay bitwise");
        println!("  block-coupled solve: {report}");
    }

    use crate::solid::fixtures;
    use crate::solid::outer::{ANDERSON_DEPTH, GMRES_RESTART, OuterControls, Relaxation};
    use crate::solid::tests::{graded_block, linear_state};
    // VAL-10's probe: the stress read-out and the residual scale it judges
    // the device residual against. Only the ignored probe uses them, so
    // they follow it out of the single build.
    #[cfg(not(feature = "single"))]
    use crate::solid::block::residual_scale;
    #[cfg(not(feature = "single"))]
    use crate::solid::stress::StressFields;

    /// `Vec3` field -> the flat `3 n` layout `rel_max` reads.
    fn flat3(v: &[Vec3]) -> Vec<Scalar> {
        v.iter().flat_map(|a| [a.x, a.y, a.z]).collect()
    }

    /// The block map's linear solve in the fixed-point tests below.
    fn block_tight() -> SolverControls {
        SolverControls {
            solver: LinearSolverKind::PBiCGStab,
            precon: Preconditioner::Dilu,
            tolerance: 1e-12,
            rel_tol: 0.0,
            max_iter: 5000,
            min_iter: 0,
            check_interval: 1,
            fixed_iters: false,
            report_residuals: true,
            gamg: crate::io::case::GamgControls::default(),
        }
    }

    /// A free-expansion displacement (`T = T_REF + DT` everywhere) with the
    /// given state uploaded, ready for `apply_block` or either loop.
    fn free_expansion_state<'m>(
        gpu: &Gpu,
        hm: &'m HostMesh,
        gm: &'m GpuMesh,
        u: &[Vec3],
        ub: &[Vec3],
        ctrl: SolverControls,
    ) -> Result<Displacement<'m>> {
        let n = hm.n_cells;
        let nbf = hm.n_boundary_faces;
        let mut d =
            Displacement::new(gpu, gm, hm, Material::steel(0.3), &bc::free_expansion(), ctrl)?;
        d.set_temperature(gpu, &vec![T_REF + DT; n], &vec![T_REF + DT; nbf], T_REF)?;
        d.set_displacement(gpu, u, ub)?;
        Ok(d)
    }

    /// The free-expansion state of Gate 95-B, uploaded exactly, is a fixed
    /// point of the BLOCK map: one `apply_block` at `u = alpha dT x`
    /// returns it, the defect the block solve's own tolerance.
    #[test]
    fn the_free_expansion_state_is_a_fixed_point_of_the_block_map() {
        let Some(gpu) = gpu() else { return };
        let mat = Material::steel(0.3);
        let hm = prototype::block(10).expect("block");
        let gm = upload(&gpu, &hm);
        let n = hm.n_cells;
        let a = mat.alpha * DT;
        let u_exact: Vec<Vec3> = hm.c.iter().map(|x| *x * a).collect();
        let ub_exact: Vec<Vec3> = hm.b_cf.iter().map(|x| *x * a).collect();
        let mut d = free_expansion_state(&gpu, &hm, &gm, &u_exact, &ub_exact, block_tight())
            .expect("displacement");
        let mut cs = CoupledSolid::new(&gpu, &d).expect("coupled");
        let mut out = gpu.zeros(n).expect("zeros");
        let perf = apply_block(&gpu, &mut d, &mut cs, &mut out).expect("apply_block");
        let f = gpu.download(&out).expect("F(u)");
        let rel = rel_max(&flat3(&f), &flat3(&u_exact));
        println!(
            "  109.6: block-map free expansion rel = {rel:e}  BiCGStab = {} iterations",
            perf.n_iterations
        );
        assert!((rel as f64) <= crate::precision::roundoff_tol(1e-9), "the block map does not fix the free-expansion state: {rel:e}");
    }

    /// Gate 95-C's linear field on the graded block is a fixed point of the
    /// block map: prescribed by value on every face, one `apply_block`
    /// returns it.
    #[test]
    fn a_linear_displacement_is_a_fixed_point_of_the_block_map() {
        let Some(gpu) = gpu() else { return };
        let hm = graded_block();
        let gm = upload(&gpu, &hm);
        let n = hm.n_cells;
        let nbf = hm.n_boundary_faces;
        let (u, ub) = linear_state(&hm);
        let mut d = Displacement::new(
            &gpu,
            &gm,
            &hm,
            Material::steel(0.3),
            &[[bc::CompBc::Fixed(0.0); 3]; 6],
            block_tight(),
        )
        .expect("displacement");
        d.set_temperature(&gpu, &vec![T_REF; n], &vec![T_REF; nbf], T_REF)
            .expect("temperature");
        d.bcs.set_fixed_values(&gpu, &ub).expect("per-face fixed values");
        d.set_displacement(&gpu, &u, &ub).expect("state");
        let mut cs = CoupledSolid::new(&gpu, &d).expect("coupled");
        let mut out = gpu.zeros(n).expect("zeros");
        let perf = apply_block(&gpu, &mut d, &mut cs, &mut out).expect("apply_block");
        let f = gpu.download(&out).expect("F(u)");
        let rel = rel_max(&flat3(&f), &flat3(&u));
        println!(
            "  109.6: block-map linear state rel = {rel:e}  BiCGStab = {} iterations",
            perf.n_iterations
        );
        assert!((rel as f64) <= crate::precision::roundoff_tol(1e-9), "the block map does not fix the linear state: {rel:e}");
    }

    /// From rest, the coupled loop finds the free-expansion state; the
    /// segregated loop on a fresh displacement of the same problem runs
    /// beside it, its count and contraction printed, not held.
    #[test]
    // f32 (F32-16): coupled loop does not converge in f32
    #[cfg_attr(feature = "single", ignore = "fails at f32: SPEC-LIT 112.3")]
    fn the_coupled_loop_finds_the_free_expansion_state_from_rest() {
        let Some(gpu) = gpu() else { return };
        let mat = Material::steel(0.3);
        let hm = prototype::block(8).expect("block");
        let gm = upload(&gpu, &hm);
        let n = hm.n_cells;
        let nbf = hm.n_boundary_faces;
        let a = mat.alpha * DT;
        let zeros: Vec<Vec3> = vec![Vec3::ZERO; n];
        let zeros_b: Vec<Vec3> = vec![Vec3::ZERO; nbf];
        let ctrl = OuterControls {
            relaxation: Relaxation::Anderson(ANDERSON_DEPTH),
            decades: 10.0,
            max_outer: 200,
            boundary_passes: 3,
        };
        let mut d = free_expansion_state(&gpu, &hm, &gm, &zeros, &zeros_b, block_tight())
            .expect("displacement");
        let rep = solve(&gpu, &mut d, &ctrl).expect("coupled solve");
        let u = gpu.download(&d.u.f).expect("u");
        let u_exact: Vec<Vec3> = hm.c.iter().map(|x| *x * a).collect();
        let scale = u_exact.iter().map(|v| v.mag()).fold(0.0 as Scalar, |m, v| m.max(v));
        let err = u
            .iter()
            .zip(&u_exact)
            .map(|(g, w)| (*g - *w).mag())
            .fold(0.0 as Scalar, |m, v| m.max(v))
            / scale;
        let mut d_seg = free_expansion_state(&gpu, &hm, &gm, &zeros, &zeros_b, block_tight())
            .expect("segregated displacement");
        let seg = crate::solid::outer::solve(&gpu, &mut d_seg, &ctrl).expect("segregated solve");
        println!(
            "  109.6: from rest, block-coupled outer = {} (observed {:.4}), \
             segregated outer = {} (observed {:.4}); err = {err:e}",
            rep.iterations, rep.observed_contraction, seg.iterations, seg.observed_contraction
        );
        assert!(rep.converged, "the coupled loop did not converge");
        assert!(
            err <= 1e-8,
            "the coupled loop's state is not the free-expansion one: {err:e}"
        );
    }

    /// The largest cell error of `d.u.f` against Gate 95-B's `alpha dT x`,
    /// relative to the largest exact displacement.
    fn free_expansion_error(gpu: &Gpu, hm: &HostMesh, d: &Displacement<'_>) -> Scalar {
        let a = Material::steel(0.3).alpha * DT;
        let u = gpu.download(&d.u.f).expect("u");
        let u_exact: Vec<Vec3> = hm.c.iter().map(|x| *x * a).collect();
        let scale = u_exact.iter().map(|v| v.mag()).fold(0.0 as Scalar, |m, v| m.max(v));
        u.iter()
            .zip(&u_exact)
            .map(|(g, w)| (*g - *w).mag())
            .fold(0.0 as Scalar, |m, v| m.max(v))
            / scale
    }

    /// SPEC-LIT §142.2's gate: from rest on the `6^3` block, GMRES(50) on
    /// the block map's state `(u, ub)` drops the true residual eight decades
    /// in at most 20 outer iterations (applications of the map) and lands on
    /// the free-expansion state. The Anderson loop on the same problem runs
    /// beside it, its count printed, not held.
    #[test]
    // f32 (F32-16): GMRES misses eight decades in f32
    #[cfg_attr(feature = "single", ignore = "fails at f32: SPEC-LIT 112.3")]
    fn gmres_reaches_eight_decades_on_the_free_expansion_block_in_twenty() {
        let Some(gpu) = gpu() else { return };
        let hm = prototype::block(6).expect("block");
        let gm = upload(&gpu, &hm);
        let zeros: Vec<Vec3> = vec![Vec3::ZERO; hm.n_cells];
        let zeros_b: Vec<Vec3> = vec![Vec3::ZERO; hm.n_boundary_faces];
        let ctrl = |relaxation| OuterControls { relaxation, decades: 8.0, max_outer: 200, boundary_passes: 3 };
        let mut d = free_expansion_state(&gpu, &hm, &gm, &zeros, &zeros_b, block_tight())
            .expect("displacement");
        let rep = solve(&gpu, &mut d, &ctrl(Relaxation::Gmres(GMRES_RESTART))).expect("GMRES solve");
        let err = free_expansion_error(&gpu, &hm, &d);
        let mut d_aa = free_expansion_state(&gpu, &hm, &gm, &zeros, &zeros_b, block_tight())
            .expect("displacement");
        let aa = solve(&gpu, &mut d_aa, &ctrl(Relaxation::Anderson(ANDERSON_DEPTH))).expect("Anderson");
        let last = rep.norms.last().copied().unwrap_or(0.0);
        println!(
            "  142.2: block(6) free expansion, GMRES(50) outer = {} (decades {:.2}, linear {}), \
             Anderson(5) outer = {} (linear {}); err = {err:e}",
            rep.iterations,
            (rep.norms[0] / last).log10(),
            rep.linear_iterations,
            aa.iterations,
            aa.linear_iterations
        );
        assert!(rep.converged, "GMRES(50) did not reach eight decades: {:?}", rep.norms);
        assert!(rep.iterations <= 20, "GMRES(50) took {} outer iterations, more than 20", rep.iterations);
        assert!(rep.omegas.is_empty(), "GMRES records no relaxation factor");
        assert!(err <= 1e-8, "GMRES's state is not the free-expansion one: {err:e}");
    }

    /// The GMRES loop is deterministic: two runs from rest capped at the
    /// same six outer iterations leave the same displacement bit for bit
    /// (the host sums run in one order, the device kernels replay).
    #[test]
    fn gmres_runs_of_a_fixed_count_are_bitwise_identical() {
        let Some(gpu) = gpu() else { return };
        let hm = prototype::block(6).expect("block");
        let gm = upload(&gpu, &hm);
        let zeros: Vec<Vec3> = vec![Vec3::ZERO; hm.n_cells];
        let zeros_b: Vec<Vec3> = vec![Vec3::ZERO; hm.n_boundary_faces];
        let ctrl = OuterControls {
            relaxation: Relaxation::Gmres(GMRES_RESTART),
            decades: 30.0,
            max_outer: 6,
            boundary_passes: 3,
        };
        let run = || {
            let mut d = free_expansion_state(&gpu, &hm, &gm, &zeros, &zeros_b, block_tight())
                .expect("displacement");
            let rep = solve(&gpu, &mut d, &ctrl).expect("GMRES solve");
            assert_eq!(rep.iterations, 6, "the cap is six applications");
            assert!(!rep.converged);
            (gpu.download(&d.u.f).expect("u"), gpu.download(&d.u.bf).expect("ub"), rep.norms)
        };
        let (u1, b1, n1) = run();
        let (u2, b2, n2) = run();
        let bits = |v: &[Vec3]| -> Vec<u64> { v.iter().flat_map(|a| [a.x, a.y, a.z]).map(|s| s.to_bits() as u64).collect() };
        assert_eq!(bits(&u1), bits(&u2), "u differs between two identical GMRES runs");
        assert_eq!(bits(&b1), bits(&b2), "ub differs between two identical GMRES runs");
        assert_eq!(n1, n2, "the norm histories differ");
    }

    /// SPEC-LIT 81.7 for the GMRES matvec: the device work of one outer
    /// iteration is one `apply_block` from a written state, and with a
    /// fixed-iteration block solve it captures and replays bitwise.
    #[test]
    fn the_gmres_matvec_replays_bitwise() {
        let Some(gpu) = gpu() else { return };
        let hm = prototype::block(6).expect("block");
        let gm = upload(&gpu, &hm);
        let n = hm.n_cells;
        let a = Material::steel(0.3).alpha * DT;
        let u0: Vec<Vec3> = hm.c.iter().map(|x| *x * (0.5 * a)).collect();
        let ub0: Vec<Vec3> = vec![Vec3::ZERO; hm.n_boundary_faces];
        let mut ctrl = block_tight();
        ctrl.fixed_iters = true;
        ctrl.max_iter = 4;
        ctrl.report_residuals = false;
        let report = crate::capture::capture_replays_bitwise(
            &gpu,
            "solid block-coupled GMRES matvec, one apply_block at a fixed-iteration solve",
            || {
                let mut d = free_expansion_state(&gpu, &hm, &gm, &u0, &ub0, ctrl)?;
                d.passes = 3;
                let cs = CoupledSolid::new(&gpu, &d)?;
                let out = gpu.zeros(n)?;
                Ok((d, cs, out))
            },
            |(d, cs, out)| apply_block(&gpu, d, cs, out).map(|_| ()),
            |(d, cs, out)| {
                Ok(vec![
                    ("out", flat3(&gpu.download(out)?)),
                    ("ub", flat3(&gpu.download(&d.u.bf)?)),
                    ("x", gpu.download(&cs.cw.x)?),
                    ("r", gpu.download(&cs.cw.w.r)?),
                ])
            },
        )
        .expect("SPEC-LIT 81.7: the GMRES matvec must capture and replay bitwise");
        println!("  GMRES matvec: {report}");
    }

    /// A wrong-length output is refused by name, before anything launches.
    #[test]
    fn the_block_map_refuses_a_wrong_output_by_name() {
        let Some(gpu) = gpu() else { return };
        let hm = prototype::block(8).expect("block");
        let gm = upload(&gpu, &hm);
        let n = hm.n_cells;
        let nbf = hm.n_boundary_faces;
        let zeros: Vec<Vec3> = vec![Vec3::ZERO; n];
        let zeros_b: Vec<Vec3> = vec![Vec3::ZERO; nbf];
        let mut d = free_expansion_state(&gpu, &hm, &gm, &zeros, &zeros_b, block_tight())
            .expect("displacement");
        let mut cs = CoupledSolid::new(&gpu, &d).expect("coupled");
        let mut out = gpu.zeros(n + 1).expect("zeros");
        let err = apply_block(&gpu, &mut d, &mut cs, &mut out)
            .expect_err("the wrong-length output must be refused");
        match err {
            Error::Config(msg) => {
                assert!(msg.contains("out"), "the refusal does not name out: {msg}");
                println!("  109.6: refusal: {msg}");
            }
            other => panic!("not a Config refusal: {other:?}"),
        }
    }

    /// The end-loaded cantilever, isothermal, its per-face load and clamp
    /// uploaded, ready for either loop - a FRESH displacement per call.
    fn cantilever_state<'m>(
        gpu: &Gpu,
        hm: &'m HostMesh,
        gm: &'m GpuMesh,
        tr: &[Vec3],
        fv: &[Vec3],
        ctrl: SolverControls,
    ) -> Result<Displacement<'m>> {
        let n = hm.n_cells;
        let nbf = hm.n_boundary_faces;
        let mut d = Displacement::new(
            gpu,
            gm,
            hm,
            fixtures::cantilever_material(),
            &fixtures::cantilever_bcs(),
            ctrl,
        )?;
        d.set_temperature(gpu, &vec![T_REF; n], &vec![T_REF; nbf], T_REF)?;
        d.bcs.set_traction_values(gpu, tr)?;
        d.bcs.set_fixed_values(gpu, fv)?;
        Ok(d)
    }

    /// The measurement itself: three slendernesses, the block-coupled loop
    /// beside the segregated one, one row per method.
    #[test]
    #[ignore = "the measurement itself; the supervisor runs it with --ignored --nocapture"]
    fn the_block_coupled_cantilever_sweep() {
        let Some(gpu) = gpu() else { return };
        let ctrl = OuterControls {
            relaxation: Relaxation::Anderson(ANDERSON_DEPTH),
            decades: 8.0,
            max_outer: 1000,
            boundary_passes: 3,
        };
        for (l, label) in [(0.5 as Scalar, "2.5:1"), (1.0 as Scalar, "5:1"), (2.0 as Scalar, "10:1")] {
            let n_y = 8usize;
            let n_x = (l * n_y as Scalar / 0.2).round() as usize;
            let hm = fixtures::cantilever_mesh(n_x, n_y, l, fixtures::CANTILEVER_C).expect("mesh");
            let exact = fixtures::CantileverExact::new(
                fixtures::CANTILEVER_E,
                fixtures::CANTILEVER_NU,
                l,
                fixtures::CANTILEVER_C,
                fixtures::CANTILEVER_P,
            );
            let (tr, fv) = fixtures::cantilever_face_values(&hm, &exact);
            let gm = upload(&gpu, &hm);
            let tip = exact.tip();

            let mut d = cantilever_state(&gpu, &hm, &gm, &tr, &fv, block_tight())
                .expect("block-coupled displacement");
            match solve(&gpu, &mut d, &ctrl) {
                Ok(rep) => {
                    let ub = gpu.download(&d.u.bf).expect("ub");
                    let v_h = fixtures::cantilever_tip_deflection(&hm, &ub);
                    println!(
                        "  95-A {label} block-coupled: outer={} converged={} observed={:.4} \
                         linear={} tip_rel={:.3e}",
                        rep.iterations,
                        rep.converged,
                        rep.observed_contraction,
                        rep.linear_iterations,
                        ((v_h - tip) / tip.abs()).abs()
                    );
                }
                Err(e) => println!("  95-A {label} block-coupled: {e}"),
            }
            let mut d = cantilever_state(&gpu, &hm, &gm, &tr, &fv, block_tight())
                .expect("segregated displacement");
            match crate::solid::outer::solve(&gpu, &mut d, &ctrl) {
                Ok(rep) => {
                    let ub = gpu.download(&d.u.bf).expect("ub");
                    let v_h = fixtures::cantilever_tip_deflection(&hm, &ub);
                    println!(
                        "  95-A {label} segregated:    outer={} converged={} observed={:.4} \
                         linear={} tip_rel={:.3e}",
                        rep.iterations,
                        rep.converged,
                        rep.observed_contraction,
                        rep.linear_iterations,
                        ((v_h - tip) / tip.abs()).abs()
                    );
                }
                Err(e) => println!("  95-A {label} segregated:    {e}"),
            }
        }
    }

    /// SPEC-LIT §142.2's probe: how far each loop's eight-decade answer on
    /// Gate 95-A's finer slender beams is from the fixed point. Each beam is
    /// solved three times - Anderson(5) and GMRES(50) at the gate's controls
    /// (eight decades, the block solve at 1e-12), and GMRES(50) at twelve
    /// decades with the block solve at 1e-14 as the reference - and the
    /// distances and `e_u` are printed, with `p_u` on 16 -> 32 per loop.
    #[test]
    #[cfg(not(feature = "single"))]
    #[ignore = "the VAL-11 probe; the supervisor runs it with --ignored --nocapture"]
    fn gate_95a_probe_how_far_eight_decades_lands_from_the_fixed_point() {
        let Some(gpu) = gpu() else { return };
        let gate_linear = |tol: Scalar| SolverControls {
            tolerance: tol,
            max_iter: 20000,
            ..block_tight()
        };
        let outer = |relaxation, decades| OuterControls {
            relaxation,
            decades,
            max_outer: 1000,
            boundary_passes: 3,
        };
        let runs = [
            ("anderson-8", Relaxation::Anderson(ANDERSON_DEPTH), 8.0, 1e-12),
            ("gmres-8", Relaxation::Gmres(GMRES_RESTART), 8.0, 1e-12),
            ("gmres-12", Relaxation::Gmres(GMRES_RESTART), 12.0, 1e-14),
        ];
        for (l, label) in [(1.0 as Scalar, "5:1"), (2.0 as Scalar, "10:1")] {
            let mut e_u = [[0.0 as Scalar; 2]; 3];
            for (k, n_y) in [16usize, 32].into_iter().enumerate() {
                let n_x = (l * n_y as Scalar / 0.2).round() as usize;
                let hm = fixtures::cantilever_mesh(n_x, n_y, l, fixtures::CANTILEVER_C).expect("mesh");
                let exact = fixtures::CantileverExact::new(
                    fixtures::CANTILEVER_E,
                    fixtures::CANTILEVER_NU,
                    l,
                    fixtures::CANTILEVER_C,
                    fixtures::CANTILEVER_P,
                );
                let (tr, fv) = fixtures::cantilever_face_values(&hm, &exact);
                let gm = upload(&gpu, &hm);
                let mut states: Vec<Vec<Vec3>> = Vec::new();
                for (j, (name, relaxation, decades, tol)) in runs.iter().enumerate() {
                    let mut d = cantilever_state(&gpu, &hm, &gm, &tr, &fv, gate_linear(*tol))
                        .expect("displacement");
                    let rep = solve(&gpu, &mut d, &outer(*relaxation, *decades)).expect("solve");
                    let u = gpu.download(&d.u.f).expect("u");
                    let mut sf = StressFields::new(&gpu, hm.n_cells).expect("stress");
                    sf.compute(&gpu, &d.material, &d.grad, &d.u.f, &d.t, d.t_ref).expect("compute");
                    let h = sf.download(&gpu).expect("download");
                    e_u[j][k] = fixtures::cantilever_errors(&hm, &u, &h.sigma, &exact).e_u;
                    println!(
                        "  142.2 probe {label} n_y={n_y} {name}: outer={} converged={} e_u={:.4e}",
                        rep.iterations, rep.converged, e_u[j][k]
                    );
                    states.push(u);
                }
                let reference = &states[2];
                let scale = reference.iter().map(|v| v.mag()).fold(0.0 as Scalar, Scalar::max);
                for j in 0..2 {
                    let dist = states[j]
                        .iter()
                        .zip(reference)
                        .map(|(a, b)| (*a - *b).mag())
                        .fold(0.0 as Scalar, Scalar::max)
                        / scale;
                    println!(
                        "  142.2 probe {label} n_y={n_y} {} from gmres-12: max|du|/max|u| = {dist:.3e}",
                        runs[j].0
                    );
                }
            }
            for (j, run) in runs.iter().enumerate() {
                let p_u = (e_u[j][0] / e_u[j][1]).ln() / (2.0 as Scalar).ln();
                println!("  142.2 probe {label} {}: p_u 16 -> 32 = {p_u:.3}", run.0);
            }
        }
    }

    // ======================================================================
    //  VAL-10: the exact discrete fixed point of the block map, by a dense
    //  direct solve in plain f64 (§142.1). In linear elasticity the block
    //  matrix A is fixed and the source affine in u once the boundary is
    //  closed, so R(u) = B u + R(0) and the fixed point solves B u = -R(0);
    //  B is built column by column, B e_k = R(e_k) - R(0). Against this the
    //  outer loop's miss at 5:1 and 10:1 is attributed: the loop, or the
    //  discretisation itself.
    // ======================================================================

    /// Right-looking Gaussian elimination with partial pivoting, row-major
    /// `a[i*n + j]`: `P B = L U`, L unit lower below the diagonal, U on and
    /// above it. `ipiv[k]` is the row swapped with row k at step k (LAPACK
    /// style). `Err(k)` when the largest `|a_ik|`, i >= k, is 0 or not
    /// finite.
    fn lu_factor(a: &mut [f64], n: usize) -> std::result::Result<Vec<usize>, usize> {
        debug_assert_eq!(a.len(), n * n);
        let mut ipiv = vec![0usize; n];
        for k in 0..n {
            let (mut p, mut best) = (k, a[k * n + k].abs());
            for i in k + 1..n {
                let v = a[i * n + k].abs();
                if v > best {
                    best = v;
                    p = i;
                }
            }
            if !best.is_finite() || best == 0.0 {
                return Err(k);
            }
            ipiv[k] = p;
            if p != k {
                for j in 0..n {
                    a.swap(k * n + j, p * n + j);
                }
            }
            let piv = a[k * n + k];
            for i in k + 1..n {
                a[i * n + k] /= piv;
            }
            for i in k + 1..n {
                let lik = a[i * n + k];
                if lik != 0.0 {
                    // The rank-1 update over the two contiguous row tails,
                    // so the zip vectorises.
                    let (head, tail) = a.split_at_mut(i * n);
                    let (row_i, _) = tail.split_at_mut(n);
                    let row_k = &head[k * n..k * n + n];
                    for (ri, rk) in row_i[k + 1..].iter_mut().zip(row_k[k + 1..].iter()) {
                        *ri -= lik * rk;
                    }
                }
            }
        }
        Ok(ipiv)
    }

    /// `B x = b` in place, from [`lu_factor`]'s output: the swaps forward
    /// in `ipiv` order, the unit-lower forward substitution, the upper back
    /// substitution.
    fn lu_solve(lu: &[f64], ipiv: &[usize], b: &mut [f64]) {
        let n = ipiv.len();
        for (k, &p) in ipiv.iter().enumerate() {
            if p != k {
                b.swap(k, p);
            }
        }
        for i in 1..n {
            let s: f64 = lu[i * n..i * n + i].iter().zip(&b[..i]).map(|(a, c)| a * c).sum();
            b[i] -= s;
        }
        for i in (0..n).rev() {
            let s: f64 = lu[i * n + i + 1..(i + 1) * n]
                .iter()
                .zip(&b[i + 1..n])
                .map(|(a, c)| a * c)
                .sum();
            b[i] = (b[i] - s) / lu[i * n + i];
        }
    }

    /// `B^T x = b` in place: `B^T = U^T L^T P`, so the lower-triangular
    /// `U^T` forward (the columns of U, strided), the unit-diagonal `L^T`
    /// backward, then the swaps in REVERSE order.
    fn lu_solve_transposed(lu: &[f64], ipiv: &[usize], b: &mut [f64]) {
        let n = ipiv.len();
        for i in 0..n {
            let mut s = b[i];
            for (j, bj) in b[..i].iter().enumerate() {
                s -= lu[j * n + i] * bj;
            }
            b[i] = s / lu[i * n + i];
        }
        for i in (0..n).rev() {
            let mut s = b[i];
            for j in i + 1..n {
                s -= lu[j * n + i] * b[j];
            }
            b[i] = s;
        }
        for k in (0..n).rev() {
            let p = ipiv[k];
            if p != k {
                b.swap(k, p);
            }
        }
    }

    /// The matrix one-norm: the largest absolute column sum.
    fn norm1(a: &[f64], n: usize) -> f64 {
        (0..n)
            .map(|j| (0..n).map(|i| a[i * n + j].abs()).sum::<f64>())
            .fold(0.0f64, f64::max)
    }

    /// Hager's estimate of `||B^-1||_1` (SIAM J. Sci. Stat. Comput. 5
    /// (1984) 311-316), the LAPACK xLACON form (Higham, ACM TOMS 14 (1988)
    /// 381-396): `x` walks to the column where the inverse's action peaks
    /// on the sign pattern of the previous solve, at most five steps, and
    /// the last `||B^-1 x||_1` is returned.
    fn hager_inv_norm1(lu: &[f64], ipiv: &[usize], n: usize) -> f64 {
        let mut x = vec![1.0 / n as f64; n];
        let mut last = 0.0f64;
        for it in 1..=5 {
            let mut y = x.clone();
            lu_solve(lu, ipiv, &mut y);
            last = y.iter().fold(0.0f64, |a, &v| a + v.abs());
            let xi: Vec<f64> =
                y.iter().map(|&v| if v >= 0.0 { 1.0 } else { -1.0 }).collect();
            let mut z = xi;
            lu_solve_transposed(lu, ipiv, &mut z);
            // The FIRST index of the largest |z_j|: strict `>` keeps it.
            let (j, zmax) = z.iter().enumerate().fold((0usize, 0.0f64), |(bj, bv), (i, &v)| {
                if v.abs() > bv {
                    (i, v.abs())
                } else {
                    (bj, bv)
                }
            });
            let z_dot_x: f64 = z.iter().zip(&x).map(|(a, c)| a * c).sum();
            if it > 1 && zmax <= z_dot_x {
                break;
            }
            x = vec![0.0f64; n];
            x[j] = 1.0;
        }
        last
    }

    /// The boundary values closed at the CURRENT cell field: `d.passes`
    /// forced to 1 and [`Displacement::correct_boundary`] repeated until the
    /// boundary values stop moving - a change at or below `1e-14` times the
    /// larger of the cell and boundary magnitudes, or exactly still -
    /// `d.passes` restored on every path. The number of calls is returned;
    /// the cap is refused by name.
    #[cfg(not(feature = "single"))]
    fn close_boundary(gpu: &Gpu, d: &mut Displacement<'_>, max_passes: usize) -> Result<usize> {
        let saved = d.passes;
        d.passes = 1;
        let out = (|| -> Result<usize> {
            let u = gpu.download(&d.u.f)?;
            let max_c = u.iter().map(|v| v.mag()).fold(0.0 as Scalar, |a, v| a.max(v));
            let mut ub_old = gpu.download(&d.u.bf)?;
            let mut last_change = Scalar::INFINITY;
            for calls in 1..=max_passes {
                d.correct_boundary(gpu)?;
                let ub_new = gpu.download(&d.u.bf)?;
                let change = ub_new
                    .iter()
                    .zip(&ub_old)
                    .map(|(a, b)| (*a - *b).mag())
                    .fold(0.0 as Scalar, |a, v| a.max(v));
                let still =
                    ub_new.iter().map(|v| v.mag()).fold(0.0 as Scalar, |a, v| a.max(v));
                ub_old = ub_new;
                last_change = change;
                if change == 0.0 || change <= 1e-14 * max_c.max(still) {
                    return Ok(calls);
                }
            }
            Err(Error::Config(format!(
                "boundary closure did not still in {max_passes} passes; the last change was \
                 {last_change:e}"
            )))
        })();
        d.passes = saved;
        out
    }

    /// SPEC-LIT (109.6)'s `R(u) = A u - source` at the closed state: `u`
    /// uploaded with zero boundary values, the boundary closed, the operator
    /// assembled there, the residual downloaded and flattened `3c + i` as
    /// f64, with the closure's pass count.
    #[cfg(not(feature = "single"))]
    fn closed_residual(
        gpu: &Gpu,
        d: &mut Displacement<'_>,
        op: &mut BlockOperator,
        r: &mut DevBuf<Vec3>,
        u: &[Vec3],
    ) -> Result<(Vec<f64>, usize)> {
        let ub = vec![Vec3::ZERO; d.mesh().n_boundary_faces];
        d.set_displacement(gpu, u, &ub)?;
        let passes = close_boundary(gpu, d, 2000)?;
        op.assemble(gpu, d)?;
        op.residual(gpu, r, &d.u.f)?;
        let rv = gpu.download(r)?;
        Ok((
            rv.iter().flat_map(|a| [a.x as f64, a.y as f64, a.z as f64]).collect(),
            passes,
        ))
    }

    /// The affine block map's Jacobian, column by column: `r0 = R(0)`, then
    /// for every flat index `k = 3c + i` the unit vector's residual - column
    /// `k` of `B`, row-major as `b[row * n3 + k]`. The largest closure pass
    /// count over all evaluations comes back with them.
    #[cfg(not(feature = "single"))]
    fn dense_block_map(
        gpu: &Gpu,
        d: &mut Displacement<'_>,
        op: &mut BlockOperator,
        r: &mut DevBuf<Vec3>,
    ) -> Result<(Vec<f64>, Vec<f64>, usize)> {
        let n = d.mesh().n_cells;
        let n3 = 3 * n;
        let zero: Vec<Vec3> = vec![Vec3::ZERO; n];
        let (r0, mut passes) = closed_residual(gpu, d, op, r, &zero)?;
        let mut b = vec![0.0f64; n3 * n3];
        let mut ek = vec![Vec3::ZERO; n];
        for k in 0..n3 {
            let (c, i) = (k / 3, k % 3);
            ek.iter_mut().for_each(|v| *v = Vec3::ZERO);
            match i {
                0 => ek[c].x = 1.0,
                1 => ek[c].y = 1.0,
                _ => ek[c].z = 1.0,
            }
            let (rv, p) = closed_residual(gpu, d, op, r, &ek)?;
            passes = passes.max(p);
            for (row, v) in rv.iter().enumerate() {
                b[row * n3 + k] = v - r0[row];
            }
        }
        Ok((b, r0, passes))
    }

    /// The probe's linear controls: [`block_tight`] with the iteration cap
    /// the slender meshes need. The tolerance is two decades tighter than
    /// [`block_tight`]: the outer loop stops at 12 decades of its residual,
    /// and the distance its state can sit from the dense fixed point is
    /// floored by the inner solve's own error - at 1e-12 that floor measured
    /// 3.6e-9 on 2.5:1 n_y=8, over the probe's 1e-9 bar.
    #[cfg(not(feature = "single"))]
    fn probe_linear() -> SolverControls {
        SolverControls { tolerance: 1e-14, max_iter: 20000, ..block_tight() }
    }

    /// The probe's outer controls: deeper decades than the sweep, easy to
    /// change by hand.
    #[cfg(not(feature = "single"))]
    fn probe_outer() -> OuterControls {
        OuterControls {
            relaxation: Relaxation::Anderson(ANDERSON_DEPTH),
            decades: 12.0,
            max_outer: 1000,
            boundary_passes: 3,
        }
    }

    #[test]
    fn dense_lu_solves_both_ways_and_hager_estimates_the_inverse_norm() {
        let b0 = [0.0, 2.0, 1.0, 3.0, 6.0, -4.0, 2.0, 1.0, 8.0];
        let mut a = b0;
        let ipiv = lu_factor(&mut a, 3).expect("factor");
        let mut b = vec![7.0, 3.0, 28.0];
        lu_solve(&a, &ipiv, &mut b);
        for (got, want) in b.iter().zip([1.0, 2.0, 3.0]) {
            assert!((got - want).abs() <= 1e-14, "lu_solve: {got} vs {want}");
        }
        let mut b = vec![12.0, 17.0, 17.0];
        lu_solve_transposed(&a, &ipiv, &mut b);
        for (got, want) in b.iter().zip([1.0, 2.0, 3.0]) {
            assert!((got - want).abs() <= 1e-14, "lu_solve_transposed: {got} vs {want}");
        }
        assert_eq!(norm1(&b0, 3), 13.0, "the one-norm of B");
        let h = hager_inv_norm1(&a, &ipiv, 3);
        assert!(
            (h - 93.0 / 73.0).abs() <= 1e-14,
            "Hager's estimate {h} is not the exact ||B^-1||_1 = 93/73"
        );
    }

    #[test]
    fn dense_lu_refuses_a_singular_matrix_by_its_pivot_column() {
        let mut a = vec![1.0, 2.0, 2.0, 4.0];
        assert_eq!(lu_factor(&mut a, 2), Err(1));
    }

    /// The VAL-10 probe: Gate 95-A's block map taken to its EXACT discrete
    /// fixed point by a dense f64 direct solve, so the miss at 5:1 and 10:1
    /// is attributed - the outer loop, or the discretisation. One line per
    /// mesh, the loop's own state beside the dense one on 2.5:1, and the
    /// observed orders per ratio.
    #[test]
    #[cfg(not(feature = "single"))]
    #[ignore = "the VAL-10 probe; the supervisor runs it with --ignored --nocapture"]
    fn gate_95a_probe_the_exact_fixed_point_of_the_block_map() {
        let Some(gpu) = gpu() else { return };
        let ln2 = (2.0 as Scalar).ln();
        for (label, l, levels) in [
            ("2.5:1", 0.5 as Scalar, vec![4usize, 8, 16]),
            ("5:1", 1.0, vec![4, 8, 16]),
            ("10:1", 2.0, vec![4, 8]),
        ] {
            let mut e_us = [0.0 as Scalar; 3];
            let mut e_xxs = [0.0 as Scalar; 3];
            let mut e_xys = [0.0 as Scalar; 3];
            for (idx, &n_y) in levels.iter().enumerate() {
                let t0 = std::time::Instant::now();
                let n_x = (l * n_y as Scalar / 0.2).round() as usize;
                let hm = fixtures::cantilever_mesh(n_x, n_y, l, fixtures::CANTILEVER_C)
                    .expect("mesh");
                let exact = fixtures::CantileverExact::new(
                    fixtures::CANTILEVER_E,
                    fixtures::CANTILEVER_NU,
                    l,
                    fixtures::CANTILEVER_C,
                    fixtures::CANTILEVER_P,
                );
                let (tr, fv) = fixtures::cantilever_face_values(&hm, &exact);
                let gm = upload(&gpu, &hm);
                let n = hm.n_cells;
                let n3 = 3 * n;
                let tip = exact.tip();
                let mut d = cantilever_state(&gpu, &hm, &gm, &tr, &fv, block_tight())
                    .expect("displacement");
                let mut op = BlockOperator::new(&gpu, &d).expect("op");
                let mut r = gpu.zeros(n).expect("residual buffer");
                let (b, r0, passes) =
                    dense_block_map(&gpu, &mut d, &mut op, &mut r).expect("dense block map");
                let nb = norm1(&b, n3);
                let mut lu = b.clone();
                let ipiv = match lu_factor(&mut lu, n3) {
                    Ok(p) => p,
                    Err(col) => {
                        println!("  142.1 {label} n_y={n_y}: exists=no (zero pivot at column {col})");
                        continue;
                    }
                };
                let mut x: Vec<f64> = r0.iter().map(|v| -v).collect();
                lu_solve(&lu, &ipiv, &mut x);
                let u_dense = unflatten3(&x);
                // The dense algebra's own consistency, then the device's:
                // host_res speaks for B x + R(0), res for the residual the
                // operator reports at u_dense (SPEC-LIT (109.6)).
                let mut bx = vec![0.0f64; n3];
                for (row, brow) in b.chunks_exact(n3).enumerate() {
                    bx[row] = brow.iter().zip(&x).map(|(a, c)| a * c).sum::<f64>() + r0[row];
                }
                let host_res = bx.iter().fold(0.0f64, |m, v| m.max(v.abs()));
                let cond1 = nb * hager_inv_norm1(&lu, &ipiv, n3);
                let (rv, _) = closed_residual(&gpu, &mut d, &mut op, &mut r, &u_dense)
                    .expect("residual at u_dense");
                let a_host = HostBlockLdu::download(&gpu, op.matrix()).expect("matrix");
                let scale = residual_scale(&a_host, &u_dense);
                let res = rv
                    .chunks_exact(3)
                    .map(|t| (t[0] * t[0] + t[1] * t[1] + t[2] * t[2]).sqrt())
                    .fold(0.0f64, |m, v| m.max(v))
                    / scale;
                assert!(
                    res <= 1e-12,
                    "SPEC-LIT (109.6): the device residual at the dense fixed point is \
                     {res:e} on {label} n_y={n_y}"
                );
                let exists = cond1.is_finite() && cond1 < 1e15 && res <= 1e-12;
                // d now sits at u_dense with its boundary closed, so the
                // gradient is the one the stress read-out asks for.
                let mut sf = StressFields::new(&gpu, n).expect("stress fields");
                sf.compute(&gpu, &d.material, &d.grad, &d.u.f, &d.t, d.t_ref)
                    .expect("stress");
                let h = sf.download(&gpu).expect("stress download");
                let errs = fixtures::cantilever_errors(&hm, &u_dense, &h.sigma, &exact);
                let ub = gpu.download(&d.u.bf).expect("ub");
                let v_h = fixtures::cantilever_tip_deflection(&hm, &ub);
                let e_tip = ((v_h - tip) / tip.abs()).abs();
                e_us[idx] = errs.e_u;
                e_xxs[idx] = errs.e_xx;
                e_xys[idx] = errs.e_xy;
                let mut loop_line = String::new();
                if label == "2.5:1" {
                    let mut d2 = cantilever_state(&gpu, &hm, &gm, &tr, &fv, probe_linear())
                        .expect("loop displacement");
                    let rep = solve(&gpu, &mut d2, &probe_outer()).expect("outer loop");
                    let u_loop = gpu.download(&d2.u.f).expect("u_loop");
                    let rel = rel_max(&flat3(&u_loop), &flat3(&u_dense));
                    assert!(rep.converged, "the outer loop did not converge on {label} n_y={n_y}");
                    assert!(
                        rel <= 1e-9,
                        "the loop state is not the dense fixed point on {label} n_y={n_y}: \
                         {rel:e}"
                    );
                    loop_line = format!(" outer={} rel={rel:.3e}", rep.iterations);
                }
                print!(
                    "  142.1 {label} n_y={n_y} cells={n} N={n3} passes={passes} \
                     cond1={cond1:.3e} exists={exists} res/scale={res:.3e} \
                     host_res={host_res:.3e} e_tip={e_tip:.3e} e_u={:.3e} e_xx={:.3e} \
                     e_xy={:.3e}{loop_line} ({:.1}s)",
                    errs.e_u,
                    errs.e_xx,
                    errs.e_xy,
                    t0.elapsed().as_secs_f32()
                );
                println!();
            }
            let pu = |coarse: Scalar, fine: Scalar| (coarse / fine).ln() / ln2;
            let psig = |cx: Scalar, cy: Scalar, fx: Scalar, fy: Scalar| {
                (cx / fx).ln().min((cy / fy).ln()) / ln2
            };
            if levels.len() == 3 {
                println!(
                    "  142.1 {label}: p_u(4->8)={:.3} p_u(8->16)={:.3} \
                     p_sigma(4->8)={:.3} p_sigma(8->16)={:.3}",
                    pu(e_us[0], e_us[1]),
                    pu(e_us[1], e_us[2]),
                    psig(e_xxs[0], e_xys[0], e_xxs[1], e_xys[1]),
                    psig(e_xxs[1], e_xys[1], e_xxs[2], e_xys[2])
                );
            } else {
                println!(
                    "  142.1 {label}: p_u(4->8)={:.3} p_sigma(4->8)={:.3}",
                    pu(e_us[0], e_us[1]),
                    psig(e_xxs[0], e_xys[0], e_xxs[1], e_xys[1])
                );
            }
        }
    }
}
