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
    #[cfg_attr(feature = "single", ignore = "fails at f32: SPEC-LIT 112.3")]
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
            assert!(e <= 1e-12, "block-DILU {name}: rel_max = {e}");
        }
    }

    /// R3's Jacobi half: the device `3x3` diagonal inverse against the host's.
    #[test]
    #[cfg_attr(feature = "single", ignore = "fails at f32: SPEC-LIT 112.3")]
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
            assert!(e <= 1e-12, "block-Jacobi {name}: rel_max = {e}");
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
    #[cfg_attr(feature = "single", ignore = "fails at f32: SPEC-LIT 112.3")]
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
        assert!(e_dilu <= 1e-10, "block-DILU vs dense: {e_dilu}");
        assert!(e_jacobi <= 1e-10, "block-Jacobi vs dense: {e_jacobi}");
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
}
