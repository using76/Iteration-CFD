// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! The SPMD PCG: §73's `dist_pcg`, one part per rank (SPEC-LIT §138.5).
//!
//! [`dist_pcg`](super::dist_pcg) holds every part in one process. In an
//! SPMD run a rank holds ONE part and reaches the others only through a
//! [`Comm`](crate::comm::Comm), so the solve is the same program on every
//! rank with §73's two cross-part steps moved onto the two transports: the
//! matrix product's halo fill goes through
//! [`SpmdHalo::scalar`](crate::halo::SpmdHalo::scalar) (§138.3) and every
//! scalar reduction goes through the [`SpmdReduction`] method of the same
//! name (§138.4). Both transports are bit for bit the in-process forms, and
//! everything between them - the kernels, their arguments, their launch
//! geometry and their order - is `dist_pcg`'s for one part, unchanged. So
//! every rank's field is bit for bit what `dist_pcg` produces for its part,
//! and, the mean still dividing by the whole mesh's cell count, the ranks
//! all stop on the same iteration.
//!
//! Every rank decides alone and every rank decides alike: the residual sum
//! and the normalisation factor are global values with the same bits on
//! every rank, and the convergence test is one kernel on those values. The
//! **flag vote** checks that instead of trusting it: after each read of the
//! convergence flag the ranks allgather one byte each, and a rank that sees
//! two different votes refuses by name. It is on by default in a debug
//! build; a `fixed_iters` solve reads no flag and takes none.
//!
//! Every exchange and every collective is a host round trip, so
//! [`spmd_pcg`] inside `Gpu::capture` is refused by name (§81.3).
//!
//! Provenance: ORIGINAL - the SPMD form is this project's own design
//! (SPEC-LIT §138.5, PAR-04a); it runs §73's kernels and recurrence
//! unchanged, one part per rank, through the transports of §138.3 and
//! §138.4. No external source. No GPL-licensed source was consulted.

use cudarc::driver::PushKernelArg;

use crate::comm::Comm;
use crate::device::{cfg_for, DevBuf, Gpu};
use crate::error::{Error, Result};
use crate::exactsum::SpmdReduction;
use crate::halo::SpmdHalo;
use crate::solver::{self, Preconditioner, SolverControls, SolverKernels};
use crate::{Label, Scalar};

use super::{DistPerformance, DistReduce, DistSystem};

/// The four reduction shapes a Krylov solve needs, on ONE rank's buffers -
/// the parent's `Reduction`, one buffer where that holds a part's slice.
enum What<'a> {
    Sum(&'a DevBuf<Scalar>),
    SumMag(&'a DevBuf<Scalar>),
    Dot(&'a DevBuf<Scalar>, &'a DevBuf<Scalar>),
    /// `A psi`, `b`, `A x_ref`, and the epsilon added once at the end.
    NormFactor(
        &'a DevBuf<Scalar>,
        &'a DevBuf<Scalar>,
        &'a DevBuf<Scalar>,
        Scalar,
    ),
}

/// The cross-rank reduction, whichever construction was chosen, into `out`.
///
/// One function for all four shapes so that the choice between the two
/// guarantees is made in exactly one place, exactly as the parent's
/// `reduce_into` does for the in-process [`crate::exactsum::ExactReduction`].
#[allow(clippy::too_many_arguments)]
fn reduce_into(
    gpu: &Gpu,
    sol: &SolverKernels,
    comm: &dyn Comm,
    red: &mut SpmdReduction,
    mode: DistReduce,
    what: What<'_>,
    out: &mut DevBuf<Scalar>,
    perf: &mut DistPerformance,
) -> Result<()> {
    match (mode, what) {
        (DistReduce::Exact, What::Sum(x)) => red.sum(gpu, sol, comm, x)?,
        (DistReduce::Exact, What::SumMag(x)) => red.sum_mag(gpu, sol, comm, x)?,
        (DistReduce::Exact, What::Dot(a, b)) => red.dot(gpu, sol, comm, a, b)?,
        (DistReduce::Exact, What::NormFactor(ap, b, ax, e)) => {
            red.norm_factor(gpu, sol, comm, ap, b, ax, e)?
        }
        (DistReduce::Gathered, What::Sum(x)) => red.gathered_sum(gpu, sol, comm, x)?,
        (DistReduce::Gathered, What::SumMag(x)) => red.gathered_sum_mag(gpu, sol, comm, x)?,
        (DistReduce::Gathered, What::Dot(a, b)) => red.gathered_dot(gpu, sol, comm, a, b)?,
        (DistReduce::Gathered, What::NormFactor(ap, b, ax, e)) => {
            red.gathered_norm_factor(gpu, sol, comm, ap, b, ax, e)?
        }
    }
    perf.n_reductions += 1;
    solver::copy_scalar(gpu, sol, out, red.out())
}

/// `y = A x` on this rank's part, with `x`'s halo filled first.
///
/// [`SpmdHalo::scalar`] where the in-process `dist_amul` calls
/// [`crate::halo::HaloExchange::scalar`] - still the only place an exchange
/// happens, one per matrix product and no others.
#[allow(clippy::too_many_arguments)]
fn spmd_amul(
    gpu: &Gpu,
    sol: &SolverKernels,
    comm: &dyn Comm,
    halo: &mut SpmdHalo,
    sys: &DistSystem,
    y: &mut DevBuf<Scalar>,
    x: &mut DevBuf<Scalar>,
    perf: &mut DistPerformance,
) -> Result<()> {
    halo.scalar(gpu, comm, x)?;
    perf.n_exchanges += 1;
    solver::amul(gpu, sol, y, x, &sys.a[0], &sys.mesh[0])
}

/// What an SPMD solve needs that is not the system or the solution: ONE
/// rank's Krylov vectors, its [`SpmdReduction`], and the flag vote.
///
/// `Send`, so a rank can hold it inside its own thread, like [`SpmdHalo`]
/// and [`SpmdReduction`]. Nothing here allocates after `new`.
pub struct SpmdWorkspace {
    /// The rank `new` ran on.
    rank: usize,
    /// `comm.size()` seen by `new`.
    n_ranks: usize,
    /// Owned cells - what every kernel launches threads for.
    n: usize,
    /// Owned + halo - the length every field buffer must have.
    n_local: usize,
    reduce: DistReduce,
    /// The flag vote: on by default in a debug build.
    flag_vote: bool,
    /// Votes taken by this workspace so far.
    n_flag_votes: u64,

    // ---- Krylov vectors, `n_local` long - the ones PCG uses --------------
    r: DevBuf<Scalar>,
    p: DevBuf<Scalar>,
    v: DevBuf<Scalar>,
    p_hat: DevBuf<Scalar>,
    apsi: DevBuf<Scalar>,
    tmp: DevBuf<Scalar>,
    y: DevBuf<Scalar>,
    /// The right-hand side, copied out of `a.source` at the top of a solve.
    b: DevBuf<Scalar>,
    /// `1/diag` (Jacobi).
    r_diag: DevBuf<Scalar>,

    // ---- the cross-rank reduction ----------------------------------------
    red: SpmdReduction,

    // ---- scalars, one value each -----------------------------------------
    rho: DevBuf<Scalar>,
    alpha: DevBuf<Scalar>,
    beta: DevBuf<Scalar>,
    num: DevBuf<Scalar>,
    den: DevBuf<Scalar>,
    x_ref: DevBuf<Scalar>,
    norm_factor: DevBuf<Scalar>,
    initial_res: DevBuf<Scalar>,
    final_res: DevBuf<Scalar>,
    flag: DevBuf<Label>,
}

impl SpmdWorkspace {
    /// Allocate for the one-part system `sys` on this rank, reducing the way
    /// `reduce` says.
    ///
    /// It is a collective - [`SpmdReduction::new`]'s counts allgather - so
    /// every rank builds it at the same point of the program. The refusals
    /// run BEFORE that collective, in the table's order.
    pub fn new(gpu: &Gpu, comm: &dyn Comm, sys: &DistSystem, reduce: DistReduce) -> Result<Self> {
        let Some(part) = sys.rank_part() else {
            return Err(Error::Config(
                "distsolve: an SPMD workspace needs the one-part system \
                 DistSystem::split_one builds; this system holds every part, \
                 which DistSystem::split built (SPEC-LIT 138.5)"
                    .to_string(),
            ));
        };
        if comm.rank() != part {
            return Err(Error::Config(format!(
                "distsolve: this Comm is rank {}, but the system holds part {}",
                comm.rank(),
                part
            )));
        }
        let owned = sys.owned()[0];
        let local = sys.local()[0];
        let red = SpmdReduction::new(gpu, comm, owned)?;
        Ok(Self {
            rank: comm.rank(),
            n_ranks: comm.size(),
            n: owned,
            n_local: local,
            reduce,
            flag_vote: cfg!(debug_assertions),
            n_flag_votes: 0,
            red,
            r: gpu.zeros(local)?,
            p: gpu.zeros(local)?,
            v: gpu.zeros(local)?,
            p_hat: gpu.zeros(local)?,
            apsi: gpu.zeros(local)?,
            tmp: gpu.zeros(local)?,
            y: gpu.zeros(local)?,
            b: gpu.zeros(local)?,
            r_diag: gpu.zeros(local)?,
            rho: gpu.zeros(1)?,
            alpha: gpu.zeros(1)?,
            beta: gpu.zeros(1)?,
            num: gpu.zeros(1)?,
            den: gpu.zeros(1)?,
            x_ref: gpu.zeros(1)?,
            norm_factor: gpu.zeros(1)?,
            initial_res: gpu.zeros(1)?,
            final_res: gpu.zeros(1)?,
            flag: gpu.zeros(1)?,
        })
    }

    /// The rank `new` ran on.
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// The rank count `new` saw through the `Comm`.
    pub fn n_ranks(&self) -> usize {
        self.n_ranks
    }

    /// Which cross-rank reduction a solve with this workspace uses.
    pub fn reduce_mode(&self) -> DistReduce {
        self.reduce
    }

    /// Whether a solve with this workspace and `precon` produces bits that
    /// do not depend on how the mesh was cut - [`super::DistWorkspace`]'s
    /// answer, reached the same way.
    pub fn partition_invariant(&self, precon: Preconditioner) -> bool {
        let reduction_ok = self.reduce == DistReduce::Exact;
        let precon_ok = matches!(precon, Preconditioner::None | Preconditioner::Diagonal);
        reduction_ok && precon_ok
    }

    /// Switch the flag vote. It is a collective, so every rank must have it
    /// switched the same way.
    pub fn set_flag_vote(&mut self, on: bool) {
        self.flag_vote = on;
    }

    /// Whether the flag vote is on - on by default in a debug build.
    pub fn flag_vote(&self) -> bool {
        self.flag_vote
    }

    /// Votes taken by this workspace so far.
    pub fn n_flag_votes(&self) -> u64 {
        self.n_flag_votes
    }

    /// The `Comm::allgather` calls this handle has made, `new`'s included.
    /// The flag vote is NOT counted here; it is counted by
    /// [`Self::n_flag_votes`] alone.
    pub fn n_allgathers(&self) -> u64 {
        self.red.n_allgathers()
    }
}

/// Read the sticky convergence flag back, and take the flag vote.
///
/// The read is the parent's `dist_read_flag` - one 4-byte transfer. The vote
/// allgathers ONE byte per rank, this rank's flag, and a rank that sees two
/// different votes refuses by name with every rank's vote in rank order:
/// every rank gathered the same bytes, so every rank refuses alike.
fn read_flag_and_vote(
    gpu: &Gpu,
    comm: &dyn Comm,
    vote: bool,
    n_ranks: usize,
    votes: &mut u64,
    flag: &DevBuf<Label>,
) -> Result<bool> {
    let converged = super::dist_read_flag(gpu, flag)?;
    if vote {
        let mine = [u8::from(converged)];
        let mut all = vec![0u8; n_ranks];
        comm.allgather(&mine, &mut all)?;
        *votes += 1;
        let first = all[0];
        if all.iter().any(|&v| v != first) {
            let list: Vec<String> = all.iter().map(|v| v.to_string()).collect();
            return Err(Error::Config(format!(
                "spmd_pcg: the ranks read different convergence flags: \
                 votes by rank: [{}]",
                list.join(", ")
            )));
        }
    }
    Ok(converged)
}

/// PCG over a decomposed mesh, one part per rank.
///
/// The recurrence is §73's [`super::dist_pcg`] exactly - Hestenes & Stiefel
/// (1952), the same kernels in the same order - with the exchange moved onto
/// [`SpmdHalo`] and the reductions onto [`SpmdReduction`]:
///
/// ```text
/// the same program on every rank r, on its own part:
///   r = b - A psi ;  z = M^-1 r ;  p = z ;  rho = (r,z)
///   repeat
///       SpmdHalo::scalar(p) ;  q = A p          the one exchange
///       alpha  = rho / (p,q)                    SpmdReduction
///       psi   += alpha p ;  r -= alpha q
///       [ sum|r| ; convergence test ; read the flag ; vote ]
///       z      = M^-1 r
///       rho'   = (r,z) ;  beta = rho'/rho ;  p = z + beta p ;  rho = rho'
/// ```
///
/// **Symmetric positive definite only**, and nothing here checks it - the
/// same rule and the same reason as the serial solver. `Diagonal` and `None`
/// precondition; `Dic` and `Dilu` are refused by name until their
/// block-local form on a rank exists.
#[allow(clippy::too_many_arguments)]
pub fn spmd_pcg(
    gpu: &Gpu,
    sol: &SolverKernels,
    comm: &dyn Comm,
    psi: &mut DevBuf<Scalar>,
    sys: &DistSystem,
    halo: &mut SpmdHalo,
    w: &mut SpmdWorkspace,
    ctrl: &SolverControls,
) -> Result<DistPerformance> {
    let mut perf = DistPerformance {
        converged: true,
        ..Default::default()
    };

    // ---- the refusals of SPEC-LIT §138.5's table, rows 1-8 in order, all
    // BEFORE anything is launched or sent: a refusal that only one rank can
    // see would leave the other ranks waiting in a collective -------------
    if gpu.is_capturing() {
        return Err(Error::Config(
            "spmd_pcg: an SPMD solve cannot run inside a CUDA-graph capture: \
             every iteration crosses the host over and over, once in the \
             halo exchange and once per reduction's allgather, and there is \
             nothing to record. Alternative: dist_pcg in one process, whose \
             exchange and gather are device-only (SPEC-LIT 81.3, 138.5)"
                .to_string(),
        ));
    }
    let Some(part) = sys.rank_part() else {
        return Err(Error::Config(
            "spmd_pcg: the system holds every part and names no rank; an \
             SPMD solve needs the one-part system DistSystem::split_one \
             builds (SPEC-LIT 138.5)"
                .to_string(),
        ));
    };
    if comm.size() != w.n_ranks {
        return Err(Error::Config(format!(
            "spmd_pcg: this world has {} ranks, but the workspace was built \
             for {} ranks",
            comm.size(),
            w.n_ranks
        )));
    }
    if comm.rank() != part {
        return Err(Error::Config(format!(
            "spmd_pcg: this handle is rank {}, but the system holds part {}",
            comm.rank(),
            part
        )));
    }
    let n = sys.owned()[0];
    let n_local = sys.local()[0];
    let sys_halo = n_local - n;
    if w.n != n || w.n_local != n_local {
        return Err(Error::Config(format!(
            "spmd_pcg: the workspace was built for a part of {} owned + {} \
             halo cells, the system holds {} owned + {} halo cells",
            w.n,
            w.n_local - w.n,
            n,
            sys_halo
        )));
    }
    if halo.n_cells() != n || halo.n_halo() != sys_halo {
        return Err(Error::Config(format!(
            "spmd_pcg: the halo plan's part is {} owned + {} halo cells, \
             the system holds {} owned + {} halo cells",
            halo.n_cells(),
            halo.n_halo(),
            n,
            sys_halo
        )));
    }
    if psi.len() < n_local {
        return Err(Error::Config(format!(
            "spmd_pcg: the solution buffer holds {} values; it must hold {} \
             - {} owned cells and {} halo cells, because lduAmul reads psi \
             at a cut face's neighbour",
            psi.len(),
            n_local,
            n,
            sys_halo
        )));
    }
    let precon = match ctrl.precon {
        Preconditioner::None | Preconditioner::Diagonal => ctrl.precon,
        Preconditioner::Dic | Preconditioner::Dilu => crate::io::contract::unsupported(
            "solvers/<var>/preconditioner",
            ctrl.precon.name(),
            &["none", "diagonal"],
            "diagonal (Jacobi); the block-local DIC/DILU on a rank is not \
             built yet (PAR-04b)",
            Preconditioner::Diagonal,
        )?,
    };

    if sys.n_global_cells() == 0 {
        return Ok(perf);
    }
    perf.converged = false;

    // ---- dist_build_preconditioner for ONE part -------------------------
    // `Dic`/`Dilu` cannot reach here: row 8 refused them, or permissive mode
    // substituted `Diagonal` - there is no part-local colouring to factorise
    // with until the block-local form exists.
    if precon == Preconditioner::Diagonal && n > 0 {
        let nl = solver::to_label(n)?;
        unsafe {
            gpu.stream()
                .launch_builder(&sol.invert_diag)
                .arg(&mut w.r_diag)
                .arg(&sys.a[0].diag)
                .arg(&nl)
                .launch(cfg_for(n))?;
        }
    }

    // The vectors, borrowed apart once - the parent's idiom. The fields the
    // destructure does not name stay reachable through `w`.
    let w = &mut *w;
    let SpmdWorkspace {
        r,
        p,
        v,
        p_hat,
        apsi,
        tmp,
        y,
        b,
        r_diag,
        red,
        rho,
        alpha,
        beta,
        num,
        den,
        x_ref,
        norm_factor,
        initial_res,
        final_res,
        flag,
        ..
    } = w;
    let reduce = w.reduce;

    // ---- dist_norm_factor for ONE part ----------------------------------
    solver::vec_copy(gpu, sol, b, &sys.a[0].source, n)?;
    reduce_into(gpu, sol, comm, red, reduce, What::Sum(psi), x_ref, &mut perf)?;

    // The mean divides by the WHOLE mesh's cell count, which the system
    // carries from the decomposition - not by this rank's `n`.
    let inv_n = 1.0 / (sys.n_global_cells() as Scalar);
    if n > 0 {
        let nl = solver::to_label(n)?;
        unsafe {
            gpu.stream()
                .launch_builder(&sol.broadcast_scaled)
                .arg(&mut *tmp)
                .arg(&*x_ref)
                .arg(&inv_n)
                .arg(&nl)
                .launch(cfg_for(n))?;
        }
    }
    spmd_amul(gpu, sol, comm, halo, sys, y, tmp, &mut perf)?;
    spmd_amul(gpu, sol, comm, halo, sys, apsi, psi, &mut perf)?;
    reduce_into(
        gpu,
        sol,
        comm,
        red,
        reduce,
        What::NormFactor(apsi, b, y, solver::NORM_EPS),
        norm_factor,
        &mut perf,
    )?;

    // ---- dist_residual, and the first reductions ------------------------
    solver::vec_sub(gpu, sol, r, b, apsi, n)?;
    reduce_into(
        gpu,
        sol,
        comm,
        red,
        reduce,
        What::SumMag(r),
        initial_res,
        &mut perf,
    )?;
    solver::copy_scalar(gpu, sol, final_res, initial_res)?;
    gpu.fill_zero(flag)?;

    // z = M^-1 r ; p = z ; rho = (r,z)
    solver::precondition_parts(
        gpu,
        sol,
        p_hat,
        r,
        r_diag,
        None,
        &sys.a[0],
        &sys.mesh[0],
        precon,
        n,
    )?;
    solver::vec_copy(gpu, sol, p, p_hat, n)?;
    reduce_into(gpu, sol, comm, red, reduce, What::Dot(r, p_hat), rho, &mut perf)?;

    let max_iter = ctrl.max_iter.max(0) as usize;
    let interval = ctrl.check_interval.max(1) as usize;
    let checking = !ctrl.fixed_iters;

    if checking {
        solver::convergence_test(
            gpu,
            sol,
            flag,
            initial_res,
            initial_res,
            norm_factor,
            ctrl,
            0,
        )?;
        perf.converged = read_flag_and_vote(
            gpu,
            comm,
            w.flag_vote,
            w.n_ranks,
            &mut w.n_flag_votes,
            flag,
        )?;
    }

    if !perf.converged {
        for it in 0..max_iter {
            let iters = it + 1;

            spmd_amul(gpu, sol, comm, halo, sys, v, p, &mut perf)?;
            reduce_into(gpu, sol, comm, red, reduce, What::Dot(p, v), den, &mut perf)?;
            solver::divide_scalar(gpu, sol, alpha, rho, den)?;

            if n > 0 {
                let nl = solver::to_label(n)?;
                unsafe {
                    gpu.stream()
                        .launch_builder(&sol.axpy)
                        .arg(&mut *psi)
                        .arg(&*p)
                        .arg(&*alpha)
                        .arg(&nl)
                        .launch(cfg_for(n))?;
                    gpu.stream()
                        .launch_builder(&sol.axmy)
                        .arg(&mut *r)
                        .arg(&*v)
                        .arg(&*alpha)
                        .arg(&nl)
                        .launch(cfg_for(n))?;
                }
            }

            perf.n_iterations = iters;

            if checking && iters % interval == 0 {
                reduce_into(
                    gpu,
                    sol,
                    comm,
                    red,
                    reduce,
                    What::SumMag(r),
                    final_res,
                    &mut perf,
                )?;
                let itl = solver::to_label(iters)?;
                solver::convergence_test(
                    gpu,
                    sol,
                    flag,
                    final_res,
                    initial_res,
                    norm_factor,
                    ctrl,
                    itl,
                )?;
                if read_flag_and_vote(
                    gpu,
                    comm,
                    w.flag_vote,
                    w.n_ranks,
                    &mut w.n_flag_votes,
                    flag,
                )? {
                    perf.converged = true;
                    break;
                }
            }

            solver::precondition_parts(
                gpu,
                sol,
                p_hat,
                r,
                r_diag,
                None,
                &sys.a[0],
                &sys.mesh[0],
                precon,
                n,
            )?;
            reduce_into(gpu, sol, comm, red, reduce, What::Dot(r, p_hat), num, &mut perf)?;
            solver::divide_scalar(gpu, sol, beta, num, rho)?;
            solver::copy_scalar(gpu, sol, rho, num)?;

            if n > 0 {
                let nl = solver::to_label(n)?;
                unsafe {
                    gpu.stream()
                        .launch_builder(&sol.p_update_cg)
                        .arg(&mut *p)
                        .arg(&*p_hat)
                        .arg(&*beta)
                        .arg(&nl)
                        .launch(cfg_for(n))?;
                }
            }
        }
    }

    // ---- dist_finish for ONE part: the true residual, recomputed rather
    // than taken from the recurrence, and the report brought back ---------
    if ctrl.report_residuals {
        spmd_amul(gpu, sol, comm, halo, sys, apsi, psi, &mut perf)?;
        solver::vec_sub(gpu, sol, tmp, b, apsi, n)?;
        reduce_into(
            gpu,
            sol,
            comm,
            red,
            reduce,
            What::SumMag(tmp),
            final_res,
            &mut perf,
        )?;

        gpu.sync()?;
        perf.initial_residual = gpu.download(initial_res)?[0];
        perf.final_residual = gpu.download(final_res)?[0];
        let nf = gpu.download(norm_factor)?[0];
        if nf > 0.0 {
            perf.initial_residual /= nf;
            perf.final_residual /= nf;
        }
        if ctrl.fixed_iters {
            let abs = perf.final_residual <= ctrl.tolerance;
            let rel = ctrl.rel_tol > 0.0
                && perf.final_residual <= ctrl.rel_tol * perf.initial_residual;
            perf.converged = abs || rel;
        }
    }
    Ok(perf)
}
