// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! The SPMD PCG and PBiCGStab, and the block-local DIC and DILU on a rank:
//! §73's `dist_pcg` and `dist_pbicgstab`, one part per rank (SPEC-LIT
//! §138.5, §138.6).
//!
//! [`dist_pcg`](super::dist_pcg) and [`dist_pbicgstab`](super::dist_pbicgstab)
//! hold every part in one process. In an SPMD run a rank holds ONE part and
//! reaches the others only through a [`Comm`](crate::comm::Comm), so the
//! solve is the same program on every rank with §73's cross-part steps moved
//! onto the two transports: the matrix product's halo fill goes through
//! [`SpmdHalo::scalar`](crate::halo::SpmdHalo::scalar) (§138.3) and every
//! scalar reduction goes through the [`SpmdReduction`] method of the same
//! name (§138.4). Both transports are bit for bit the in-process forms, and
//! everything between them - the kernels, their arguments, their launch
//! geometry and their order - is the in-process solver's for one part,
//! unchanged. So every rank's field is bit for bit what the in-process
//! solver produces for its part, and, the mean still dividing by the whole
//! mesh's cell count, the ranks all stop on the same iteration.
//!
//! Every rank decides alone and every rank decides alike: the residual sum
//! and the normalisation factor are global values with the same bits on
//! every rank, and the convergence test is one kernel on those values. The
//! **flag vote** checks that instead of trusting it: after each read of the
//! convergence flag the ranks allgather one byte each, and a rank that sees
//! two different votes refuses by name - naming the solver that took the
//! vote. It is on by default in a debug build; a `fixed_iters` solve reads
//! no flag and takes none.
//!
//! The **block-local factorisation** is per rank too: [`SpmdWorkspace::colour`]
//! builds `MultiColour::new` on the rank's own part mesh - what
//! [`DistWorkspace::colour`](super::DistWorkspace::colour) builds for that
//! part - and the factorisation and its two sweeps then walk the part's own
//! faces only, so a cut face is absent from them (§73.5). It is still not
//! partition-invariant, and [`SpmdWorkspace::partition_invariant`] says so.
//!
//! The **distributed GAMG** of SPEC-LIT §141.11 (AMG-07b) is attached by
//! [`SpmdWorkspace::set_gamg`]: an `amg::DistGamg`, this rank's part of a
//! block-local hierarchy with its coarsest level solved on every rank. It is
//! `Preconditioner::Gamg` of both solvers and, through [`spmd_gamg`],
//! `LinearSolverKind::Gamg` of [`spmd_solve`]. A cut changes the hierarchy,
//! so it is not partition-invariant either.
//!
//! Every exchange and every collective is a host round trip, so
//! [`spmd_pcg`], [`spmd_pbicgstab`] and [`spmd_solve`] inside `Gpu::capture`
//! are refused by name (§81.3).
//!
//! Provenance: ORIGINAL - the SPMD form is this project's own design
//! (SPEC-LIT §138.5, PAR-04a; §138.6, PAR-04b); it runs §73's kernels and
//! recurrences and §21's factorisation unchanged, one part per rank, through
//! the transports of §138.3 and §138.4. No external source. No GPL-licensed
//! source was consulted.

use cudarc::driver::PushKernelArg;

use crate::amg::DistGamg;
use crate::comm::Comm;
use crate::device::{cfg_for, DevBuf, Gpu};
use crate::error::{Error, Result};
use crate::exactsum::SpmdReduction;
use crate::halo::SpmdHalo;
use crate::precon::MultiColour;
use crate::solver::{self, LinearSolverKind, Preconditioner, SolverControls, SolverKernels};
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
/// rank's Krylov vectors, its [`SpmdReduction`], the flag vote and, once
/// [`Self::colour`] has run, the part-local colouring both solvers'
/// incomplete factorisations sweep through.
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
    /// `1/diag` (Jacobi), and the factorisation's diagonal once `Dic`/`Dilu`
    /// has been built.
    r_diag: DevBuf<Scalar>,

    // ---- the ones PBiCGStab adds, each `n_local` long --------------------
    /// The shadow residual `r0` the recurrences dot against.
    r0: DevBuf<Scalar>,
    /// `s = r - alpha v`, and its preconditioned form `s_hat`.
    s: DevBuf<Scalar>,
    s_hat: DevBuf<Scalar>,
    /// `t = A s_hat`, the second product's result.
    t: DevBuf<Scalar>,

    // ---- the cross-rank reduction ----------------------------------------
    red: SpmdReduction,

    // ---- the part-local colouring, `None` until `colour` -----------------
    multicolour: Option<MultiColour>,

    // ---- the distributed GAMG cycle, `None` until `set_gamg` (§141.11) ---
    gamg: Option<DistGamg>,

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
    /// PBiCGStab's `rho_old` and `omega`, both starting at 1.
    rho_old: DevBuf<Scalar>,
    omega: DevBuf<Scalar>,

    /// SPEC-LIT §118.12: the refinement's state for this rank's part,
    /// `single` only. `None` only while a refined solve has it out.
    #[cfg(feature = "single")]
    pub(super) ir: Option<super::refine::DistIr>,
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
            r0: gpu.zeros(local)?,
            s: gpu.zeros(local)?,
            s_hat: gpu.zeros(local)?,
            t: gpu.zeros(local)?,
            multicolour: None,
            gamg: None,
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
            rho_old: gpu.zeros(1)?,
            omega: gpu.zeros(1)?,
            #[cfg(feature = "single")]
            ir: Some(super::refine::DistIr::new(gpu, &[owned], &[local])?),
        })
    }

    /// The double solution of the last REFINED solve on this rank's part,
    /// `n_local` long - the owned cells are the answer (SPEC-LIT §118.12).
    /// `single` only.
    #[cfg(feature = "single")]
    pub fn refined_solution(&self) -> Option<&DevBuf<f64>> {
        self.ir.as_ref().and_then(|i| i.x_hi().first())
    }

    /// Outer refinement steps of the LAST solve; 0 for a solve that was
    /// not refined (SPEC-LIT §118.12). `single` only.
    #[cfg(feature = "single")]
    pub fn ir_outer_steps(&self) -> usize {
        self.ir.as_ref().map(|i| i.outer_steps).unwrap_or(0)
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
    /// answer, reached the same way. `Gamg` is `false`: the distributed
    /// cycle aggregates block-locally, so every cut is another hierarchy
    /// (SPEC-LIT §141.11).
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

    /// Colour this rank's own part, so `Dic` and `Dilu` become available to
    /// both solvers (SPEC-LIT §138.6).
    ///
    /// The colouring is [`MultiColour::new`] on the part's mesh - what
    /// [`super::DistWorkspace::colour`] builds for this part - so it is the
    /// same colouring, greedy in the part's local cell order. **It is
    /// local**: it sends nothing, and every rank must run it or the ranks
    /// that did are refused by the ones that did not, and after any `Err`
    /// the world is spent.
    pub fn colour(&mut self, gpu: &Gpu, sys: &DistSystem) -> Result<()> {
        let Some(part) = sys.rank_part() else {
            return Err(Error::Config(
                "distsolve: colouring needs the one-part system \
                 DistSystem::split_one builds; this system holds every part, \
                 which DistSystem::split built (SPEC-LIT 138.6)"
                    .to_string(),
            ));
        };
        if part != self.rank {
            return Err(Error::Config(format!(
                "distsolve: this workspace is rank {}'s, but the system \
                 holds part {}",
                self.rank, part
            )));
        }
        let owned = sys.owned()[0];
        let local = sys.local()[0];
        if self.n != owned || self.n_local != local {
            return Err(Error::Config(format!(
                "distsolve: the workspace was built for a part of {} owned \
                 + {} halo cells, the system holds {} owned + {} halo cells",
                self.n,
                self.n_local - self.n,
                owned,
                local - owned
            )));
        }
        self.multicolour = Some(MultiColour::new(gpu, &sys.mesh[0])?);
        Ok(())
    }

    /// Colours on this rank's part, or 0 before [`Self::colour`].
    pub fn colours(&self) -> usize {
        self.multicolour.as_ref().map_or(0, |m| m.n_colours())
    }

    /// Attach this rank's distributed GAMG cycle (SPEC-LIT §141.11), so
    /// `Preconditioner::Gamg` becomes available to both solvers and
    /// `LinearSolverKind::Gamg` to [`spmd_solve`]. It was built for this
    /// rank, this world size and this part's owned and halo cells, or it is
    /// refused by name. Local: it sends nothing, and every rank must attach
    /// one, because the cycle is a collective.
    pub fn set_gamg(&mut self, g: DistGamg) -> Result<()> {
        if g.rank() != self.rank || g.n_ranks() != self.n_ranks {
            return Err(Error::Config(format!(
                "distsolve: the GAMG cycle was built on rank {} of {}, this \
                 workspace is rank {} of {}",
                g.rank(),
                g.n_ranks(),
                self.rank,
                self.n_ranks
            )));
        }
        if g.n_fine() != self.n || g.n_fine_local() != self.n_local {
            return Err(Error::Config(format!(
                "distsolve: the GAMG cycle was built for a part of {} owned + \
                 {} halo cells, this workspace for {} owned + {} halo cells",
                g.n_fine(),
                g.n_fine_local() - g.n_fine(),
                self.n,
                self.n_local - self.n
            )));
        }
        self.gamg = Some(g);
        Ok(())
    }

    /// The attached distributed GAMG cycle, if any.
    pub fn gamg(&self) -> Option<&DistGamg> {
        self.gamg.as_ref()
    }

    /// Which preconditioner a solve will really run, under the SPEC-LIT
    /// §13.4 rule: supported, or refused by name with the alternative.
    ///
    /// [`super::DistWorkspace::effective_preconditioner`]'s answer for one
    /// rank: `Dic` and `Dilu` need [`Self::colour`] first.
    pub fn effective_preconditioner(&self, requested: Preconditioner) -> Result<Preconditioner> {
        match requested {
            Preconditioner::None | Preconditioner::Diagonal => Ok(requested),
            Preconditioner::Dic | Preconditioner::Dilu => {
                if self.multicolour.is_some() {
                    Ok(requested)
                } else {
                    crate::io::contract::unsupported(
                        "solvers/<var>/preconditioner",
                        requested.name(),
                        &["none", "diagonal"],
                        "diagonal (Jacobi): this SPMD workspace has not been \
                         coloured, so there is no part-local LDU graph to \
                         factorise. Call SpmdWorkspace::colour first",
                        Preconditioner::Diagonal,
                    )
                }
            }
            Preconditioner::Gamg => {
                if self.gamg.is_some() {
                    Ok(requested)
                } else {
                    crate::io::contract::unsupported(
                        "solvers/<var>/preconditioner",
                        requested.name(),
                        &["none", "diagonal"],
                        "diagonal (Jacobi): this SPMD workspace has no \
                         distributed GAMG cycle attached; build one with \
                         amg::DistGamg::new and attach it with \
                         SpmdWorkspace::set_gamg (SPEC-LIT 141.11)",
                        Preconditioner::Diagonal,
                    )
                }
            }
        }
    }
}

/// Read the sticky convergence flag back, and take the flag vote.
///
/// The read is the parent's `dist_read_flag` - one 4-byte transfer. The vote
/// allgathers ONE byte per rank, this rank's flag, and a rank that sees two
/// different votes refuses by name with every rank's vote in rank order:
/// every rank gathered the same bytes, so every rank refuses alike. `who` is
/// the name of the solver that took the vote, which the refusal starts with.
fn read_flag_and_vote(
    gpu: &Gpu,
    comm: &dyn Comm,
    vote: bool,
    n_ranks: usize,
    votes: &mut u64,
    flag: &DevBuf<Label>,
    who: &str,
) -> Result<bool> {
    let converged = super::dist_read_flag(gpu, flag)?;
    if vote {
        take_vote(comm, n_ranks, votes, converged, who)?;
    }
    Ok(converged)
}

/// The vote itself: ONE byte per rank allgathered, and a refusal by name,
/// with every rank's vote in rank order, when two differ.
fn take_vote(comm: &dyn Comm, n_ranks: usize, votes: &mut u64, value: bool, who: &str) -> Result<()> {
    let mine = [u8::from(value)];
    let mut all = vec![0u8; n_ranks];
    comm.allgather(&mine, &mut all)?;
    *votes += 1;
    let first = all[0];
    if all.iter().any(|&v| v != first) {
        let list: Vec<String> = all.iter().map(|v| v.to_string()).collect();
        return Err(Error::Config(format!(
            "{who}: the ranks read different convergence flags: \
             votes by rank: [{}]",
            list.join(", ")
        )));
    }
    Ok(())
}

/// The refinement's outer verdict (SPEC-LIT §118.12), voted like a flag
/// read when the vote is on: `stop` is this rank's "the outer loop ends
/// here". Counted in [`SpmdWorkspace::n_flag_votes`].
#[cfg(feature = "single")]
pub(super) fn vote_verdict(comm: &dyn Comm, w: &mut SpmdWorkspace, stop: bool, who: &str) -> Result<()> {
    if !w.flag_vote {
        return Ok(());
    }
    take_vote(comm, w.n_ranks, &mut w.n_flag_votes, stop, who)
}

/// The refusals of SPEC-LIT §138.5's table, rows 1-8 in order, all BEFORE
/// anything is launched or sent: a refusal that only one rank can see would
/// leave the other ranks waiting in a collective. Both SPMD solvers check
/// the same rows here and differ in the prefix and the named alternative
/// alone - `who` is the solver's name, `alt` the in-process solver to use
/// instead. Returns the part's owned-cell count and the preconditioner the
/// solve will really run.
#[allow(clippy::too_many_arguments)]
fn preflight(
    gpu: &Gpu,
    comm: &dyn Comm,
    psi: &DevBuf<Scalar>,
    sys: &DistSystem,
    halo: &SpmdHalo,
    w: &SpmdWorkspace,
    ctrl: &SolverControls,
    who: &str,
    alt: &str,
) -> Result<(usize, Preconditioner)> {
    if gpu.is_capturing() {
        return Err(Error::Config(format!(
            "{who}: an SPMD solve cannot run inside a CUDA-graph capture: \
             every iteration crosses the host over and over, once in the \
             halo exchange and once per reduction's allgather, and there is \
             nothing to record. Alternative: {alt} in one process, whose \
             exchange and gather are device-only (SPEC-LIT 81.3, 138.5)"
        )));
    }
    let Some(part) = sys.rank_part() else {
        return Err(Error::Config(format!(
            "{who}: the system holds every part and names no rank; an \
             SPMD solve needs the one-part system DistSystem::split_one \
             builds (SPEC-LIT 138.5)"
        )));
    };
    if comm.size() != w.n_ranks {
        return Err(Error::Config(format!(
            "{who}: this world has {} ranks, but the workspace was built \
             for {} ranks",
            comm.size(),
            w.n_ranks
        )));
    }
    if comm.rank() != part {
        return Err(Error::Config(format!(
            "{who}: this handle is rank {}, but the system holds part {}",
            comm.rank(),
            part
        )));
    }
    let n = sys.owned()[0];
    let n_local = sys.local()[0];
    let sys_halo = n_local - n;
    if w.n != n || w.n_local != n_local {
        return Err(Error::Config(format!(
            "{who}: the workspace was built for a part of {} owned + {} \
             halo cells, the system holds {} owned + {} halo cells",
            w.n,
            w.n_local - w.n,
            n,
            sys_halo
        )));
    }
    if halo.n_cells() != n || halo.n_halo() != sys_halo {
        return Err(Error::Config(format!(
            "{who}: the halo plan's part is {} owned + {} halo cells, \
             the system holds {} owned + {} halo cells",
            halo.n_cells(),
            halo.n_halo(),
            n,
            sys_halo
        )));
    }
    if psi.len() < n_local {
        return Err(Error::Config(format!(
            "{who}: the solution buffer holds {} values; it must hold {} \
             - {} owned cells and {} halo cells, because lduAmul reads psi \
             at a cut face's neighbour",
            psi.len(),
            n_local,
            n,
            sys_halo
        )));
    }
    let precon = w.effective_preconditioner(ctrl.precon)?;
    Ok((n, precon))
}

/// `dst = M^-1 src` on this rank's part: the distributed GAMG cycle for
/// `Gamg` (SPEC-LIT §141.11), the parent's `dist_precondition` for ONE part
/// - `solver::precondition_parts` unchanged - for every other kind.
#[allow(clippy::too_many_arguments)]
fn apply_preconditioner(
    gpu: &Gpu,
    sol: &SolverKernels,
    comm: &dyn Comm,
    gamg: &mut Option<DistGamg>,
    dst: &mut DevBuf<Scalar>,
    src: &DevBuf<Scalar>,
    r_diag: &DevBuf<Scalar>,
    multicolour: Option<&MultiColour>,
    sys: &DistSystem,
    precon: Preconditioner,
    n: usize,
) -> Result<()> {
    if precon == Preconditioner::Gamg {
        let Some(g) = gamg.as_mut() else {
            return Err(Error::Config(
                "distsolve: GAMG was selected without a distributed cycle; \
                 SpmdWorkspace::effective_preconditioner should have rejected \
                 this already"
                    .to_string(),
            ));
        };
        return g.precondition(gpu, comm, &sys.mesh[0], &sys.a[0], dst, src);
    }
    solver::precondition_parts(
        gpu,
        sol,
        dst,
        src,
        r_diag,
        multicolour,
        &sys.a[0],
        &sys.mesh[0],
        precon,
        n,
    )
}

/// The cycle's halo exchanges so far, 0 without a cycle.
fn gamg_exchanges(w: &SpmdWorkspace) -> u64 {
    w.gamg.as_ref().map_or(0, |g| g.n_exchanges())
}

/// Build the preconditioner on this rank's part - the parent's
/// `dist_build_preconditioner` for ONE part, the same kernels in the same
/// order: nothing for `None`, `invert_diag` for `Diagonal`, the part's own
/// colour-by-colour factorisation for `Dic`/`Dilu`.
fn build_preconditioner(
    gpu: &Gpu,
    sol: &SolverKernels,
    sys: &DistSystem,
    w: &mut SpmdWorkspace,
    precon: Preconditioner,
) -> Result<()> {
    if precon == Preconditioner::None {
        return Ok(());
    }
    let n = w.n;
    if n == 0 {
        return Ok(());
    }
    if precon == Preconditioner::Diagonal {
        let nl = solver::to_label(n)?;
        unsafe {
            gpu.stream()
                .launch_builder(&sol.invert_diag)
                .arg(&mut w.r_diag)
                .arg(&sys.a[0].diag)
                .arg(&nl)
                .launch(cfg_for(n))?;
        }
        return Ok(());
    }
    // GAMG launches nothing here: the cycle was built and uploaded when
    // `DistGamg::new` ran, and `set_gamg` checked its shape (§141.11).
    if precon == Preconditioner::Gamg {
        return Ok(());
    }
    let SpmdWorkspace { r_diag, multicolour, .. } = w;
    let Some(mc) = multicolour.as_ref() else {
        return Err(Error::Config(
            "distsolve: DIC/DILU was selected without a colouring; \
             SpmdWorkspace::effective_preconditioner should have rejected \
             this already"
                .to_string(),
        ));
    };
    mc.factorise(
        gpu,
        r_diag,
        &sys.a[0],
        &sys.mesh[0],
        precon == Preconditioner::Dic,
    )
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
/// same rule and the same reason as the serial solver. `Diagonal`, `None`
/// and, once the workspace is coloured, the block-local `Dic` and `Dilu`
/// precondition, and, once a cycle is attached, the distributed GAMG with a
/// symmetric cycle (`n_pre == n_post`, §141.11).
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
    // Under `single` the solve is refined (SPEC-LIT §118.12), every refusal
    // of the preflight first - before the refinement's first collective.
    #[cfg(feature = "single")]
    {
        if let Some(i) = w.ir.as_mut() {
            i.outer_steps = 0;
        }
        if super::refine::refines(ctrl, sys.n_global_cells()) {
            preflight(gpu, comm, psi, sys, halo, w, ctrl, "spmd_pcg", "dist_pcg")?;
            return super::refine::spmd_refined(
                gpu,
                sol,
                comm,
                psi,
                sys,
                halo,
                w,
                ctrl,
                super::refine::Method::Pcg,
            );
        }
    }
    spmd_pcg_float(gpu, sol, comm, psi, None, sys, halo, w, ctrl)
}

/// [`spmd_pcg`]'s float body against `rhs` (`None`: the part's `a.source`, the
/// f64 path) - unchanged otherwise.
#[allow(clippy::too_many_arguments)]
pub(super) fn spmd_pcg_float(
    gpu: &Gpu,
    sol: &SolverKernels,
    comm: &dyn Comm,
    psi: &mut DevBuf<Scalar>,
    rhs: Option<&DevBuf<Scalar>>,
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
    let (n, precon) = preflight(gpu, comm, psi, sys, halo, w, ctrl, "spmd_pcg", "dist_pcg")?;
    // A V-cycle with nPreSweeps != nPostSweeps is not a symmetric operator,
    // and CG needs one - refused before any launch, as `solve_pcg` does
    // (SPEC-LIT 141.5, 141.11). Every rank was built from the same config.
    if precon == Preconditioner::Gamg {
        if let Some(g) = w.gamg.as_ref().filter(|g| !g.is_symmetric_cycle()) {
            return Err(Error::Config(format!(
                "spmd_pcg: GAMG preconditioning with nPreSweeps {} and \
                 nPostSweeps {} is not a symmetric operator (the cycle is \
                 symmetric only when they are equal), and conjugate gradients \
                 need a symmetric one; make the cycle symmetric or use \
                 spmd_pbicgstab",
                g.config().n_pre,
                g.config().n_post
            )));
        }
    }

    if sys.n_global_cells() == 0 {
        return Ok(perf);
    }
    perf.converged = false;
    let ex0 = gamg_exchanges(w);

    // ---- dist_build_preconditioner for ONE part -------------------------
    // `Dic`/`Dilu` reach here only on a coloured workspace: row 8 refused
    // them otherwise, or permissive mode substituted `Diagonal`.
    build_preconditioner(gpu, sol, sys, w, precon)?;

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
    solver::vec_copy(gpu, sol, b, rhs.unwrap_or(&sys.a[0].source), n)?;
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

    // z = M^-1 r ; p = z ; rho = (r,z) - `dist_precondition` for ONE part,
    // the part's own colouring when `Dic`/`Dilu`, `None` otherwise
    apply_preconditioner(
        gpu,
        sol,
        comm,
        &mut w.gamg,
        p_hat,
        r,
        r_diag,
        w.multicolour.as_ref(),
        sys,
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
            "spmd_pcg",
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
                    "spmd_pcg",
                )? {
                    perf.converged = true;
                    break;
                }
            }

            apply_preconditioner(
                gpu,
                sol,
                comm,
                &mut w.gamg,
                p_hat,
                r,
                r_diag,
                w.multicolour.as_ref(),
                sys,
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
    // The cycle's level products are matrix products too (§141.11).
    perf.n_exchanges += (gamg_exchanges(w) - ex0) as usize;
    Ok(perf)
}

// ==========================================================================
//  PBiCGStab, one part per rank - SPEC-LIT §138.6
// ==========================================================================

/// PBiCGStab over a decomposed mesh, one part per rank.
///
/// The recurrence is §73's [`super::dist_pbicgstab`] exactly - van der Vorst
/// (1992), the same kernels in the same order, `(t,s)` and `(t,t)` still two
/// reductions - with the two exchanges moved onto [`SpmdHalo`] and the four
/// reductions per iteration onto [`SpmdReduction`]:
///
/// ```text
/// the same program on every rank r, on its own part:
///   r = b - A psi ;  r0 = r ;  p = 0 ;  v = 0 ;  rho_old = alpha = omega = 1
///   repeat
///       rho   = (r0,r)                            SpmdReduction
///       beta  = (rho/rho_old)(alpha/omega)
///       p     = r + beta (p - omega v)
///       pHat  = M^-1 p
///       SpmdHalo::scalar(pHat) ;  v = A pHat      exchange 1
///       alpha = rho / (r0,v)                      SpmdReduction
///       s     = r - alpha v
///       sHat  = M^-1 s
///       SpmdHalo::scalar(sHat) ;  t = A sHat      exchange 2
///       omega = (t,s) / (t,t)                     two SpmdReductions
///       psi  += alpha pHat + omega sHat ;  r = s - omega t ;  rho_old = rho
///       [ sum|r| ; convergence test ; read the flag ; vote ]
/// ```
///
/// Every launch is `dist_pbicgstab`'s for one part (§138.5's rule), so every
/// rank's field is bit for bit `dist_pbicgstab`'s field for its part, and
/// with the exact reduction and `Diagonal` or `None` also the one-part field.
/// `Diagonal`, `None`, on a coloured workspace the block-local `Dic` and
/// `Dilu`, and with a cycle attached the distributed GAMG (§141.11)
/// precondition. The flag vote's refusal starts `spmd_pbicgstab:
/// `, naming the solver that took the vote.
#[allow(clippy::too_many_arguments)]
pub fn spmd_pbicgstab(
    gpu: &Gpu,
    sol: &SolverKernels,
    comm: &dyn Comm,
    psi: &mut DevBuf<Scalar>,
    sys: &DistSystem,
    halo: &mut SpmdHalo,
    w: &mut SpmdWorkspace,
    ctrl: &SolverControls,
) -> Result<DistPerformance> {
    // Under `single` the solve is refined (SPEC-LIT §118.12), every refusal
    // of the preflight first - before the refinement's first collective.
    #[cfg(feature = "single")]
    {
        if let Some(i) = w.ir.as_mut() {
            i.outer_steps = 0;
        }
        if super::refine::refines(ctrl, sys.n_global_cells()) {
            preflight(gpu, comm, psi, sys, halo, w, ctrl, "spmd_pbicgstab", "dist_pbicgstab")?;
            return super::refine::spmd_refined(
                gpu,
                sol,
                comm,
                psi,
                sys,
                halo,
                w,
                ctrl,
                super::refine::Method::Pbicgstab,
            );
        }
    }
    spmd_pbicgstab_float(gpu, sol, comm, psi, None, sys, halo, w, ctrl)
}

/// [`spmd_pbicgstab`]'s float body against `rhs` (`None`: the part's `a.source`, the
/// f64 path) - unchanged otherwise.
#[allow(clippy::too_many_arguments)]
pub(super) fn spmd_pbicgstab_float(
    gpu: &Gpu,
    sol: &SolverKernels,
    comm: &dyn Comm,
    psi: &mut DevBuf<Scalar>,
    rhs: Option<&DevBuf<Scalar>>,
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
    let (n, precon) = preflight(
        gpu,
        comm,
        psi,
        sys,
        halo,
        w,
        ctrl,
        "spmd_pbicgstab",
        "dist_pbicgstab",
    )?;

    if sys.n_global_cells() == 0 {
        return Ok(perf);
    }
    perf.converged = false;
    let ex0 = gamg_exchanges(w);

    // ---- dist_build_preconditioner for ONE part -------------------------
    build_preconditioner(gpu, sol, sys, w, precon)?;

    // The vectors, borrowed apart once - the parent's idiom, the one
    // `spmd_pcg` uses. The fields the destructure does not name (`reduce`,
    // `flag_vote`, `n_flag_votes`, `multicolour`) stay reachable through
    // `w`.
    let w = &mut *w;
    let SpmdWorkspace {
        r,
        r0,
        p,
        v,
        s,
        t,
        p_hat,
        s_hat,
        apsi,
        tmp,
        y,
        b,
        r_diag,
        red,
        rho,
        rho_old,
        alpha,
        beta,
        num,
        den,
        omega,
        x_ref,
        norm_factor,
        initial_res,
        final_res,
        flag,
        ..
    } = w;
    let reduce = w.reduce;

    // ---- dist_norm_factor for ONE part ----------------------------------
    solver::vec_copy(gpu, sol, b, rhs.unwrap_or(&sys.a[0].source), n)?;
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

    // ---- dist_residual, r0 = r, and the first reductions ----------------
    solver::vec_sub(gpu, sol, r, b, apsi, n)?;
    solver::vec_copy(gpu, sol, r0, r, n)?;
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

    // p = 0 ; v = 0 ; rho_old = alpha = omega = 1
    gpu.fill_zero(p)?;
    gpu.fill_zero(v)?;
    gpu.fill_zero(flag)?;
    solver::set_scalar(gpu, sol, rho_old, 1.0)?;
    solver::set_scalar(gpu, sol, alpha, 1.0)?;
    solver::set_scalar(gpu, sol, omega, 1.0)?;

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
            "spmd_pbicgstab",
        )?;
    }

    if !perf.converged {
        for it in 0..max_iter {
            let iters = it + 1;

            // rho = (r0, r) ; beta = (rho/rho_old)(alpha/omega)
            reduce_into(gpu, sol, comm, red, reduce, What::Dot(r0, r), rho, &mut perf)?;
            unsafe {
                gpu.stream()
                    .launch_builder(&sol.beta_bicg)
                    .arg(&mut *beta)
                    .arg(&*rho)
                    .arg(&*rho_old)
                    .arg(&*alpha)
                    .arg(&*omega)
                    .launch(solver::one_thread())?;
            }

            // p = r + beta (p - omega v)
            if n > 0 {
                let nl = solver::to_label(n)?;
                unsafe {
                    gpu.stream()
                        .launch_builder(&sol.p_update)
                        .arg(&mut *p)
                        .arg(&*r)
                        .arg(&*v)
                        .arg(&*beta)
                        .arg(&*omega)
                        .arg(&nl)
                        .launch(cfg_for(n))?;
                }
            }

            // pHat = M^-1 p ; exchange 1 ; v = A pHat ; alpha = rho / (r0,v)
            apply_preconditioner(
                gpu,
                sol,
                comm,
                &mut w.gamg,
                p_hat,
                p,
                r_diag,
                w.multicolour.as_ref(),
                sys,
                precon,
                n,
            )?;
            spmd_amul(gpu, sol, comm, halo, sys, v, p_hat, &mut perf)?;
            reduce_into(gpu, sol, comm, red, reduce, What::Dot(r0, v), den, &mut perf)?;
            solver::divide_scalar(gpu, sol, alpha, rho, den)?;

            // s = r - alpha v
            if n > 0 {
                let nl = solver::to_label(n)?;
                unsafe {
                    gpu.stream()
                        .launch_builder(&sol.s_update)
                        .arg(&mut *s)
                        .arg(&*r)
                        .arg(&*v)
                        .arg(&*alpha)
                        .arg(&nl)
                        .launch(cfg_for(n))?;
                }
            }

            // sHat = M^-1 s ; exchange 2 ; t = A sHat
            apply_preconditioner(
                gpu,
                sol,
                comm,
                &mut w.gamg,
                s_hat,
                s,
                r_diag,
                w.multicolour.as_ref(),
                sys,
                precon,
                n,
            )?;
            spmd_amul(gpu, sol, comm, halo, sys, t, s_hat, &mut perf)?;

            // omega = (t,s) / (t,t) - two reductions, as in-process
            reduce_into(gpu, sol, comm, red, reduce, What::Dot(t, s), num, &mut perf)?;
            reduce_into(gpu, sol, comm, red, reduce, What::Dot(t, t), den, &mut perf)?;
            solver::divide_scalar(gpu, sol, omega, num, den)?;

            // psi += alpha pHat + omega sHat ;  r = s - omega t
            if n > 0 {
                let nl = solver::to_label(n)?;
                unsafe {
                    gpu.stream()
                        .launch_builder(&sol.x_update)
                        .arg(&mut *psi)
                        .arg(&*p_hat)
                        .arg(&*s_hat)
                        .arg(&*alpha)
                        .arg(&*omega)
                        .arg(&nl)
                        .launch(cfg_for(n))?;
                    gpu.stream()
                        .launch_builder(&sol.r_update)
                        .arg(&mut *r)
                        .arg(&*s)
                        .arg(&*t)
                        .arg(&*omega)
                        .arg(&nl)
                        .launch(cfg_for(n))?;
                }
            }
            solver::copy_scalar(gpu, sol, rho_old, rho)?;

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
                    "spmd_pbicgstab",
                )? {
                    perf.converged = true;
                    break;
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
    perf.n_exchanges += (gamg_exchanges(w) - ex0) as usize;
    Ok(perf)
}

// ==========================================================================
//  The stationary distributed GAMG - SPEC-LIT §141.11
// ==========================================================================

/// The distributed V-cycle as a solver on its own (SPEC-LIT §141.11), the
/// SPMD form of `solver::solve_gamg`: `psi <- V(psi, b)` per iteration, one
/// iteration one cycle, with §8.4's residual of `b - A psi` tested every
/// `check_interval` iterations by the kernels and the flag vote of
/// [`spmd_pcg`]. The cycle comes from the workspace
/// ([`SpmdWorkspace::set_gamg`]); a request without one is refused by name,
/// and `ctrl.precon` is ignored - the cycle IS the method here.
#[allow(clippy::too_many_arguments)]
pub fn spmd_gamg(
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
    // `ctrl.precon` is not read: the rows of §138.5's table are checked
    // with `None`, so a preconditioner this workspace cannot build does not
    // refuse a method that never applies one.
    let plain = SolverControls {
        precon: Preconditioner::None,
        ..*ctrl
    };
    let (n, _) = preflight(gpu, comm, psi, sys, halo, w, &plain, "spmd_gamg", "solve_gamg")?;
    if w.gamg.is_none() {
        return Err(Error::Config(
            "spmd_gamg: solver GAMG was requested, but this SPMD workspace \
             has no distributed GAMG cycle attached; build one with \
             amg::DistGamg::new and attach it with SpmdWorkspace::set_gamg \
             (SPEC-LIT 141.11), or solve with PCG or PBiCGStab"
                .to_string(),
        ));
    }
    if sys.n_global_cells() == 0 {
        return Ok(perf);
    }
    perf.converged = false;
    let ex0 = gamg_exchanges(w);

    let w = &mut *w;
    let SpmdWorkspace {
        r,
        apsi,
        tmp,
        y,
        b,
        red,
        x_ref,
        norm_factor,
        initial_res,
        final_res,
        flag,
        gamg,
        ..
    } = w;
    let reduce = w.reduce;

    // ---- dist_norm_factor for ONE part, as `spmd_pcg` -------------------
    solver::vec_copy(gpu, sol, b, &sys.a[0].source, n)?;
    reduce_into(gpu, sol, comm, red, reduce, What::Sum(psi), x_ref, &mut perf)?;
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
            "spmd_gamg",
        )?;
    }

    if !perf.converged {
        let g = gamg.as_mut().ok_or_else(|| {
            Error::Config("spmd_gamg: the GAMG cycle vanished mid-solve".to_string())
        })?;
        for it in 0..max_iter {
            let iters = it + 1;
            // One iteration is one cycle, from the current psi.
            g.cycle(gpu, comm, &sys.mesh[0], &sys.a[0], psi, b)?;
            perf.n_iterations = iters;

            if checking && iters % interval == 0 {
                spmd_amul(gpu, sol, comm, halo, sys, apsi, psi, &mut perf)?;
                solver::vec_sub(gpu, sol, r, b, apsi, n)?;
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
                    "spmd_gamg",
                )? {
                    perf.converged = true;
                    break;
                }
            }
        }
    }

    // ---- dist_finish for ONE part, as `spmd_pcg` ------------------------
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
    perf.n_exchanges += (gamg_exchanges(w) - ex0) as usize;
    Ok(perf)
}

// ==========================================================================
//  Dispatch - SPEC-LIT §13.4 applied to the SPMD path
// ==========================================================================

/// Solve `A psi = b` on this rank's part with the method the case asked for.
///
/// Dispatches on `ctrl.solver` as `solver::solve` does: `PCG` and
/// `PBiCGStab` as [`super::dist_solve`] does, and `GAMG` - since SPEC-LIT
/// §141.11 - as the distributed cycle on its own, [`spmd_gamg`], which
/// needs a cycle attached by [`SpmdWorkspace::set_gamg`] and refuses by name
/// without one.
#[allow(clippy::too_many_arguments)]
pub fn spmd_solve(
    gpu: &Gpu,
    sol: &SolverKernels,
    comm: &dyn Comm,
    psi: &mut DevBuf<Scalar>,
    sys: &DistSystem,
    halo: &mut SpmdHalo,
    w: &mut SpmdWorkspace,
    ctrl: &SolverControls,
) -> Result<DistPerformance> {
    match ctrl.solver {
        LinearSolverKind::PBiCGStab => spmd_pbicgstab(gpu, sol, comm, psi, sys, halo, w, ctrl),
        LinearSolverKind::PCG => spmd_pcg(gpu, sol, comm, psi, sys, halo, w, ctrl),
        LinearSolverKind::Gamg => spmd_gamg(gpu, sol, comm, psi, sys, halo, w, ctrl),
    }
}
