// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! SPEC-LIT 139.2 (PAR-10a): the exchange points of a decomposed field.
//!
//! §139.1 (PAR-09) gave every cell buffer on a part room for the halo and
//! left the room zero. This module fills it. A [`RankCtx`] is what one rank
//! of a parallel flow solve holds besides its part - the transport, the
//! rank's half of the exchange plan, the exact cross-part reduction and the
//! part number - and its four `exchange_*` methods are the places the flow
//! path touches another rank:
//!
//! | call | what it fills |
//! |---|---|
//! | [`RankCtx::exchange_field`] | the ghost cells of a scalar field's `f`, then `field_ops::correct_boundary_conditions`, so a processor face's `bf` is `w psi_P + (1 - w) psi_ghost` with the NEIGHBOUR RANK'S value - what the whole mesh's internal face saw |
//! | [`RankCtx::exchange_field_vector`] | the same for a vector field, through `correct_boundary_conditions_vector` |
//! | [`RankCtx::exchange_gradient`] | the ghost cells of a cell gradient of a scalar, for the operators that read `grad` at the far cell of a face |
//! | [`RankCtx::exchange_gradient_tensor`] | the same for the cell gradient of a vector field, nine components as one `Tensor` |
//!
//! Every exchange is an [`SpmdHalo`] exchange: a pack kernel, device-to-host
//! copies, the bytes handed to the [`Comm`], the receive copied back into
//! the halo, two stream synchronisations. It is therefore a host round trip
//! by construction and **not capturable**: [`SpmdHalo`] refuses inside a
//! CUDA-graph capture by name (SPEC-LIT 81.3, 138.3), so an exchange of
//! this module inside `Gpu::capture` fails with a message starting
//! `halo: `.
//!
//! SPEC-LIT 139.3 (PAR-10b) routes a part's linear solves through the SPMD
//! Krylov of SPEC-LIT 138.5/138.6, which is what makes every residual norm
//! the world's and every rank report the same bits: a part still solved by
//! the serial `solver::solve` would norm its OWN cells only, and two ranks
//! of one SIMPLE iteration would disagree about whether the equation
//! converged. `DistSystem::for_rank` and `load_matrix` hold the
//! part-assembled matrix - assembled on the part, never split from a whole -
//! and of its state only the two constraint arrays cross the halo, because
//! they are the only matrix state a kernel reads at a halo index:
//!
//! | type | what it is |
//! |---|---|
//! | [`SpmdKrylov`] | one rank's system, halo, workspace and kernels behind one `run`: load the part's matrix in, exchange the constraints, `spmd_solve` - and a refusal table that fires before anything collective |
//! | [`SpmdPressureBackend`] | the pressure equation's [`PressureBackend`] over that Krylov, dropped into `Simple`'s existing `&mut dyn PressureBackend` seam |
//! | [`SpmdMomentumRoute`] | the momentum predictor's [`ComponentRoute`](crate::momentum::ComponentRoute), set through `Simple::set_momentum_route`; `None` keeps the old call launch for launch |
//!
//! Each routed solve is a host round trip by construction and therefore NOT
//! capturable: `spmd_solve` refuses inside a CUDA-graph capture by name
//! (SPEC-LIT 81.3, 138.5), and so every solve this module routes fails a
//! capture rather than silently serialising it.
//!
//! Provenance: ORIGINAL - the exchange points are this project's own design
//! (SPEC-LIT 139.2, PAR-10a; 139.3, PAR-10b) over the SPMD exchange of
//! 138.3 (PAR-02), the boundary evaluation this crate already ships and the
//! SPMD Krylov of 138.5/138.6 (PAR-04a/b); no kernel is launched here and
//! no existing numerics change. No external source. No GPL-licensed
//! source was consulted.

use crate::comm::Comm;
use crate::decompose::{Decomposition, PartMesh};
use crate::device::{DevBuf, Gpu};
use crate::distsolve::{spmd_solve, DistPerformance, DistReduce, DistSystem, SpmdWorkspace};
use crate::error::{Error, Result};
use crate::exactsum::SpmdReduction;
use crate::field::{GpuScalarField, GpuVectorField};
use crate::field_ops::{self, FieldKernels};
use crate::halo::SpmdHalo;
use crate::io::case::SolverControls;
use crate::ldu::GpuLduMatrix;
use crate::mesh::GpuMesh;
use crate::momentum::ComponentRoute;
use crate::pressure::{PressureBackend, SystemProbe};
use crate::solver::{
    LinearSolverKind, Preconditioner, SolverKernels, SolverPerformance,
};
use crate::HostMesh;
use crate::{Scalar, Tensor, Vec3};

/// One rank's view of a decomposed flow solve: its part, the transport, and
/// the two objects every cross-part step goes through.
///
/// - `comm` is the transport (`ThreadComm` today, MPI later), borrowed for
///   the life of the world. It moves bytes and does no arithmetic on them
///   (SPEC-LIT 138), so a ghost cell holds its owner's value bit for bit
///   whatever the transport is.
/// - `halo` is the rank's half of 71.4's exchange plan plus 138.3's host
///   round trip, built once here and never rebuilt: an exchange allocates
///   nothing on the device.
/// - `reduce` is the rank's exact cross-part reduction (138.4). The flow
///   path does not reduce yet - PAR-10b's Krylov routing and PAR-10c's
///   SIMPLE take their global norms through it - and it carries state, so
///   it is handed out mutable through [`RankCtx::reduce_mut`].
/// - `part` is the part number, which IS the rank number: row 1 of
///   [`RankCtx::new`]'s refusal table refuses anything else.
pub struct RankCtx<'c> {
    comm: &'c dyn Comm,
    halo: SpmdHalo,
    reduce: SpmdReduction,
    part: usize,
}

impl<'c> RankCtx<'c> {
    /// Build the rank's context. `part` must be the part `comm.rank()`
    /// holds, and `mesh` the upload of `part.mesh` WITH its halo room -
    /// [`GpuMesh::upload_with_halo`](crate::mesh::GpuMesh::upload_with_halo).
    ///
    /// Three refusals, in order, all `Error::Config` starting `distflow: `,
    /// all BEFORE any collective and any device allocation, so the refusing
    /// rank sends nothing. A peer that did not refuse goes on into
    /// `SpmdReduction::new`'s allgather and waits there until the
    /// transport's timeout: after an `Err` the world is spent (SPEC-LIT
    /// 138.1, R10). The checks are local and every rank of a correct
    /// program passes them alike.
    ///
    /// 1. `comm.rank() != part.part` - the handle belongs to another rank;
    /// 2. `mesh.n_cells != part.mesh.n_cells` - the mesh is another part's;
    /// 3. `mesh.n_halo != part.n_halo` - the mesh was uploaded without its
    ///    halo room, which SPEC-LIT 139.1 makes a read past every cell
    ///    buffer the moment a processor face is evaluated.
    pub fn new(gpu: &Gpu, comm: &'c dyn Comm, part: &PartMesh, mesh: &GpuMesh) -> Result<Self> {
        let r = comm.rank();
        if r != part.part {
            return Err(Error::Config(format!(
                "distflow: this handle is rank {r}, but the part given is \
                 part {}; one rank holds one part and the two numbers must \
                 agree",
                part.part
            )));
        }
        if mesh.n_cells != part.mesh.n_cells {
            return Err(Error::Config(format!(
                "distflow: the uploaded mesh has {} cells but part {} has \
                 {}; the mesh given to RankCtx::new must be the upload of \
                 the part's own mesh",
                mesh.n_cells,
                part.part,
                part.mesh.n_cells
            )));
        }
        if mesh.n_halo != part.n_halo {
            return Err(Error::Config(format!(
                "distflow: the uploaded mesh has n_halo {}, but part {} has \
                 n_halo {}; upload the part with GpuMesh::upload_with_halo, \
                 which makes every cell buffer n_cells + n_halo long, or the \
                 processor faces' neighbours read past them (SPEC-LIT 139.1)",
                mesh.n_halo,
                part.part,
                part.n_halo
            )));
        }
        Ok(Self {
            comm,
            part: part.part,
            halo: SpmdHalo::new(gpu, part, comm.size())?,
            reduce: SpmdReduction::new(gpu, comm, part.mesh.n_cells)?,
        })
    }

    /// The transport. Every cross-part call a later unit adds goes through
    /// this one borrowed object, so the flow path holds no transport type.
    pub fn comm(&self) -> &'c dyn Comm {
        self.comm
    }

    /// This rank's number - `comm.rank()`, which row 1 of [`RankCtx::new`]'s
    /// refusal table pins to the part number.
    pub fn rank(&self) -> usize {
        self.part
    }

    /// The rank count the context was built with.
    pub fn n_ranks(&self) -> usize {
        self.comm.size()
    }

    /// The rank's half of the exchange plan: its neighbour list, its send
    /// slices and where its halo lands.
    pub fn halo(&self) -> &SpmdHalo {
        &self.halo
    }

    /// The rank's exact cross-part reduction.
    pub fn reduce(&self) -> &SpmdReduction {
        &self.reduce
    }

    /// The same, mutable: the reduction counts its own collectives, and a
    /// caller that gathers through it moves that count.
    pub fn reduce_mut(&mut self) -> &mut SpmdReduction {
        &mut self.reduce
    }

    /// Fill a scalar field's ghost cells from the neighbour ranks, then
    /// re-evaluate every boundary face.
    ///
    /// The halo exchange moves each neighbour's owned values into this
    /// part's halo `n_cells .. n_cells + n_halo`, bit for bit the owner's.
    /// `field_ops::correct_boundary_conditions` then runs over every
    /// boundary face, so on a processor face - a `Cyclic` face whose
    /// `b_nbr_cell` names a halo cell (SPEC-LIT 139.1) - `bf` becomes
    /// `w psi_P + (1 - w) psi_ghost` with the neighbour rank's value, which
    /// is what the whole mesh's internal face read. The physical patches
    /// re-evaluate from the same values they always had and do not move.
    pub fn exchange_field(
        &mut self,
        gpu: &Gpu,
        k: &FieldKernels,
        f: &mut GpuScalarField,
        m: &GpuMesh,
    ) -> Result<()> {
        self.halo.scalar(gpu, self.comm, &mut f.f)?;
        field_ops::correct_boundary_conditions(gpu, k, f, m)
    }

    /// The same for a vector field, through
    /// [`field_ops::correct_boundary_conditions_vector`].
    pub fn exchange_field_vector(
        &mut self,
        gpu: &Gpu,
        k: &FieldKernels,
        u: &mut GpuVectorField,
        m: &GpuMesh,
    ) -> Result<()> {
        self.halo.vector(gpu, self.comm, &mut u.f)?;
        field_ops::correct_boundary_conditions_vector(gpu, k, u, m)
    }

    /// Fill the halo of a cell gradient of a scalar field -
    /// [`SpmdHalo::vector`](crate::halo::SpmdHalo::vector), a gradient being
    /// three components per cell.
    ///
    /// The operators that read the gradient at the FAR cell of a face - the
    /// deferred corrections and the limited reconstructions - read it on
    /// ghost cells too, so a gradient is exchanged like a field. No boundary
    /// evaluation follows it: a gradient carries no boundary state, and the
    /// cells it fills are read only.
    pub fn exchange_gradient(&mut self, gpu: &Gpu, g: &mut DevBuf<Vec3>) -> Result<()> {
        self.halo.vector(gpu, self.comm, g)
    }

    /// The same for the cell gradient of a vector field - nine components
    /// moved as one [`Tensor`], the exchange plan's third payload.
    pub fn exchange_gradient_tensor(&mut self, gpu: &Gpu, g: &mut DevBuf<Tensor>) -> Result<()> {
        self.halo.tensor(gpu, self.comm, g)
    }
}

// ==========================================================================
//  The Krylov routing - SPEC-LIT 139.3 (PAR-10b)
// ==========================================================================

/// The GAMG refusal [`SpmdKrylov::run`] and [`SpmdPressureBackend::new`]
/// both make, the latter BEFORE the collective `SpmdKrylov::new` performs,
/// so every rank given the same controls refuses alike and no rank is left
/// waiting in a collective (SPEC-LIT 138.1, R10).
fn refuse_gamg(ctrl: &SolverControls) -> Result<()> {
    if ctrl.solver == LinearSolverKind::Gamg || ctrl.precon == Preconditioner::Gamg {
        return Err(Error::Config(format!(
            "distflow: GAMG is refused on a part's flow solve (solver {}, \
             preconditioner {}); a part's flow solve routes PCG and \
             PBiCGStab only - the distributed cycle of SPEC-LIT 141.11 is \
             not attached on the flow path (SPEC-LIT 139.3)",
            ctrl.solver.name(),
            ctrl.precon.name()
        )));
    }
    Ok(())
}

/// A distributed solve's record, mapped onto the [`SolverPerformance`] the
/// flow path carries. The counters `spmd_solve` alone owns - the exchange
/// and reduction counts - stop here: the flow path's record never had them.
fn as_solver_performance(d: DistPerformance) -> SolverPerformance {
    SolverPerformance {
        initial_residual: d.initial_residual,
        final_residual: d.final_residual,
        n_iterations: d.n_iterations,
        converged: d.converged,
    }
}

/// One rank's SPMD Krylov solve, wired for the flow path (SPEC-LIT 139.3,
/// PAR-10b).
///
/// [`DistSystem::for_rank`] builds the part's system - a matrix ASSEMBLED ON
/// THE PART, never split from a whole - and [`SpmdWorkspace`] holds this
/// rank's Krylov vectors, its [`SpmdReduction`] and, once a DIC or DILU
/// preconditioner first asks, the part-local colouring. One [`Self::run`] is
/// then: load the caller's matrix in, exchange the two constraint arrays,
/// and hand the whole thing to [`spmd_solve`], whose every residual norm is
/// a cross-part reduction and so the same bits on every rank.
///
/// The refusals - GAMG, and the two size rows - fire BEFORE anything
/// collective, so every rank given the same input refuses alike and no rank
/// waits for one that refused (SPEC-LIT 138.1, R10).
pub struct SpmdKrylov<'c> {
    comm: &'c dyn Comm,
    halo: SpmdHalo,
    sys: DistSystem,
    ws: SpmdWorkspace,
    kernels: SolverKernels,
    /// Whether the part-local colouring is built; a DIC/DILU solve needs it
    /// once, and building it is a device pass, not a collective.
    coloured: bool,
    n_solves: usize,
    last: Option<DistPerformance>,
}

impl<'c> SpmdKrylov<'c> {
    /// Build the rank's Krylov route: `comm.rank()`'s part of `dec`, its
    /// halo plan, its system, its workspace and the solver kernels.
    ///
    /// Refused (`Error::Config`, starting `distflow: `) when
    /// `comm.size() != dec.n_parts`, naming both numbers, BEFORE anything
    /// collective - [`SpmdWorkspace::new`]'s reduction allgathers, and a
    /// rank that refuses there strands the others.
    pub fn new(
        gpu: &Gpu,
        comm: &'c dyn Comm,
        dec: &Decomposition,
        reduce: DistReduce,
    ) -> Result<Self> {
        if comm.size() != dec.n_parts {
            return Err(Error::Config(format!(
                "distflow: the Krylov route needs one rank per part: the \
                 world has {} ranks, the decomposition has {} parts",
                comm.size(),
                dec.n_parts
            )));
        }
        let rank = comm.rank();
        let halo = SpmdHalo::new(gpu, &dec.parts[rank], comm.size())?;
        let sys = DistSystem::for_rank(gpu, comm, dec)?;
        let ws = SpmdWorkspace::new(gpu, comm, &sys, reduce)?;
        let kernels = SolverKernels::new(gpu)?;
        Ok(Self {
            comm,
            halo,
            sys,
            ws,
            kernels,
            coloured: false,
            n_solves: 0,
            last: None,
        })
    }

    /// This rank's number - `comm.rank()`.
    pub fn rank(&self) -> usize {
        self.comm.rank()
    }

    /// The rank count the route was built with.
    pub fn n_ranks(&self) -> usize {
        self.comm.size()
    }

    /// The part's OWNED cells - what a kernel launches threads for and what
    /// an owned answer is `0..` this long.
    pub fn n_cells(&self) -> usize {
        self.sys.owned()[0]
    }

    /// The rank's SPMD workspace: its vectors, its reduction, its counts.
    pub fn workspace(&self) -> &SpmdWorkspace {
        &self.ws
    }

    /// Turn the workspace's flag vote on or off. A COLLECTIVE setting - call
    /// it alike on every rank, or the vote counts disagree.
    pub fn set_flag_vote(&mut self, on: bool) {
        self.ws.set_flag_vote(on);
    }

    /// How many solves this route has run.
    pub fn n_solves(&self) -> usize {
        self.n_solves
    }

    /// The last solve's distributed record, `None` before the first.
    pub fn last_performance(&self) -> Option<&DistPerformance> {
        self.last.as_ref()
    }

    /// The double solution `x_hi` of the last solve (SPEC-LIT 118.12), which
    /// under `single` is the answer's precision class A form. `single` only.
    #[cfg(feature = "single")]
    pub fn refined_solution(&self) -> Option<&DevBuf<f64>> {
        self.ws.refined_solution()
    }

    /// Solve `a·psi = b` for this rank's part, the norms global.
    ///
    /// In order: refuse GAMG as solver or preconditioner; refuse a mesh or a
    /// solution buffer that is not the part's; build the part-local
    /// colouring the first DIC/DILU solve needs; load the caller's
    /// part-assembled matrix in and exchange the constraints through the
    /// halo; [`spmd_solve`]; count and remember. Every rank given the same
    /// controls sees the same refusals, so the world refuses alike or not at
    /// all.
    pub fn run(
        &mut self,
        gpu: &Gpu,
        psi: &mut DevBuf<Scalar>,
        a: &GpuLduMatrix,
        m: &GpuMesh,
        ctrl: &SolverControls,
    ) -> Result<DistPerformance> {
        // (1) GAMG: the distributed cycle of SPEC-LIT 141.11 is not attached
        // on the flow path.
        refuse_gamg(ctrl)?;
        // (2) The sizes: the mesh must be this rank's part, and psi must be
        // long enough for the halo the solve's matrix product reads.
        if m.n_cells != self.n_cells() {
            return Err(Error::Config(format!(
                "distflow: the mesh given to the Krylov route has {} cells, \
                 but this rank's part owns {}; the route runs on the part's \
                 own mesh",
                m.n_cells,
                self.n_cells()
            )));
        }
        let n_local = self.sys.local()[0];
        if psi.len() < n_local {
            return Err(Error::Config(format!(
                "distflow: psi is {} long, the part's solve needs {} (owned \
                 + halo); upload the part with GpuMesh::upload_with_halo so \
                 every cell buffer carries the halo (SPEC-LIT 139.1)",
                psi.len(),
                n_local
            )));
        }
        // (3) The part-local colouring, once, when the preconditioner sweeps
        // a coloured order.
        if !self.coloured && matches!(ctrl.precon, Preconditioner::Dic | Preconditioner::Dilu) {
            self.ws.colour(gpu, &self.sys)?;
            self.coloured = true;
        }
        // (4) The part-assembled matrix, in; the constraints exchanged.
        self.sys
            .load_matrix(gpu, &mut self.halo, self.comm, a)?;
        // (5) The solve, every residual norm a cross-part reduction.
        let d = spmd_solve(
            gpu,
            &self.kernels,
            self.comm,
            psi,
            &self.sys,
            &mut self.halo,
            &mut self.ws,
            ctrl,
        )?;
        // (6) Counted and remembered.
        self.n_solves += 1;
        self.last = Some(d.clone());
        Ok(d)
    }
}

/// The pressure equation's backend over one rank's [`SpmdKrylov`] (SPEC-LIT
/// 139.3, PAR-10b).
///
/// Dropped into the seam `Simple` already takes - `&mut dyn
/// PressureBackend` - so a decomposed driver changes what the pressure
/// equation is SOLVED BY and nothing about the equation. `setup` checks the
/// meshes are the part's and nothing else: a part's system holds no
/// hierarchy and no factorisation to refresh, and every solve loads the
/// matrix as the caller assembled it.
pub struct SpmdPressureBackend<'c> {
    krylov: SpmdKrylov<'c>,
    ctrl: SolverControls,
}

impl<'c> SpmdPressureBackend<'c> {
    /// Build the backend for `comm.rank()`'s part of `dec`, solving to
    /// `ctrl`.
    ///
    /// The GAMG refusal fires BEFORE [`SpmdKrylov::new`], whose workspace
    /// constructor allgathers: a case that asks for GAMG on a part's
    /// pressure equation refuses on every rank alike instead of stranding
    /// the world in a collective.
    pub fn new(
        gpu: &Gpu,
        comm: &'c dyn Comm,
        dec: &Decomposition,
        ctrl: SolverControls,
        reduce: DistReduce,
    ) -> Result<Self> {
        refuse_gamg(&ctrl)?;
        Ok(Self {
            krylov: SpmdKrylov::new(gpu, comm, dec, reduce)?,
            ctrl,
        })
    }

    /// The Krylov route underneath: its rank, its workspace, its counts.
    pub fn krylov(&self) -> &SpmdKrylov<'c> {
        &self.krylov
    }

    /// The same, mutable - the flag vote and the workspace's counters live
    /// there.
    pub fn krylov_mut(&mut self) -> &mut SpmdKrylov<'c> {
        &mut self.krylov
    }
}

impl PressureBackend for SpmdPressureBackend<'_> {
    fn name(&self) -> &'static str {
        "SPMD Krylov"
    }

    fn applicable(&self, _probe: &SystemProbe) -> bool {
        // A part's pressure equation is an ordinary LDU system with a
        // halo-coupled boundary, exactly what the SPMD Krylov solves; no
        // structural fact of the WHOLE mesh can take it out.
        true
    }

    fn setup(&mut self, _gpu: &Gpu, hm: &HostMesh, m: &GpuMesh, _probe: &SystemProbe) -> Result<()> {
        let owned = self.krylov.n_cells();
        if hm.n_cells != owned || m.n_cells != owned {
            return Err(Error::Config(format!(
                "distflow: the SPMD Krylov pressure backend's setup on rank \
                 {} was given a host mesh of {} cells and an uploaded mesh \
                 of {} cells, but the part this rank owns has {}; set the \
                 backend up on the part's own mesh, uploaded with \
                 GpuMesh::upload_with_halo",
                self.krylov.rank(),
                hm.n_cells,
                m.n_cells,
                owned
            )));
        }
        Ok(())
    }

    fn solve(
        &mut self,
        gpu: &Gpu,
        p: &mut DevBuf<Scalar>,
        a: &GpuLduMatrix,
        m: &GpuMesh,
    ) -> Result<SolverPerformance> {
        self.krylov
            .run(gpu, p, a, m, &self.ctrl)
            .map(as_solver_performance)
    }

    /// The workspace's refined solution (SPEC-LIT 118.12), unless these
    /// controls are `fixed_iters`, which is never refined. `single` only.
    #[cfg(feature = "single")]
    fn refined_solution(&self) -> Option<&DevBuf<f64>> {
        if self.ctrl.fixed_iters {
            return None;
        }
        self.krylov.refined_solution()
    }
}

/// The momentum predictor's component-solve route over one rank's
/// [`SpmdKrylov`] (SPEC-LIT 139.3, PAR-10b).
///
/// Set on a part's `Momentum` through [`ComponentRoute`]
/// (`Simple::set_momentum_route` passes it through); `None` keeps the
/// serial `solver::solve` call launch for launch. Each of the three
/// component systems is one [`SpmdKrylov::run`], so the norms are the
/// world's and the same bits land on every rank.
pub struct SpmdMomentumRoute<'c> {
    krylov: SpmdKrylov<'c>,
}

impl<'c> SpmdMomentumRoute<'c> {
    /// Build the route for `comm.rank()`'s part of `dec`.
    pub fn new(
        gpu: &Gpu,
        comm: &'c dyn Comm,
        dec: &Decomposition,
        reduce: DistReduce,
    ) -> Result<Self> {
        Ok(Self {
            krylov: SpmdKrylov::new(gpu, comm, dec, reduce)?,
        })
    }

    /// The Krylov route underneath.
    pub fn krylov(&self) -> &SpmdKrylov<'c> {
        &self.krylov
    }

    /// The same, mutable.
    pub fn krylov_mut(&mut self) -> &mut SpmdKrylov<'c> {
        &mut self.krylov
    }
}

impl ComponentRoute for SpmdMomentumRoute<'_> {
    fn solve_component(
        &mut self,
        gpu: &Gpu,
        psi: &mut DevBuf<Scalar>,
        a: &GpuLduMatrix,
        m: &GpuMesh,
        ctrl: &SolverControls,
    ) -> Result<SolverPerformance> {
        self.krylov.run(gpu, psi, a, m, ctrl)
            .map(as_solver_performance)
    }
}


