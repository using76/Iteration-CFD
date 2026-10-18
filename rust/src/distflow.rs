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
//! SPEC-LIT 139.4 (PAR-10c) closes the loop: [`FlowExchange`] is the ONE
//! interface the SIMPLE loop needs to run on a part, and a `RankCtx` is what
//! implements it on a decomposed run - so `Simple` holds a `Box<dyn
//! FlowExchange>` where a serial run holds `None`, and every place the loop
//! touches another rank goes through it:
//!
//! | call | what crosses a rank |
//! |---|---|
//! | [`FlowExchange`] | the trait behind `Simple`'s decomposed run: a field's or a raw cell buffer's halo, the world's max, the world's OR, the world's mean, one owner's scalar, and the whole-mesh id of an owned cell |
//!
//! The same subsection is this module's cavity fixture - the lid-driven
//! cavity of Ghia, Ghia & Shin (1982) at Re = 100, run serial and
//! decomposed by [`cavity_serial`] and [`cavity_part`] - which is the gate
//! `tests/distflow_simple.rs` and `ofgpu-par -solve simple` measure.
//!
//! SPEC-LIT 139.5 (PAR-11) hands the energy equation and a passive scalar
//! the same two seams - `Energy::set_route` / `ScalarTransport::set_route`
//! (the SPMD Krylov behind [`ComponentRoute`]) and `Energy::set_exchange` /
//! `ScalarTransport::set_exchange` (the same [`FlowExchange`], now carried
//! by two more structs) - and this module the gate they answer to: the
//! differentially heated cavity of **G. de Vahl Davis, Int. J. Numer. Meth.
//! Fluids 3 (1983) 249-264** (DOI 10.1002/fld.1650030305) at `Ra = 1e4`
//! under the Boussinesq body force (SPEC-LIT 121), run serial and
//! decomposed by [`dvd_serial`] and [`dvd_part`] - the gate
//! `tests/distflow_thermal.rs` measures.
//!
//! Provenance: ORIGINAL - the exchange points are this project's own design
//! (SPEC-LIT 139.2, PAR-10a; 139.3, PAR-10b; 139.4, PAR-10c; 139.5, PAR-11)
//! over the SPMD
//! exchange of 138.3 (PAR-02), the boundary evaluation this crate already
//! ships and the SPMD Krylov of 138.5/138.6 (PAR-04a/b); no kernel is
//! launched here and no existing numerics change. No external source. No
//! GPL-licensed source was consulted.

use crate::blockgen::{build_case, CaseKind};
use crate::comm::Comm;
use crate::decompose::{Decomposition, PartMesh};
use crate::device::{DevBuf, Gpu};
use crate::distsolve::{spmd_solve, DistPerformance, DistReduce, DistSystem, SpmdWorkspace};
use crate::energy::{DomainKind, Energy, EnergyControls, GasProperties, GasState};
use crate::error::{Error, Result};
use crate::exactsum::SpmdReduction;
use crate::field::{BcKind, GpuScalarField, GpuVectorField};
use crate::field_ops::{self, FieldKernels};
use crate::fv::{DivScheme, SnGradScheme};
use crate::halo::SpmdHalo;
use crate::io::case::{ResidualControl, SolverControls, TurbulenceControls};
use crate::io::dict::FoamDict;
use crate::io::schemes::DivEntry;
use crate::ldu::GpuLduMatrix;
use crate::mesh::GpuMesh;
use crate::momentum::{BuoyancyCoeffs, BuoyancyForm, ComponentRoute, MomentumControls};
use crate::pressure::{PbicgstabBackend, PressureBackend, SystemProbe};
use crate::scalar_transport::{ScalarTransport, ScalarTransportCoeffs};
use crate::simple::{Simple, SimpleControls};
use crate::turbulence::FlowState;
use crate::solver::{
    LinearSolverKind, Preconditioner, SolverKernels, SolverPerformance,
};
use crate::HostMesh;
use crate::{Label, Scalar, Tensor, Vec3};

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
    global: Vec<Label>,
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
            // The whole-mesh id of every OWNED cell, local order - what
            // `local_of_global` answers out of (SPEC-LIT 139.4).
            global: part.global_cell[..part.mesh.n_cells].to_vec(),
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
//  The SIMPLE exchange seam - SPEC-LIT 139.4 (PAR-10c)
// ==========================================================================

/// SPEC-LIT 139.4 (PAR-10c): every place the SIMPLE loop touches another
/// rank.
///
/// A serial run never calls any of it - `Simple` holds the exchange as an
/// `Option` and every method's `None` arm is the launch it always made - so
/// the serial path is provably unchanged. A decomposed run hands `Simple` a
/// `Box<dyn FlowExchange>` over this module's [`RankCtx`].
///
/// Every reduction folds the gathered bytes in RANK ORDER, so every rank
/// gets identical bits whatever the transport does between them.
pub trait FlowExchange {
    /// Halo of `f.f`, then `field_ops::correct_boundary_conditions` - what
    /// [`RankCtx::exchange_field`] does.
    fn field(
        &mut self,
        gpu: &Gpu,
        k: &FieldKernels,
        f: &mut GpuScalarField,
        m: &GpuMesh,
    ) -> Result<()>;
    /// The same for a vector field - what [`RankCtx::exchange_field_vector`]
    /// does.
    fn field_vector(
        &mut self,
        gpu: &Gpu,
        k: &FieldKernels,
        u: &mut GpuVectorField,
        m: &GpuMesh,
    ) -> Result<()>;
    /// Halo of a raw cell buffer ([`SpmdHalo::scalar`] /
    /// [`SpmdHalo::vector`](crate::halo::SpmdHalo::vector)); no boundary
    /// evaluation follows.
    fn cells(&mut self, gpu: &Gpu, v: &mut DevBuf<Scalar>) -> Result<()>;
    /// The same for a three-component cell buffer.
    fn cells_vector(&mut self, gpu: &Gpu, v: &mut DevBuf<Vec3>) -> Result<()>;
    /// Halo of a raw Label cell buffer ([`SpmdHalo::label`]) - the integer
    /// masks a coupled face's neighbour is tested for; no boundary evaluation.
    /// SPEC-LIT 139.6 (PAR-12): what carries a constraint matrix's `is_fixed`
    /// across the cut before `set_values` eliminates the marked columns.
    fn cells_label(&mut self, gpu: &Gpu, v: &mut DevBuf<Label>) -> Result<()>;
    /// The world's max; NaN anywhere -> NaN on every rank
    /// ([`crate::simple::nan_propagating_max`]).
    fn max(&mut self, x: Scalar) -> Result<Scalar>;
    /// The world's OR.
    fn any(&mut self, b: bool) -> Result<bool>;
    /// Exactly one rank passes `Some(v)`; every rank gets `v`, bit for bit.
    fn owner_value(&mut self, mine: Option<Scalar>) -> Result<Scalar>;
    /// The LOCAL owned-cell index of whole-mesh cell `g` on this rank, if
    /// this rank owns it.
    fn local_of_global(&self, g: usize) -> Option<usize>;
    /// The world's mean of a per-cell quantity: this rank's sum over its
    /// OWNED cells and its owned count in, `Σ sums / Σ counts` out - every
    /// rank the same bits (a rank-order fold of the gathered f64 records).
    /// The pinned pressure equation's source is removed by the mean of the
    /// WHOLE source - `assemble_pressure`'s own invariant - and on a part
    /// that mean is a cross-part quantity, which is what this moves.
    fn world_mean(&mut self, local_sum: Scalar, n_owned: usize) -> Result<Scalar>;
}

impl FlowExchange for RankCtx<'_> {
    fn field(
        &mut self,
        gpu: &Gpu,
        k: &FieldKernels,
        f: &mut GpuScalarField,
        m: &GpuMesh,
    ) -> Result<()> {
        self.exchange_field(gpu, k, f, m)
    }

    fn field_vector(
        &mut self,
        gpu: &Gpu,
        k: &FieldKernels,
        u: &mut GpuVectorField,
        m: &GpuMesh,
    ) -> Result<()> {
        self.exchange_field_vector(gpu, k, u, m)
    }

    fn cells(&mut self, gpu: &Gpu, v: &mut DevBuf<Scalar>) -> Result<()> {
        self.halo.scalar(gpu, self.comm, v)
    }

    fn cells_vector(&mut self, gpu: &Gpu, v: &mut DevBuf<Vec3>) -> Result<()> {
        self.halo.vector(gpu, self.comm, v)
    }

    fn cells_label(&mut self, gpu: &Gpu, v: &mut DevBuf<Label>) -> Result<()> {
        self.halo.label(gpu, self.comm, v)
    }

    fn max(&mut self, x: Scalar) -> Result<Scalar> {
        let n = self.comm.size();
        let mut all = vec![0u8; 8 * n];
        self.comm
            .allgather(&(x as f64).to_le_bytes(), &mut all)?;
        let mut it = all
            .chunks_exact(8)
            .map(|c| f64::from_le_bytes(c.try_into().expect("8 bytes")));
        let first = it.next().unwrap_or(f64::NAN);
        // Folded in f64 so both builds reduce identically; `max` only picks,
        // never rounds, so the answer is one rank's own value.
        Ok(it
            .fold(first, |a: f64, b: f64| {
                if a.is_nan() || b.is_nan() {
                    f64::NAN
                } else {
                    a.max(b)
                }
            })
            as Scalar)
    }

    fn any(&mut self, b: bool) -> Result<bool> {
        let n = self.comm.size();
        let mut all = vec![0u8; 8 * n];
        let one = if b { 1.0f64 } else { 0.0f64 };
        self.comm.allgather(&one.to_le_bytes(), &mut all)?;
        Ok(all
            .chunks_exact(8)
            .any(|c| f64::from_le_bytes(c.try_into().expect("8 bytes")) != 0.0))
    }

    fn owner_value(&mut self, mine: Option<Scalar>) -> Result<Scalar> {
        // A 16-byte record per rank: the flag 1.0/0.0 as a little-endian
        // f64, then the value as a little-endian f64. The f64 bytes are the
        // same in both builds, so the record is too. The count is folded
        // AFTER the gather, so every rank sees the same array and refuses
        // alike.
        const REC: usize = 16;
        let n = self.comm.size();
        let mut rec = [0u8; REC];
        rec[..8].copy_from_slice(&if mine.is_some() {
            1.0f64.to_le_bytes()
        } else {
            0.0f64.to_le_bytes()
        });
        if let Some(v) = mine {
            rec[8..].copy_from_slice(&(v as f64).to_le_bytes());
        }
        let mut all = vec![0u8; REC * n];
        self.comm.allgather(&rec, &mut all)?;
        let mut count = 0usize;
        let mut v = 0.0f64;
        for c in all.chunks_exact(REC) {
            if f64::from_le_bytes(c[..8].try_into().expect("8 bytes")) != 0.0 {
                count += 1;
                v = f64::from_le_bytes(c[8..].try_into().expect("8 bytes"));
            }
        }
        if count != 1 {
            return Err(Error::Config(format!(
                "distflow: owner_value needs exactly one rank to pass Some, \
                 but {count} did - the reference cell must be owned by one \
                 rank and one rank only"
            )));
        }
        Ok(v as Scalar)
    }

    fn local_of_global(&self, g: usize) -> Option<usize> {
        self.global.iter().position(|&x| x as usize == g)
    }

    fn world_mean(&mut self, local_sum: Scalar, n_owned: usize) -> Result<Scalar> {
        // A 16-byte record per rank: the rank's sum as a little-endian f64,
        // then its owned-cell count as a little-endian f64 - the f64 bytes
        // are the same in both builds. The fold is in RANK ORDER, so every
        // rank divides the same two numbers.
        const REC: usize = 16;
        let n = self.comm.size();
        let mut rec = [0u8; REC];
        rec[..8].copy_from_slice(&(local_sum as f64).to_le_bytes());
        rec[8..].copy_from_slice(&(n_owned as f64).to_le_bytes());
        let mut all = vec![0u8; REC * n];
        self.comm.allgather(&rec, &mut all)?;
        let (mut ss, mut nn) = (0.0f64, 0.0f64);
        for c in all.chunks_exact(REC) {
            ss += f64::from_le_bytes(c[..8].try_into().expect("8 bytes"));
            nn += f64::from_le_bytes(c[8..].try_into().expect("8 bytes"));
        }
        Ok((ss / nn) as Scalar)
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



// ==========================================================================
//  The cavity fixture - SPEC-LIT 139.4 (PAR-10c)
// ==========================================================================

/// The gate constants, FIXED BEFORE MEASURING (docs/17 §12, row PAR-10c):
/// neither the test nor the driver may loosen them, and a run that misses
/// keeps the assert and reports its numbers.
///
/// The cavity: 64 cells per side, the lid-driven square of **U. Ghia, K. N.
/// Ghia, C. T. Shin, J. Comput. Phys. 48 (1982) 387-411**, DOI
/// 10.1016/0021-9991(82)90058-4, at `Re = U·L/nu = 100` (lid 1 m/s, side
/// 0.1 m, nu 1e-3).
pub const CAVITY_N: usize = 64;
/// The REDUCED cavity of the verify run: the same gate constants at 32
/// cells per side, because the 64 x 64 gate runs 20-46 min per rank count on
/// the shared card (SPEC-LIT 139.4) - the size is reduced, never a tolerance.
pub const CAVITY_N_REDUCED: usize = 32;
/// The outer loop's hard ceiling; the run stops on `residualControl` long
/// before it.
pub const CAVITY_MAX_OUTER: usize = 4000;
/// `residualControl` on U - a SOLVER control (the outer loop's stopping
/// criterion), not a gate tolerance.
pub const CAVITY_U_TOL: Scalar = if cfg!(feature = "single") { 1e-5 } else { 1e-7 };
/// The gate: decomposed vs serial, and centreline vs Ghia within the serial
/// run's own error. Fixed before measuring.
pub const CAVITY_TOL_FIELD: Scalar = if cfg!(feature = "single") { 1e-3 } else { 1e-6 };
pub const CAVITY_TOL_CENTRE: Scalar = CAVITY_TOL_FIELD;
/// The SIMPLE iteration count may move by this much of the serial count.
pub const CAVITY_ITER_BAND: f64 = 0.02;

/// `u` along the vertical centreline at `Re = 100`, Table I of Ghia, Ghia &
/// Shin (1982): `(y, u)`, `y` from the stationary floor, `u` scaled by the
/// lid speed. `src/simple.rs`'s own copy of the same table, byte for byte.
pub const GHIA_U_RE100: [(f64, f64); 17] = [
    (1.0000, 1.00000),
    (0.9766, 0.84123),
    (0.9688, 0.78871),
    (0.9609, 0.73722),
    (0.9531, 0.68717),
    (0.8516, 0.23151),
    (0.7344, 0.00332),
    (0.6172, -0.13641),
    (0.5000, -0.20581),
    (0.4531, -0.21090),
    (0.2813, -0.15662),
    (0.1719, -0.10150),
    (0.1016, -0.06434),
    (0.0703, -0.04775),
    (0.0625, -0.04192),
    (0.0547, -0.03717),
    (0.0000, 0.00000),
];

/// What one cavity run reports - a serial run's whole mesh, a decomposed
/// run's OWNED cells only (in the part's local order, `global_cell[i]` the
/// whole-mesh id of owned cell `i`).
pub struct CavityRun {
    pub global_cell: Vec<usize>,
    pub u: Vec<Vec3>,
    pub p: Vec<Scalar>,
    pub iterations: usize,
    pub converged: bool,
    pub u_residual: Scalar,
    pub continuity: Scalar,
}

/// The cavity mesh: the block generator's own `cavity` case, `n x n x 1` -
/// the cube 0..0.1 m, one cell deep, `leftWall`/`rightWall`/`fixedWall`/
/// `movingWall` walls and two `empty` patches.
pub fn cavity_mesh(n: usize) -> Result<HostMesh> {
    Ok(build_case(CaseKind::Cavity, n, n, 1)?.0)
}

/// Exactly `src/simple.rs`'s lid-driven-cavity controls, at `nu = 1e-3` so
/// `Re = 1·0.1/1e-3 = 100`: central convection (the cell Reynolds number is
/// 1.6), uncorrected snGrad, no variable-viscosity stress, relaxation 0.7 /
/// 0.3, PBiCGStab over the default Diagonal preconditioner, both solves at
/// `1e-10` in f64 (`1e-6` within 300 iterations under `single`, below) -
/// solver controls, not gate tolerances.
pub fn cavity_controls() -> SimpleControls {
    use crate::fv::DivScheme;
    // The inner pressure solve asks what the build's precision delivers on
    // this singular system. At f32 a deep PBiCGStab degenerates - alpha over
    // a vanished (r0, v) writes NaN into p (SPEC-LIT 112.3, 141.9's raw
    // float floor) - so the single build caps the float body at 300
    // iterations, inside which it reaches tol 1e-6 in ~100, and §118.4's
    // refinement carries the accuracy. The f64 ask is the measured 1e-10.
    #[cfg(feature = "single")]
    let (u_tol, p_tol, p_max): (Scalar, Scalar, i32) = (1e-6, 1e-6, 300);
    #[cfg(not(feature = "single"))]
    let (u_tol, p_tol, p_max): (Scalar, Scalar, i32) = (1e-10, 1e-10, 600);
    SimpleControls {
        momentum: MomentumControls {
            nu: 1e-3,
            u_solver: SolverControls {
                tolerance: u_tol,
                max_iter: 200,
                check_interval: 5,
                ..SolverControls::default()
            },
            u_relax: 0.7,
            div_scheme: DivScheme::Central,
            bounded_convection: true,
            sn_grad: SnGradScheme::Uncorrected,
            variable_viscosity_stress: false,
            ..MomentumControls::default()
        },
        p_solver: SolverControls {
            tolerance: p_tol,
            max_iter: p_max,
            check_interval: 5,
            ..SolverControls::default()
        },
        p_relax: 0.3,
        ..SimpleControls::default()
    }
}

/// The boundary-face patch of every boundary face, ascending.
fn patch_of(hm: &HostMesh) -> Vec<usize> {
    let mut v = vec![0usize; hm.n_boundary_faces];
    for (p, pi) in hm.patches.iter().enumerate() {
        for k in 0..pi.size {
            v[pi.start + k] = p;
        }
    }
    v
}

/// The cavity's boundary conditions, on the whole mesh and on a part alike,
/// BY PATCH NAME: `U` fixedValue (1,0,0) on `movingWall`, fixedValue 0 on
/// the other three walls (`fr` 1); `p` zeroGradient there. `Empty` patches
/// stay `Empty` and processor patches stay `Cyclic` with `fr` 0 - what
/// `GpuScalarField::zeros` seeds them with - so this writes the wall faces
/// only.
pub fn cavity_bcs(gpu: &Gpu, hm: &HostMesh, s: &mut Simple<'_>) -> Result<()> {
    let owner = patch_of(hm);
    let nbf = hm.n_boundary_faces;

    // Start from what `GpuVectorField::zeros` seeded - `Empty` on the empty
    // patches, `Cyclic` (fr 0) on every processor patch, zeroGradient
    // elsewhere - and overwrite the four walls only: a processor face
    // written zeroGradient would leave the coupled branch every kernel tests.
    let mut ukind: Vec<crate::Label> = gpu.download(&s.u().bc_kind)?;
    let mut ufr: Vec<Scalar> = gpu.download(&s.u().fr)?;
    let mut urv: Vec<Vec3> = gpu.download(&s.u().ref_value)?;
    let urg: Vec<Vec3> = gpu.download(&s.u().ref_grad)?;
    if ukind.len() != nbf {
        return Err(Error::Config(format!(
            "distflow: cavity_bcs was given a host mesh of {nbf} boundary faces and a Simple of {}",
            ukind.len()
        )));
    }
    for (i, &p) in owner.iter().enumerate() {
        let name = hm.patches[p].name.as_str();
        let wall = matches!(
            name,
            "movingWall" | "leftWall" | "rightWall" | "fixedWall"
        );
        if wall {
            ukind[i] = BcKind::FixedValue as crate::Label;
            ufr[i] = 1.0;
            urv[i] = if name == "movingWall" {
                Vec3::new(1.0, 0.0, 0.0)
            } else {
                Vec3::ZERO
            };
        }
    }
    {
        let u = s.u_mut();
        gpu.write(&mut u.bc_kind, &ukind)?;
        gpu.write(&mut u.fr, &ufr)?;
        gpu.write(&mut u.ref_value, &urv)?;
        gpu.write(&mut u.ref_grad, &urg)?;
    }
    Ok(())
}

/// Linear interpolation of a profile sampled at cell centres, extended to
/// the two walls by the boundary values the case prescribes - the same
/// sampler `src/simple.rs`'s cavity test uses.
fn sample(profile: &[(Scalar, Scalar)], at: Scalar) -> Scalar {
    if at <= profile[0].0 {
        return profile[0].1;
    }
    for w in profile.windows(2) {
        let (x0, v0) = w[0];
        let (x1, v1) = w[1];
        if at <= x1 {
            let t = (at - x0) / (x1 - x0);
            return v0 + t * (v1 - v0);
        }
    }
    profile[profile.len() - 1].1
}

/// |u_centreline(y) - u_Ghia| at the 17 stations. `u` is the whole mesh's
/// cell velocities in whole-mesh order; the centreline value at row `j` is
/// the mean of columns `n/2 - 1` and `n/2` - `x = 0.05` falls exactly
/// between two cell centres on the even mesh - and the profile runs wall to
/// wall, `(0, 0)` ... `(1, 1)`, linearly interpolated at each Ghia `y`.
pub fn cavity_ghia_errors(hm: &HostMesh, u: &[Vec3]) -> Vec<Scalar> {
    let n = (hm.n_cells as f64).sqrt() as usize;
    let h = 0.1 / n as Scalar; // the cavity cube is 0.1 m on a side
    let (ia, ib) = (n / 2 - 1, n / 2);

    let mut line = vec![0.0 as Scalar; n];
    let mut seen = vec![0usize; n];
    for (c, &cc) in hm.c.iter().enumerate() {
        let i = ((cc.x / h).floor() as usize).min(n - 1);
        let j = ((cc.y / h).floor() as usize).min(n - 1);
        if i == ia || i == ib {
            line[j] += u[c].x;
            seen[j] += 1;
        }
    }
    let mut prof: Vec<(Scalar, Scalar)> = vec![(0.0, 0.0)];
    for j in 0..n {
        // Ghia's y is normalised by the cavity side, L = n·h.
        let y = (j as Scalar + 0.5) * h / (h * n as Scalar);
        prof.push((y, line[j] / seen[j].max(1) as Scalar));
    }
    prof.push((1.0, 1.0));

    GHIA_U_RE100
        .iter()
        .map(|&(y, want)| (sample(&prof, y as Scalar) - want as Scalar).abs())
        .collect()
}

/// The outer loop both runners share: `residualControl` on U at
/// [`CAVITY_U_TOL`], then up to [`CAVITY_MAX_OUTER`] calls of
/// [`Simple::solve_step`], stopping when the loop says `converged`.
/// Returns (iterations, converged, worst U initial residual of the last
/// call, its continuity error).
fn cavity_loop(
    gpu: &Gpu,
    s: &mut Simple<'_>,
    backend: &mut dyn PressureBackend,
    nut: &GpuScalarField,
    t: &GpuScalarField,
) -> Result<(usize, bool, Scalar, Scalar)> {
    let src = format!("SIMPLE {{ residualControl {{ U {CAVITY_U_TOL:e}; }} }}");
    let d = FoamDict::parse(&src, "fvSolution")?;
    s.set_residual_control(ResidualControl::read(&d));

    let mut iterations = 0usize;
    let mut last = crate::simple::PimplePerformance::default();
    for _ in 0..CAVITY_MAX_OUTER {
        last = s.solve_step(gpu, backend, nut, t)?;
        iterations += 1;
        if last.converged {
            break;
        }
    }
    let u_residual = last
        .last
        .u
        .iter()
        .map(|q| q.initial_residual)
        .fold(0.0 as Scalar, Scalar::max);
    Ok((iterations, last.converged, u_residual, last.last.continuity_error))
}

/// A zero eddy viscosity, never written - a laminar run.
fn cavity_nut(gpu: &Gpu, m: &GpuMesh) -> Result<GpuScalarField> {
    GpuScalarField::zeros(gpu, m, "nut")
}

/// A uniform temperature at the reference, corrected once, so the buoyancy
/// body force is identically zero (`g = Vec3::ZERO` besides).
fn cavity_t(gpu: &Gpu, m: &GpuMesh, fk: &FieldKernels) -> Result<GpuScalarField> {
    let t_ref = BuoyancyCoeffs::default().t_ref;
    let mut t = GpuScalarField::zeros(gpu, m, "T")?;
    gpu.write(&mut t.f, &vec![t_ref; m.n_local()])?;
    field_ops::correct_boundary_conditions(gpu, fk, &mut t, m)?;
    Ok(t)
}

/// The cavity on the WHOLE mesh, on one card - the serial reference. No
/// route, no exchange: the calls `Simple` has always made.
pub fn cavity_serial(gpu: &Gpu, hm: &HostMesh) -> Result<CavityRun> {
    let m = GpuMesh::upload(gpu, hm)?;
    let ctrl = cavity_controls();
    let mut s = Simple::new(
        gpu,
        hm,
        &m,
        ctrl,
        BuoyancyCoeffs {
            g: Vec3::ZERO,
            ..BuoyancyCoeffs::default()
        },
    )?;
    cavity_bcs(gpu, hm, &mut s)?;
    s.initialise(gpu)?;

    let fk = FieldKernels::new(gpu)?;
    let nut = cavity_nut(gpu, &m)?;
    let t = cavity_t(gpu, &m, &fk)?;

    let mut backend = PbicgstabBackend::new(cavity_controls().p_solver);
    backend.setup(gpu, hm, &m, &SystemProbe::default())?;

    let (iterations, converged, u_residual, continuity) =
        cavity_loop(gpu, &mut s, &mut backend, &nut, &t)?;
    let n = hm.n_cells;
    Ok(CavityRun {
        global_cell: (0..n).collect(),
        u: gpu.download(&s.u().f)?,
        p: gpu.download(&s.p().f)?,
        iterations,
        converged,
        u_residual,
        continuity,
    })
}

/// The cavity on rank `comm.rank()`'s part of `dec` - the decomposed run:
/// the part uploaded with its halo, the momentum predictor and the pressure
/// equation routed through the SPMD Krylov (139.3), the exchange set (139.4)
/// and the SAME boundary conditions, by patch name, the serial run gave the
/// whole mesh.
pub fn cavity_part(gpu: &Gpu, comm: &dyn Comm, dec: &Decomposition) -> Result<CavityRun> {
    let part = &dec.parts[comm.rank()];
    let hm = &part.mesh;
    let gm = GpuMesh::upload_with_halo(gpu, hm, part.n_halo)?;
    let mut s = Simple::new(
        gpu,
        hm,
        &gm,
        cavity_controls(),
        BuoyancyCoeffs {
            g: Vec3::ZERO,
            ..BuoyancyCoeffs::default()
        },
    )?;
    s.set_momentum_route(Some(Box::new(SpmdMomentumRoute::new(
        gpu,
        comm,
        dec,
        DistReduce::Exact,
    )?)));
    s.set_exchange(Some(Box::new(RankCtx::new(gpu, comm, part, &gm)?)))?;

    let mut backend =
        SpmdPressureBackend::new(gpu, comm, dec, cavity_controls().p_solver, DistReduce::Exact)?;
    backend.setup(gpu, hm, &gm, &SystemProbe::default())?;

    cavity_bcs(gpu, hm, &mut s)?;
    s.initialise(gpu)?;

    let fk = FieldKernels::new(gpu)?;
    let nut = cavity_nut(gpu, &gm)?;
    let t = cavity_t(gpu, &gm, &fk)?;

    let (iterations, converged, u_residual, continuity) =
        cavity_loop(gpu, &mut s, &mut backend, &nut, &t)?;
    let n = hm.n_cells;
    Ok(CavityRun {
        global_cell: part.global_cell[..n].iter().map(|&g| g as usize).collect(),
        u: gpu.download(&s.u().f)?[..n].to_vec(),
        p: gpu.download(&s.p().f)?[..n].to_vec(),
        iterations,
        converged,
        u_residual,
        continuity,
    })
}

// ==========================================================================
//  The differentially heated cavity - SPEC-LIT 139.5 (PAR-11)
// ==========================================================================

/// The gate constants, FIXED BEFORE MEASURING (docs/17 §12, row PAR-11):
/// neither the test nor the driver may loosen them, and a run that misses
/// keeps the assert and reports its numbers.
///
/// The cavity: 40 cells per side, the differentially heated square of
/// **G. de Vahl Davis, Int. J. Numer. Meth. Fluids 3 (1983) 249-264**, DOI
/// 10.1002/fld.1650030305, at `Ra = g beta DeltaT L^3/(nu alpha) = 1e4`
/// under the Boussinesq body force (SPEC-LIT 121): `leftWall` hot at
/// 304.5 K, `rightWall` cold at 295.5 K, the horizontal walls insulated,
/// side 0.1 m, the mean hot-wall Nusselt number 2.243 (read, as SPEC-LIT
/// 121.8 reads it, from Qi et al., Nanoscale Research Letters 8 (2013) 56,
/// Table 3, open access).
pub const DVD_N: usize = 40;
/// The REDUCED cavity of the verify run: the same gate constants at 20
/// cells per side - the size is reduced, never a tolerance.
pub const DVD_N_REDUCED: usize = 20;
pub const DVD_RA: f64 = 1.0e4;
/// de Vahl Davis (1983)'s mean hot-wall Nusselt number, as SPEC-LIT 121.8
/// reads it.
pub const DVD_NU_PUBLISHED: f64 = 2.243;
/// Class P: the serial run's Nu within this of the published 2.243, the
/// same check in both builds.
pub const DVD_NU_BAND: f64 = 0.01;
/// Class R: decomposed against serial, judged at
/// [`crate::precision::roundoff_tol`] of this.
pub const DVD_TOL_SERIAL: f64 = 1e-6;
/// Class D: the outer iteration count may move by this much of the serial
/// count.
pub const DVD_ITER_BAND: f64 = 0.02;
/// The outer loop's hard ceiling; the run stops on its residuals long
/// before it.
pub const DVD_MAX_OUTER: usize = 3000;
/// The convergence ask on the four initial residuals - a SOLVER control,
/// not a gate tolerance.
pub const DVD_RESIDUAL: Scalar = if cfg!(feature = "single") { 1e-5 } else { 1e-7 };

/// The fixture's physical numbers, in f64: side, the two wall temperatures,
/// their difference, the reference, `beta = 1/T_ref` (so `beta DeltaT =
/// 0.03`, as Gate CMP-B's), `nu` and `alpha` (`Pr = 0.71`, air at the
/// reference).
pub const DVD_L: f64 = 0.1;
pub const DVD_T_HOT: f64 = 304.5;
pub const DVD_T_COLD: f64 = 295.5;
pub const DVD_DELTA_T: f64 = DVD_T_HOT - DVD_T_COLD;
pub const DVD_T_REF: f64 = 300.0;
pub const DVD_BETA: f64 = 1.0 / 300.0;
pub const DVD_NU: f64 = 0.71e-3;
pub const DVD_ALPHA: f64 = 1.0e-3;
/// The Boussinesq body force's magnitude that REALISES the Rayleigh number:
/// `Ra = g beta DeltaT L^3/(nu alpha)` solved for
/// `g = Ra nu alpha/(beta DeltaT L^3)` - 236.666... , computed here, never
/// rounded.
pub const DVD_G_Y: f64 =
    DVD_RA * DVD_NU * DVD_ALPHA / (DVD_BETA * DVD_DELTA_T * DVD_L * DVD_L * DVD_L);

/// What one heated-cavity run reports - a serial run's whole mesh, a
/// decomposed run's OWNED cells only (in the part's local order,
/// `global_cell[i]` the whole-mesh id of owned cell `i`).
pub struct DvdRun {
    pub global_cell: Vec<usize>,
    pub u: Vec<Vec3>,
    pub p: Vec<Scalar>,
    pub t: Vec<Scalar>,
    pub s: Vec<Scalar>,
    pub iterations: usize,
    pub converged: bool,
    /// U, p, T and s INITIAL residuals of the last iteration, in that
    /// order - the four numbers the convergence test reads.
    pub residuals: [Scalar; 4],
}

/// (121.9) of SPEC-LIT 121.8 on the whole mesh, in f64: the mean hot-wall
/// Nusselt number
///
/// ```text
/// Nu = L * sum_b (T_h - T_P(b)) delta_b |S_b| / (DeltaT * sum_b |S_b|)
/// ```
///
/// over the faces `b` of patch `leftWall`, `T_P(b)` the hot wall's
/// neighbour cell (`t[hm.b_face_cells[b]]`), `delta_b = hm.b_delta_coeffs`
/// and `|S_b| = hm.b_mag_sf`. The caller hands the WHOLE mesh's field - a
/// decomposed run gathers its owned cells first.
pub fn dvd_nusselt(hm: &HostMesh, t: &[Scalar]) -> f64 {
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    for pi in &hm.patches {
        if pi.name != "leftWall" {
            continue;
        }
        for b in pi.start..pi.start + pi.size {
            let t_p = t[hm.b_face_cells[b] as usize] as f64;
            let delta = hm.b_delta_coeffs[b] as f64;
            let mag_sf = hm.b_mag_sf[b] as f64;
            num += (DVD_T_HOT - t_p) * delta * mag_sf;
            den += mag_sf;
        }
    }
    DVD_L * num / (DVD_DELTA_T * den)
}

/// The heated cavity's controls: [`cavity_controls`]' central, bounded,
/// uncorrected PBiCGStab-Diagonal momentum at the Boussinesq `nu` that
/// realises `Ra = 1e4` with `alpha = 1e-3`.
///
/// Under `single` the pressure solve's float body is capped at 100
/// iterations instead of the cavity's 300: at 40 x 40 the 300 cap let the
/// f32 PBiCGStab degenerate (NaN in U and p at outer 22), the raw float
/// floor SPEC-LIT 141.9 names, and its cure - an input control, the
/// reachable `1e-6` inside the cap with 118.4's refinement carrying the
/// accuracy. The f64 build keeps `cavity_controls`' ask unchanged.
pub fn dvd_controls() -> SimpleControls {
    let mut c = cavity_controls();
    c.momentum.nu = DVD_NU as Scalar;
    #[cfg(feature = "single")]
    {
        c.p_solver.max_iter = 100;
    }
    c
}

/// The T and s linear solves: PBiCGStab over the default Diagonal
/// preconditioner, `1e-10` in f64 and `1e-6` under `single`, 200 iterations,
/// checked every 5 - solver controls, not gate tolerances.
fn dvd_transport_solver() -> SolverControls {
    SolverControls {
        tolerance: if cfg!(feature = "single") { 1e-6 } else { 1e-10 },
        max_iter: 200,
        check_interval: 5,
        ..SolverControls::default()
    }
}

/// The T equation's controls: relaxation 0.7, central bounded convection,
/// snGrad uncorrected, steady - exactly what a decomposed run may run
/// (SPEC-LIT 139.5's refusal table).
fn dvd_energy_controls() -> EnergyControls {
    EnergyControls {
        t_solver: dvd_transport_solver(),
        div_scheme: DivEntry { scheme: DivScheme::Central, bounded: true },
        sn_grad: SnGradScheme::Uncorrected,
        ..EnergyControls::default()
    }
}

/// The scalar's controls: the same asks as T's, on the k-entry struct the
/// scalar reads them from.
fn dvd_scalar_controls() -> TurbulenceControls {
    TurbulenceControls {
        k_solver: dvd_transport_solver(),
        div_scheme: DivScheme::Central,
        bounded_convection: true,
        sn_grad: SnGradScheme::Uncorrected,
        ..TurbulenceControls::default()
    }
}

/// The gas: ideal-gas properties pinned so `rho = p0/(R_s T_ref) = 1
/// kg/m^3` (`W = R rho_ref T_ref/p0`), `cp = 1`, `k = 1e-3` - so the
/// thermal diffusity `k/(rho cp)` IS the `alpha = 1e-3` the Rayleigh
/// number asks for - and `Pr = 0.71`.
pub fn dvd_props() -> GasProperties {
    GasProperties {
        w: 8.314_462_618 * 1.0 * 300.0 / 1e5,
        cp: 1.0,
        k: 1e-3,
        pr: 0.71,
        ..GasProperties::default()
    }
}

/// The Boussinesq body force (SPEC-LIT 121.3): gravity down `-y` at
/// [`DVD_G_Y`], reference 300 K, `beta = 1/300`.
pub fn dvd_buoyancy() -> BuoyancyCoeffs {
    BuoyancyCoeffs {
        g: Vec3::new(0.0, -(DVD_G_Y as Scalar), 0.0),
        t_ref: DVD_T_REF as Scalar,
        form: BuoyancyForm::Boussinesq { beta: DVD_BETA as Scalar },
        ..BuoyancyCoeffs::default()
    }
}

/// The two isothermal (hot `leftWall`, cold `rightWall`) patches of a
/// scalar field, on the whole mesh and on a part alike, BY PATCH NAME:
/// `fixedValue` at `hot` / `cold`, the horizontal walls left at the
/// zeroGradient `GpuScalarField::zeros` seeded, and `Empty`/`Cyclic` faces
/// untouched - the same by-patch-name stance `cavity_bcs` takes.
fn dvd_wall_values(
    gpu: &Gpu,
    hm: &HostMesh,
    f: &mut GpuScalarField,
    hot: Scalar,
    cold: Scalar,
) -> Result<()> {
    let owner = patch_of(hm);
    let mut kind: Vec<Label> = gpu.download(&f.bc_kind)?;
    let mut fr: Vec<Scalar> = gpu.download(&f.fr)?;
    let mut rv: Vec<Scalar> = gpu.download(&f.ref_value)?;
    for (i, &p) in owner.iter().enumerate() {
        let value = match hm.patches[p].name.as_str() {
            "leftWall" => hot,
            "rightWall" => cold,
            _ => continue,
        };
        kind[i] = BcKind::FixedValue as Label;
        fr[i] = 1.0;
        rv[i] = value;
    }
    gpu.write(&mut f.bc_kind, &kind)?;
    gpu.write(&mut f.fr, &fr)?;
    gpu.write(&mut f.ref_value, &rv)
}

/// The heated cavity's boundary conditions, on the whole mesh and on a part
/// alike, BY PATCH NAME: `U` fixedValue (0,0,0) (`fr` 1) on ALL FOUR walls -
/// de Vahl Davis's cavity is driven by buoyancy alone, and the lid motion
/// [`cavity_bcs`] writes on `movingWall` would run a lid-driven buoyant
/// cavity instead - and `p` left at the zeroGradient seed. `Empty` patches
/// stay `Empty` and processor patches stay `Cyclic` with `fr` 0 - what
/// `GpuScalarField`/`GpuVectorField::zeros` seed them with - so this writes
/// the wall faces only.
pub fn dvd_bcs(gpu: &Gpu, hm: &HostMesh, s: &mut Simple<'_>) -> Result<()> {
    let owner = patch_of(hm);

    // Start from what `GpuVectorField::zeros` seeded - `Empty` on the empty
    // patches, `Cyclic` (fr 0) on every processor patch, zeroGradient
    // elsewhere - and overwrite the four walls only: a processor face
    // written zeroGradient would leave the coupled branch every kernel tests.
    let mut ukind: Vec<crate::Label> = gpu.download(&s.u().bc_kind)?;
    let mut ufr: Vec<Scalar> = gpu.download(&s.u().fr)?;
    let mut urv: Vec<Vec3> = gpu.download(&s.u().ref_value)?;
    let urg: Vec<Vec3> = gpu.download(&s.u().ref_grad)?;
    if ukind.len() != hm.n_boundary_faces {
        return Err(Error::Config(format!(
            "distflow: dvd_bcs was given a host mesh of {} boundary faces and a Simple of {}",
            hm.n_boundary_faces,
            ukind.len()
        )));
    }
    for (i, &p) in owner.iter().enumerate() {
        let wall = matches!(
            hm.patches[p].name.as_str(),
            "movingWall" | "leftWall" | "rightWall" | "fixedWall"
        );
        if wall {
            ukind[i] = BcKind::FixedValue as crate::Label;
            ufr[i] = 1.0;
            urv[i] = Vec3::ZERO;
        }
    }
    let u = s.u_mut();
    gpu.write(&mut u.bc_kind, &ukind)?;
    gpu.write(&mut u.fr, &ufr)?;
    gpu.write(&mut u.ref_value, &urv)?;
    gpu.write(&mut u.ref_grad, &urg)
}

/// One outer iteration of the heated cavity, and the convergence test over
/// it: SIMPLE's momentum predictor + pressure equation, then `T`, then the
/// passive scalar `s`; converged when the worst U initial residual and the
/// p, T and s initial residuals are ALL under [`DVD_RESIDUAL`]. Returns
/// `(iterations, converged, [U, p, T, s])` with the residuals of the last
/// iteration.
#[allow(clippy::too_many_arguments)]
fn dvd_loop(
    gpu: &Gpu,
    s: &mut Simple<'_>,
    backend: &mut dyn PressureBackend,
    energy: &mut Energy<'_>,
    gas: &GasState<'_>,
    scalar: &mut ScalarTransport<'_>,
    nut: &GpuScalarField,
    tke: &DevBuf<Scalar>,
    nu: Scalar,
) -> Result<(usize, bool, [Scalar; 4])> {
    let mut iterations = 0usize;
    let mut converged = false;
    let mut last = [0.0 as Scalar; 4];
    for _ in 0..DVD_MAX_OUTER {
        let perf = s.correct_outer(gpu, backend, nut, energy.field(), false)?;
        let e_perf = energy.correct(gpu, s.convective_flux(), nut, tke, nu, gas)?;
        let flow = FlowState { u: s.u(), phi: s.convective_flux(), nu };
        let s_perf = scalar.correct(gpu, &flow, nut)?;
        let u_res = perf
            .u
            .iter()
            .map(|q| q.initial_residual)
            .fold(0.0 as Scalar, Scalar::max);
        last = [u_res, perf.p.initial_residual, e_perf.initial_residual, s_perf.initial_residual];
        iterations += 1;
        if last.iter().all(|&r| r < DVD_RESIDUAL) {
            converged = true;
            break;
        }
    }
    Ok((iterations, converged, last))
}

/// The heated cavity on the WHOLE mesh, on one card - the serial reference.
/// No route, no exchange on `T` or `s`: the calls `Energy` and
/// `ScalarTransport` have always made.
pub fn dvd_serial(gpu: &Gpu, hm: &HostMesh) -> Result<DvdRun> {
    let m = GpuMesh::upload(gpu, hm)?;
    let mut s = Simple::new(gpu, hm, &m, dvd_controls(), dvd_buoyancy())?;
    dvd_bcs(gpu, hm, &mut s)?;
    s.initialise(gpu)?;

    let mut backend = PbicgstabBackend::new(dvd_controls().p_solver);
    backend.setup(gpu, hm, &m, &SystemProbe::default())?;

    let props = dvd_props();
    let mut energy = Energy::new(gpu, &m, dvd_energy_controls(), props)?;
    let mut gas = GasState::new(gpu, &m, props, DomainKind::Open, 1e5)?;
    let mut scalar = ScalarTransport::new(
        gpu,
        hm,
        &m,
        "s",
        ScalarTransportCoeffs { pr: 0.71, prt: 0.85 },
        dvd_scalar_controls(),
    )?;

    let nut = cavity_nut(gpu, &m)?;
    let tke = gpu.zeros(m.n_local())?;

    // T: uniform at the reference, then the two isothermal walls, then the
    // boundary evaluation and the old-time seed - `initialise`. The gas
    // state's density comes from it ONCE: Boussinesq pins it (SPEC-LIT
    // 121.7), and the density-ratio divide never runs on this force law.
    gpu.write(&mut energy.field_mut().f, &vec![DVD_T_REF as Scalar; m.n_local()])?;
    dvd_wall_values(gpu, hm, energy.field_mut(), DVD_T_HOT as Scalar, DVD_T_COLD as Scalar)?;
    energy.initialise(gpu)?;
    gas.update_density(gpu, energy.field())?;

    // s: zero everywhere (`zeros`' seed), hot on `leftWall`, cold on
    // `rightWall`.
    dvd_wall_values(gpu, hm, scalar.field_mut(), 1.0, 0.0)?;
    scalar.initialise(gpu)?;

    let (iterations, converged, residuals) = dvd_loop(
        gpu,
        &mut s,
        &mut backend,
        &mut energy,
        &gas,
        &mut scalar,
        &nut,
        &tke,
        DVD_NU as Scalar,
    )?;
    let n = hm.n_cells;
    Ok(DvdRun {
        global_cell: (0..n).collect(),
        u: gpu.download(&s.u().f)?,
        p: gpu.download(&s.p().f)?,
        t: gpu.download(&energy.field().f)?,
        s: gpu.download(&scalar.field().f)?,
        iterations,
        converged,
        residuals,
    })
}

/// The heated cavity on rank `comm.rank()`'s part of `dec` - the decomposed
/// run: `cavity_part`'s Simple except the walls (`dvd_bcs`: all four
/// no-slip, buoyancy-driven) - SPMD momentum route, SPMD pressure backend,
/// exchange - plus, for `T` and for `s` each, its own
/// [`SpmdMomentumRoute`] through `set_route` and its own [`RankCtx`] through
/// `set_exchange` (SPEC-LIT 139.5). Every rank constructs them in the same
/// order - they are collective.
pub fn dvd_part(gpu: &Gpu, comm: &dyn Comm, dec: &Decomposition) -> Result<DvdRun> {
    let part = &dec.parts[comm.rank()];
    let hm = &part.mesh;
    let gm = GpuMesh::upload_with_halo(gpu, hm, part.n_halo)?;
    let mut s = Simple::new(gpu, hm, &gm, dvd_controls(), dvd_buoyancy())?;
    s.set_momentum_route(Some(Box::new(SpmdMomentumRoute::new(
        gpu,
        comm,
        dec,
        DistReduce::Exact,
    )?)));
    s.set_exchange(Some(Box::new(RankCtx::new(gpu, comm, part, &gm)?)))?;

    let mut backend =
        SpmdPressureBackend::new(gpu, comm, dec, dvd_controls().p_solver, DistReduce::Exact)?;
    backend.setup(gpu, hm, &gm, &SystemProbe::default())?;

    dvd_bcs(gpu, hm, &mut s)?;
    s.initialise(gpu)?;

    let props = dvd_props();
    let mut energy = Energy::new(gpu, &gm, dvd_energy_controls(), props)?;
    let mut gas = GasState::new(gpu, &gm, props, DomainKind::Open, 1e5)?;
    energy.set_route(Some(Box::new(SpmdMomentumRoute::new(
        gpu,
        comm,
        dec,
        DistReduce::Exact,
    )?)));
    energy.set_exchange(Some(Box::new(RankCtx::new(gpu, comm, part, &gm)?)))?;
    let mut scalar = ScalarTransport::new(
        gpu,
        hm,
        &gm,
        "s",
        ScalarTransportCoeffs { pr: 0.71, prt: 0.85 },
        dvd_scalar_controls(),
    )?;
    scalar.set_route(Some(Box::new(SpmdMomentumRoute::new(
        gpu,
        comm,
        dec,
        DistReduce::Exact,
    )?)));
    scalar.set_exchange(Some(Box::new(RankCtx::new(gpu, comm, part, &gm)?)))?;

    let nut = cavity_nut(gpu, &gm)?;
    let tke = gpu.zeros(gm.n_local())?;

    gpu.write(&mut energy.field_mut().f, &vec![DVD_T_REF as Scalar; gm.n_local()])?;
    dvd_wall_values(gpu, hm, energy.field_mut(), DVD_T_HOT as Scalar, DVD_T_COLD as Scalar)?;
    energy.initialise(gpu)?;
    gas.update_density(gpu, energy.field())?;
    dvd_wall_values(gpu, hm, scalar.field_mut(), 1.0, 0.0)?;
    scalar.initialise(gpu)?;

    let (iterations, converged, residuals) = dvd_loop(
        gpu,
        &mut s,
        &mut backend,
        &mut energy,
        &gas,
        &mut scalar,
        &nut,
        &tke,
        DVD_NU as Scalar,
    )?;
    let n = hm.n_cells;
    Ok(DvdRun {
        global_cell: part.global_cell[..n].iter().map(|&g| g as usize).collect(),
        u: gpu.download(&s.u().f)?[..n].to_vec(),
        p: gpu.download(&s.p().f)?[..n].to_vec(),
        t: gpu.download(&energy.field().f)?[..n].to_vec(),
        s: gpu.download(&scalar.field().f)?[..n].to_vec(),
        iterations,
        converged,
        residuals,
    })
}

// ==========================================================================
//  The streamwise-periodic channel - SPEC-LIT 139.6 (PAR-12)
// ==========================================================================

use crate::field_setup::{NutRoughness, WallFaces};
use crate::io::case::CaseControls;
use crate::mesh::PatchKind;
use crate::models::{KEpsilon, KEpsilonCoeffs, KOmegaSst, KOmegaSstCoeffs};
use crate::sources::SourceSpec;

/// The gate constants, FIXED BEFORE MEASURING (docs/17 §12, row PAR-12):
/// neither the test nor the driver may loosen them, and a run that misses
/// keeps the assert and reports its numbers.
///
/// The channel: the case `cases/channelPeriodicWF.jsonc` - a streamwise
/// CYCLIC plane duct, 8 x 6 x 4 cells (192), the two `wallHot` and the two
/// `wallSide` patches no-slip walls, the flow driven by the case's own
/// `momentumSource` body force (3.9 m/s² along x) in place of an inlet, run
/// with k-epsilon and with k-omega SST on the wall functions the case names
/// (`kappa` 0.41, `E` 9.8).
pub const CHANNEL_P: usize = 4;
/// Class R: decomposed against serial, judged at
/// [`crate::precision::roundoff_tol`] of this.
pub const CHANNEL_TOL_SERIAL: f64 = 1e-6;
/// The outer loop's hard ceiling of the plan's gate.
pub const CHANNEL_MAX_OUTER: usize = 3000;
/// The outer loop's hard ceiling of the short-trajectory verify run.
pub const CHANNEL_SHORT_OUTER: usize = 20;
/// The convergence ask on the four initial residuals - a SOLVER control, not
/// a gate tolerance.
pub const CHANNEL_RESIDUAL: Scalar = if cfg!(feature = "single") { 1e-5 } else { 1e-7 };
/// The case's `numerics.relaxation`, verbatim.
pub const CHANNEL_U_RELAX: Scalar = 0.5;
pub const CHANNEL_P_RELAX: Scalar = 0.3;
/// The case's `initial`, verbatim - and SST's `omega` from the same
/// epsilon, `omega = epsilon/(beta_star k)`.
pub const CHANNEL_U0: Scalar = 3.0;
pub const CHANNEL_K0: Scalar = 0.03;
pub const CHANNEL_EPS0: Scalar = 0.3;
pub const CHANNEL_OMEGA0: Scalar = CHANNEL_EPS0 / (0.09 * CHANNEL_K0);

/// Which turbulence model the channel runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChannelModel {
    /// Standard k-epsilon (SPEC-LIT 6.1), constant sigma.
    KEpsilon,
    /// Menter k-omega SST (SPEC-LIT 6.3), blended sigma - the model the
    /// coupled diffusivity exists for.
    KOmegaSst,
}

impl ChannelModel {
    /// The model, named as the case file names it.
    pub fn name(self) -> &'static str {
        match self {
            Self::KEpsilon => "kEpsilon",
            Self::KOmegaSst => "kOmegaSST",
        }
    }
}

/// The channel, as the case file lowers it: the host mesh, the case controls
/// and the `sources[]` entries - the momentum body force a periodic domain
/// uses in place of an inlet.
pub struct ChannelCase {
    pub hm: HostMesh,
    pub cc: CaseControls,
    pub sources: Vec<SourceSpec>,
}

/// Read and lower `cases/channelPeriodicWF.jsonc`.
pub fn channel_case() -> Result<ChannelCase> {
    let path = std::path::Path::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../cases/channelPeriodicWF.jsonc"
    ));
    let lowered = crate::io::case_json::read_case_jsonc(path)?.lower()?;
    let hm = lowered.build_host_mesh()?;
    let cc = lowered.to_case_controls();
    Ok(ChannelCase {
        hm,
        cc,
        sources: lowered.sources,
    })
}

/// The channel's flow controls: the case's own solvers and relaxations, with
/// the numerics a decomposed run refuses (SPEC-LIT 139.4's refusal table)
/// replaced by the SAME numerics on both sides of the comparison -
/// `linearUpwind` -> `upwind` and `corrected` -> `uncorrected` snGrad (the
/// mesh is orthogonal, where the correction is identically zero, so the
/// substitution is exact on the serial side too), the pressure solve's `DIC`
/// preconditioner -> `Diagonal` (cavity_controls' precedent), and the
/// transpose-free stress. What one side may not run, neither runs.
pub fn channel_controls(cc: &CaseControls) -> SimpleControls {
    let mut p_solver = cc.p_solver;
    p_solver.precon = Preconditioner::Diagonal;
    if cfg!(feature = "single") {
        p_solver.max_iter = p_solver.max_iter.min(300);
    }
    SimpleControls {
        momentum: MomentumControls {
            nu: cc.nu,
            u_solver: cc.u_solver,
            u_relax: CHANNEL_U_RELAX,
            div_scheme: DivScheme::Upwind,
            bounded_convection: false,
            sn_grad: SnGradScheme::Uncorrected,
            variable_viscosity_stress: false,
            ..MomentumControls::default()
        },
        p_solver,
        p_relax: CHANNEL_P_RELAX,
        ..SimpleControls::default()
    }
}

/// The channel's turbulence controls: the case's own, with `snGrad`
/// `uncorrected` and zero non-orthogonal correctors - the two refusals of
/// [`crate::turbulence::RasCore::set_exchange`]'s table, applied to the serial
/// side too so the comparison runs one numerics.
pub fn channel_turbulence(cc: &CaseControls) -> TurbulenceControls {
    TurbulenceControls {
        sn_grad: SnGradScheme::Uncorrected,
        n_non_orth_correctors: 0,
        ..cc.turb
    }
}

/// A whole-mesh field restricted to a part's OWNED cells, local order:
/// `whole[global_cell[i]]` for `i < part.mesh.n_cells`.
pub fn split_cells(whole: &[Scalar], part: &PartMesh) -> Vec<Scalar> {
    part.global_cell[..part.mesh.n_cells]
        .iter()
        .map(|&g| whole[g as usize])
        .collect()
}

/// Tucker's Poisson wall distance of SPEC-LIT 6.6, solved on the WHOLE mesh -
/// a decomposed run never solves it on a part; it splits this by
/// [`split_cells`] instead. The first `hm.n_cells` values of `y.f`.
pub fn channel_wall_distance(gpu: &Gpu, hm: &HostMesh, cc: &CaseControls) -> Result<Vec<Scalar>> {
    let m = GpuMesh::upload(gpu, hm)?;
    let wd = crate::walldistance::wall_distance(gpu, hm, &m, &cc.p_solver, 0)?;
    let y = gpu.download(&wd.y.f)?;
    Ok(y[..hm.n_cells].to_vec())
}

/// What one channel run reports - a serial run's whole mesh, a decomposed
/// run's OWNED cells only (in the part's local order, `global_cell[i]` the
/// whole-mesh id of owned cell `i`). `w` is `epsilon` or `omega`, whichever
/// the model transports.
pub struct ChannelRun {
    pub global_cell: Vec<usize>,
    pub u: Vec<Vec3>,
    pub k: Vec<Scalar>,
    pub w: Vec<Scalar>,
    pub nut: Vec<Scalar>,
    pub iterations: usize,
    pub converged: bool,
    pub diverged: bool,
    /// U, p, k and w INITIAL residuals of the last iteration, in that order -
    /// the four numbers the convergence test reads.
    pub residuals: [Scalar; 4],
}

/// The channel's boundary conditions, on the whole mesh and on a part alike:
/// `U` fixedValue 0 on every face of a `PatchKind::Wall` patch, and NOTHING
/// else written - the cyclic pair's faces and every processor face keep the
/// `Cyclic` kind `GpuVectorField::zeros` seeds them with (`fr` 0), and `p`,
/// `k`, `epsilon`/`omega` and `nut` keep their zeroGradient walls: the wall
/// functions own the wall cells, exactly as `cavity_bcs` writes walls only.
pub fn channel_bcs(gpu: &Gpu, hm: &HostMesh, s: &mut Simple<'_>) -> Result<()> {
    let mut ukind: Vec<Label> = gpu.download(&s.u().bc_kind)?;
    let mut ufr: Vec<Scalar> = gpu.download(&s.u().fr)?;
    let mut urv: Vec<Vec3> = gpu.download(&s.u().ref_value)?;
    let urg: Vec<Vec3> = gpu.download(&s.u().ref_grad)?;
    if ukind.len() != hm.n_boundary_faces {
        return Err(Error::Config(format!(
            "distflow: channel_bcs was given a host mesh of {} boundary faces \
             and a Simple of {}",
            hm.n_boundary_faces,
            ukind.len()
        )));
    }
    for (i, &k) in hm.b_kind.iter().enumerate() {
        if k == PatchKind::Wall as Label {
            ukind[i] = BcKind::FixedValue as Label;
            ufr[i] = 1.0;
            urv[i] = Vec3::ZERO;
        }
    }
    let u = s.u_mut();
    gpu.write(&mut u.bc_kind, &ukind)?;
    gpu.write(&mut u.fr, &ufr)?;
    gpu.write(&mut u.ref_value, &urv)?;
    gpu.write(&mut u.ref_grad, &urg)
}

/// The initial value of a transported scalar, `f` AND `f0` - a steady run
/// reads neither, and writing both is the one form of the seed.
fn write_initial_scalar(gpu: &Gpu, f: &mut GpuScalarField, v: Scalar, n: usize) -> Result<()> {
    gpu.write(&mut f.f, &vec![v; n])?;
    gpu.write(&mut f.f0, &vec![v; n])
}

/// One channel model, as the shared outer loop sees it: the two transport
/// equations behind one enum, so the loop is written once and the two models
/// cannot drift apart in what one outer iteration does.
enum ChannelTurb<'m> {
    KE(Box<KEpsilon<'m>>),
    SST(Box<KOmegaSst<'m>>),
}

impl<'m> ChannelTurb<'m> {
    fn nut(&self) -> &GpuScalarField {
        match self {
            Self::KE(m) => m.nut(),
            Self::SST(m) => m.nut(),
        }
    }

    fn correct(
        &mut self,
        gpu: &Gpu,
        flow: &FlowState<'_>,
    ) -> Result<(SolverPerformance, SolverPerformance)> {
        match self {
            Self::KE(m) => m.correct(gpu, flow),
            Self::SST(m) => m.correct(gpu, flow),
        }
    }

    fn initialise(&mut self, gpu: &Gpu, flow: &FlowState<'_>) -> Result<()> {
        match self {
            Self::KE(m) => m.initialise(gpu, flow),
            Self::SST(m) => m.initialise(gpu, flow),
        }
    }

    fn k(&self) -> &GpuScalarField {
        match self {
            Self::KE(m) => m.k(),
            Self::SST(m) => m.k(),
        }
    }

    /// `epsilon` or `omega`, whichever the model transports.
    fn w(&self) -> &GpuScalarField {
        match self {
            Self::KE(m) => m.epsilon(),
            Self::SST(m) => m.omega(),
        }
    }
}

/// The outer loop both runners share: up to `outer` times
/// `Simple::correct_outer` then the model's `correct`, stopping when the four
/// initial residuals are all under [`CHANNEL_RESIDUAL`] (`stop`) or any of
/// them is not finite (`diverged`). Returns (iterations, converged, diverged)
/// and the last iteration's `[U, p, k, w]` residuals.
fn channel_loop(
    gpu: &Gpu,
    s: &mut Simple<'_>,
    backend: &mut dyn PressureBackend,
    turb: &mut ChannelTurb<'_>,
    t: &GpuScalarField,
    outer: usize,
    stop: bool,
) -> Result<(usize, bool, bool, [Scalar; 4])> {
    let mut iterations = 0usize;
    let mut converged = false;
    let mut diverged = false;
    let mut last = [0.0 as Scalar; 4];
    for _ in 0..outer {
        let perf = s.correct_outer(gpu, backend, turb.nut(), t, false)?;
        let flow = s.flow_state();
        let (w_perf, k_perf) = turb.correct(gpu, &flow)?;
        let u_res = perf
            .u
            .iter()
            .map(|q| q.initial_residual)
            .fold(0.0 as Scalar, Scalar::max);
        last = [
            u_res,
            perf.p.initial_residual,
            k_perf.initial_residual,
            w_perf.initial_residual,
        ];
        iterations += 1;
        if last.iter().any(|r| !r.is_finite()) {
            diverged = true;
            break;
        }
        if stop && last.iter().all(|&r| r < CHANNEL_RESIDUAL) {
            converged = true;
            break;
        }
    }
    Ok((iterations, converged, diverged, last))
}

/// The run's report, OWNED cells only: `U`, `k`, `epsilon`/`omega` and `nut`
/// of every cell this rank owns, local order.
#[allow(clippy::too_many_arguments)]
fn channel_report(
    gpu: &Gpu,
    hm: &HostMesh,
    s: &Simple<'_>,
    turb: &ChannelTurb<'_>,
    global_cell: Vec<usize>,
    iterations: usize,
    converged: bool,
    diverged: bool,
    residuals: [Scalar; 4],
) -> Result<ChannelRun> {
    let n = hm.n_cells;
    Ok(ChannelRun {
        global_cell,
        u: gpu.download(&s.u().f)?[..n].to_vec(),
        k: gpu.download(&turb.k().f)?[..n].to_vec(),
        w: gpu.download(&turb.w().f)?[..n].to_vec(),
        nut: gpu.download(&turb.nut().f)?[..n].to_vec(),
        iterations,
        converged,
        diverged,
        residuals,
    })
}

/// Zero gravity - the case is forced convection only.
fn channel_buoyancy() -> BuoyancyCoeffs {
    BuoyancyCoeffs {
        g: Vec3::ZERO,
        ..BuoyancyCoeffs::default()
    }
}

/// The wall-function face flags the fixture gives every model: the same
/// `PatchKind::Wall` answer for both sets (the constrained cells and the
/// `nut` faces), because a cartesian duct's walls are walls for everything.
fn channel_wall_faces(hm: &HostMesh) -> (WallFaces, NutRoughness) {
    let on: Vec<bool> = hm
        .b_kind
        .iter()
        .map(|&k| k == PatchKind::Wall as Label)
        .collect();
    (
        WallFaces { constrained_cells: on.clone(), nut: on },
        NutRoughness::none(hm.n_boundary_faces),
    )
}

/// The channel on the WHOLE mesh, on one card - the serial reference. No
/// route, no exchange: the calls `Simple`, `KEpsilon` and `KOmegaSst` have
/// always made.
pub fn channel_serial(
    gpu: &Gpu,
    case: &ChannelCase,
    model: ChannelModel,
    outer: usize,
    stop: bool,
) -> Result<ChannelRun> {
    let hm = &case.hm;
    let m = GpuMesh::upload(gpu, hm)?;
    let mut s = Simple::new(gpu, hm, &m, channel_controls(&case.cc), channel_buoyancy())?;
    let mut backend = PbicgstabBackend::new(channel_controls(&case.cc).p_solver);
    backend.setup(gpu, hm, &m, &SystemProbe::default())?;
    channel_bcs(gpu, hm, &mut s)?;
    for spec in &case.sources {
        if spec.field == "U" {
            let src = spec.build(gpu, hm)?;
            s.momentum_mut().sources_mut().push(src);
        }
    }
    gpu.write(&mut s.u_mut().f, &vec![Vec3::new(CHANNEL_U0, 0.0, 0.0); m.n_local()])?;
    s.initialise(gpu)?;

    let fk = FieldKernels::new(gpu)?;
    let t = cavity_t(gpu, &m, &fk)?;

    let (wall_faces, roughness) = channel_wall_faces(hm);
    let n = m.n_cells;
    let mut turb = match model {
        ChannelModel::KEpsilon => {
            let mut ke = KEpsilon::new(
                gpu,
                hm,
                &m,
                KEpsilonCoeffs::default(),
                channel_turbulence(&case.cc),
                case.cc.wall,
                &wall_faces,
                &roughness,
            )?;
            write_initial_scalar(gpu, ke.k_mut(), CHANNEL_K0, n)?;
            write_initial_scalar(gpu, ke.epsilon_mut(), CHANNEL_EPS0, n)?;
            ChannelTurb::KE(Box::new(ke))
        }
        ChannelModel::KOmegaSst => {
            // Tucker's Poisson wall distance of SPEC-LIT 6.6, on the whole
            // mesh, once.
            let y_host = channel_wall_distance(gpu, hm, &case.cc)?;
            let mut y = gpu.zeros(n)?;
            gpu.write(&mut y, &y_host)?;
            let mut sst = KOmegaSst::new(
                gpu,
                hm,
                &m,
                KOmegaSstCoeffs::default(),
                channel_turbulence(&case.cc),
                case.cc.wall,
                &wall_faces,
                &y,
                &roughness,
            )?;
            write_initial_scalar(gpu, sst.k_mut(), CHANNEL_K0, n)?;
            write_initial_scalar(gpu, sst.omega_mut(), CHANNEL_OMEGA0, n)?;
            ChannelTurb::SST(Box::new(sst))
        }
    };
    turb.initialise(gpu, &s.flow_state())?;

    let (iterations, converged, diverged, residuals) =
        channel_loop(gpu, &mut s, &mut backend, &mut turb, &t, outer, stop)?;
    channel_report(
        gpu,
        hm,
        &s,
        &turb,
        (0..hm.n_cells).collect(),
        iterations,
        converged,
        diverged,
        residuals,
    )
}

/// The channel on rank `comm.rank()`'s part of `dec` - the decomposed run:
/// `cavity_part`'s Simple (SPMD momentum route, SPMD pressure backend,
/// exchange) plus, for the model, its own [`SpmdMomentumRoute`] through
/// `set_route` and its own [`RankCtx`] through `set_exchange`, BEFORE the
/// initial `k`/`epsilon`/`omega` are written and `initialise` runs - they are
/// collective, and every rank constructs them in the same order. The wall
/// distance comes in whole-mesh (`y_whole`) and is split by
/// [`split_cells`]; no part ever solves for it.
pub fn channel_part(
    gpu: &Gpu,
    comm: &dyn Comm,
    dec: &Decomposition,
    case: &ChannelCase,
    model: ChannelModel,
    y_whole: &[Scalar],
    outer: usize,
    stop: bool,
) -> Result<ChannelRun> {
    let part = &dec.parts[comm.rank()];
    let hm = &part.mesh;
    let gm = GpuMesh::upload_with_halo(gpu, hm, part.n_halo)?;
    let mut s = Simple::new(gpu, hm, &gm, channel_controls(&case.cc), channel_buoyancy())?;
    s.set_momentum_route(Some(Box::new(SpmdMomentumRoute::new(
        gpu,
        comm,
        dec,
        DistReduce::Exact,
    )?)));
    s.set_exchange(Some(Box::new(RankCtx::new(gpu, comm, part, &gm)?)))?;

    let mut backend = SpmdPressureBackend::new(
        gpu,
        comm,
        dec,
        channel_controls(&case.cc).p_solver,
        DistReduce::Exact,
    )?;
    backend.setup(gpu, hm, &gm, &SystemProbe::default())?;

    channel_bcs(gpu, hm, &mut s)?;
    for spec in &case.sources {
        if spec.field == "U" {
            let src = spec.build(gpu, hm)?;
            s.momentum_mut().sources_mut().push(src);
        }
    }
    gpu.write(&mut s.u_mut().f, &vec![Vec3::new(CHANNEL_U0, 0.0, 0.0); gm.n_local()])?;
    s.initialise(gpu)?;

    let fk = FieldKernels::new(gpu)?;
    let t = cavity_t(gpu, &gm, &fk)?;

    let (wall_faces, roughness) = channel_wall_faces(hm);
    let n_local = gm.n_local();
    let mut turb = match model {
        ChannelModel::KEpsilon => {
            let mut ke = KEpsilon::new(
                gpu,
                hm,
                &gm,
                KEpsilonCoeffs::default(),
                channel_turbulence(&case.cc),
                case.cc.wall,
                &wall_faces,
                &roughness,
            )?;
            ke.set_route(Some(Box::new(SpmdMomentumRoute::new(
                gpu,
                comm,
                dec,
                DistReduce::Exact,
            )?)));
            ke.set_exchange(gpu, hm, Some(Box::new(RankCtx::new(gpu, comm, part, &gm)?)))?;
            write_initial_scalar(gpu, ke.k_mut(), CHANNEL_K0, n_local)?;
            write_initial_scalar(gpu, ke.epsilon_mut(), CHANNEL_EPS0, n_local)?;
            ChannelTurb::KE(Box::new(ke))
        }
        ChannelModel::KOmegaSst => {
            // The WHOLE mesh's wall distance, split to this part's owned
            // cells - never solved here (SPEC-LIT 139.6).
            let mut y = gpu.zeros(part.mesh.n_cells)?;
            gpu.write(&mut y, &split_cells(y_whole, part))?;
            let mut sst = KOmegaSst::new(
                gpu,
                hm,
                &gm,
                KOmegaSstCoeffs::default(),
                channel_turbulence(&case.cc),
                case.cc.wall,
                &wall_faces,
                &y,
                &roughness,
            )?;
            sst.set_route(Some(Box::new(SpmdMomentumRoute::new(
                gpu,
                comm,
                dec,
                DistReduce::Exact,
            )?)));
            sst.set_exchange(gpu, hm, Some(Box::new(RankCtx::new(gpu, comm, part, &gm)?)))?;
            write_initial_scalar(gpu, sst.k_mut(), CHANNEL_K0, n_local)?;
            write_initial_scalar(gpu, sst.omega_mut(), CHANNEL_OMEGA0, n_local)?;
            ChannelTurb::SST(Box::new(sst))
        }
    };
    turb.initialise(gpu, &s.flow_state())?;

    let (iterations, converged, diverged, residuals) =
        channel_loop(gpu, &mut s, &mut backend, &mut turb, &t, outer, stop)?;
    channel_report(
        gpu,
        hm,
        &s,
        &turb,
        part.global_cell[..hm.n_cells].iter().map(|&g| g as usize).collect(),
        iterations,
        converged,
        diverged,
        residuals,
    )
}
