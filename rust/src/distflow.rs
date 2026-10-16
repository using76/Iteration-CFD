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
//! Provenance: ORIGINAL - the exchange points are this project's own design
//! (SPEC-LIT 139.2, PAR-10a) over the SPMD exchange of 138.3 (PAR-02) and
//! the boundary evaluation this crate already ships; no kernel is launched
//! here and no existing numerics change. No external source. No GPL-licensed
//! source was consulted.

use crate::comm::Comm;
use crate::decompose::PartMesh;
use crate::device::{DevBuf, Gpu};
use crate::error::{Error, Result};
use crate::exactsum::SpmdReduction;
use crate::field::{GpuScalarField, GpuVectorField};
use crate::field_ops::{self, FieldKernels};
use crate::halo::SpmdHalo;
use crate::mesh::GpuMesh;
use crate::{Tensor, Vec3};

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


