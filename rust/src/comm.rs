// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! The SPMD transport: how one rank of a parallel run talks to the others.
//!
//! In an SPMD run every rank owns one part of the mesh and runs the same
//! program. It reaches the other ranks through exactly one object, a
//! [`Comm`], and every operation on it moves bytes and does no arithmetic on
//! them. A ghost cell therefore holds its owner's value bit for bit whatever
//! the transport, and the collectives stay gathers, never reductions
//! (SPEC-LIT §138, §138.1.1).
//!
//! The trait has the operations a halo exchange and an exact reduction need
//! and nothing else:
//!
//! | operation | what it does |
//! |---|---|
//! | [`Comm::rank`], [`Comm::size`] | who this rank is, and how many there are |
//! | [`Comm::barrier`] | return only after every rank has entered |
//! | [`Comm::exchange`] | send byte messages to some peers, fill buffers from some peers |
//! | [`Comm::allgather`] | every rank's equal-length block, in rank order |
//! | [`Comm::allgatherv`] | every rank's block, lengths free, in rank order |
//! | [`Comm::bcast`] | one rank's bytes to every rank |
//!
//! [`ThreadComm`] is the transport `cargo test` runs on: N ranks as N threads
//! of one process, so a one-GPU machine with no MPI can still exercise a
//! multi-rank program. Another transport (MPI, a later unit) implements the
//! same trait, so the program above it is one code path. The rules both must
//! keep are R1-R10 of SPEC-LIT §138.1.2, restated on [`Comm`] below.
//!
//! This module is host only: no `Gpu`, no device memory and no `Scalar`, so
//! the default build and `--features single` compile and run the same code.
//!
//! Provenance: ORIGINAL - the design is ours (docs/17 PAR-01); the semantics
//! are written from the MPI-4.1 report (MPI Forum, 2023), sections 4.2.3,
//! 4.5, 7.1, 7.3, 7.4, 7.7, 7.14 and 10.1.2.1, read as a specification. No
//! MPI implementation's source was opened. No GPL-licensed source was
//! consulted.

pub mod thread;

#[cfg(test)]
mod tests;

pub use thread::ThreadComm;

use crate::error::Result;
use std::time::Duration;

/// The largest tag a caller may use in [`Comm::exchange`].
///
/// MPI-4.1 section 10.1.2.1 guarantees that the upper bound of tag values
/// (`MPI_TAG_UB`) is at least 32767, so a tag this transport accepts is valid
/// on every MPI.
pub const MAX_TAG: u32 = 32767;

/// How long a [`ThreadComm`] call waits for a message before it fails by
/// name.
pub const DEFAULT_TIMEOUT: Duration = Duration::from_secs(120);

/// One rank's view of an SPMD world.
///
/// Every failure is `Error::Config` and its message starts `comm: `. After
/// any `Err` the world is spent (R10): no later call on it is specified.
///
/// The trait is object-safe, so a program can take `&dyn Comm`.
pub trait Comm: Send {
    /// This rank's number, in `0..size()` (R1). Fixed for the life of the
    /// world.
    fn rank(&self) -> usize;

    /// The number of ranks, at least 1 (R1). One rank is not a special case:
    /// every operation works with the one rank alone.
    fn size(&self) -> usize;

    /// The transport's name, for logs and reports (`"threads"`).
    fn transport(&self) -> &'static str;

    /// Return only after every rank has entered the barrier (R8, MPI-4.1
    /// section 7.3).
    fn barrier(&self) -> Result<()>;

    /// Send `sends[i].1` to rank `sends[i].0` under `tag`, and fill each
    /// `recvs[j].1` with the message from rank `recvs[j].0` under `tag` (R2).
    ///
    /// * A peer may be the calling rank itself; a message to self is
    ///   delivered like any other. The order of entries in either list does
    ///   not change any result.
    /// * Messages match on the envelope `(src, dst, tag)` and are FIFO within
    ///   one envelope; arrival order across envelopes is never consulted
    ///   (R3, MPI-4.1 sections 4.2.3 and 4.5).
    /// * Sends are buffered, so the call returns once every entry in `recvs`
    ///   is filled and never waits for a receiver. Consistent lists cannot
    ///   deadlock (R4).
    /// * A message's length must equal its buffer's length exactly (R5),
    ///   which is stricter than MPI.
    /// * A tag above [`MAX_TAG`], a peer outside `0..size()` and a peer
    ///   listed twice in one list fail before anything is sent (R9).
    ///
    /// Exchange traffic never matches collective traffic, whatever the tag
    /// (R7).
    fn exchange(
        &self,
        tag: u32,
        sends: &[(usize, &[u8])],
        recvs: &mut [(usize, &mut [u8])],
    ) -> Result<()>;

    /// Every rank passes `mine` of the same length `n` and `all` of length
    /// `n * size()`; afterwards `all[q*n .. (q+1)*n]` is rank q's `mine`, on
    /// every rank (R8, MPI-4.1 section 7.7).
    ///
    /// All ranks must call the collectives in the same order, and the block
    /// length is part of what is checked (R6).
    fn allgather(&self, mine: &[u8], all: &mut [u8]) -> Result<()>;

    /// Return `size()` vectors in rank order, where entry q is rank q's
    /// `mine`. The lengths may differ and zero is allowed; every rank
    /// receives identical vectors (R8, MPI-4.1 section 7.7).
    fn allgatherv(&self, mine: &[u8]) -> Result<Vec<Vec<u8>>>;

    /// Every rank passes `buf` of the same length and the same `root`;
    /// afterwards every `buf` holds the root's bytes, and the root's own
    /// `buf` is unchanged (R8, MPI-4.1 section 7.4).
    fn bcast(&self, root: usize, buf: &mut [u8]) -> Result<()>;
}
