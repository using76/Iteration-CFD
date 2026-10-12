// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! The SPMD exact reduction: §72's accumulator, one part per rank
//! (SPEC-LIT §138.4).
//!
//! [`ExactReduction`](super::ExactReduction) holds every part in one process
//! and gathers the per-part results with a device-to-device copy (§72.4,
//! step 1). In an SPMD run a rank holds **one** part and reaches the others
//! only through a [`Comm`](crate::comm::Comm), so the gather goes through
//! the host: the rank's value is downloaded, handed to `Comm::allgather`,
//! and the `n_ranks` values that come back are written into the gathered
//! buffer the in-process form fills by copy. The kernels are §72's own,
//! unchanged, and so is the order of the gathered values: slot q is rank
//! q's, as slot q was part q's. The per-rank device work is exactly what
//! `ExactReduction` does for ONE part - same kernels, same arguments, same
//! launch geometry - and the cross-rank combine is exactly its cross-part
//! combine over `n_ranks` values in rank order. So every rank holds the
//! same answer, and it is the in-process answer bit for bit.
//!
//! An exact reduction is two allgathers - the limbs are cut against the
//! anchor, so the anchor must be global first (§72.3) - and a maximum or a
//! gathered partial is one. The wire carries the values' in-memory
//! representation in native byte order: the term count as a `u64`, a
//! maximum or a gathered partial as the rank's `Scalar` widened on the host
//! to `Acc` (8 bytes, exact in BOTH builds - SPEC-LIT §118.1), the limbs as
//! `EXACT_WORDS` `i64` (40 bytes, the last holding the non-finite term
//! count). Both host transfers go through pageable memory
//! (`Gpu::download`, `Gpu::write`), which the driver completes before the
//! call returns, so there is no separate stream wait as in the halo
//! exchange (§138.3). A reduction allocates nothing on the device: every
//! buffer is allocated once in [`SpmdReduction::new`].
//!
//! Not capturable: host round trips, one allgather per collective, so a
//! reduction inside `Gpu::capture` is refused by name. In one process
//! `ExactReduction` stays the capturable form.
//!
//! Provenance: ORIGINAL - the SPMD form is this project's own design
//! (SPEC-LIT §138.4, PAR-03) on the accumulator of §72 unchanged; it moves
//! the gather through `Comm::allgather` and consults no external source.
//! No GPL-licensed source was consulted.

use cudarc::driver::{LaunchConfig, PushKernelArg};

use crate::comm::Comm;
use crate::device::{DevBuf, Gpu};
use crate::error::{Error, Result};
use crate::solver::{self, one_block, reduce_geometry, to_label, SolverKernels};
use crate::{Acc, Label, Scalar};

use super::{one_thread, ExactKernels, EXACT_K, EXACT_MAX_TERMS, EXACT_W, EXACT_WORDS};

/// The host half of the wire: the rank's `Scalar` result widened to `Acc`.
/// Exact, because every float is a double (SPEC-LIT §118.1); in the f64
/// build the two are the same type and the widening is the identity.
#[cfg(feature = "single")]
fn widen(s: Scalar) -> Acc {
    s as Acc
}

/// The host half of the wire, f64 build: the two are the same type.
#[cfg(not(feature = "single"))]
fn widen(s: Scalar) -> Acc {
    s
}

/// The SPMD form of §72's reduction: one rank, one part, one [`Comm`].
///
/// Built once per rank, at set-up. Nothing in a reduction allocates on the
/// device and every failure is `Error::Config` starting `exactsum: `.
/// `Send`, so a rank can move it into its thread, like the SPMD halo.
pub struct SpmdReduction {
    /// Every entry point in `cuda/exactsum.cu`, resolved once - §72's own.
    k: ExactKernels,
    /// The rank `new` ran on.
    rank: usize,
    /// `comm.size()` seen by `new`.
    n_ranks: usize,
    /// This rank's term count.
    n: usize,
    /// Every rank's term count, rank order - what the allgather in `new`
    /// gathered and every rank checked alike, before any device allocation.
    counts: Vec<usize>,
    /// `Comm::allgather` calls this handle has made, `new`'s included.
    allgathers: u64,
    /// This rank's stage-one scalar partials - the maxima, and the plain
    /// sums of the gathered-partial construction. `Acc` in BOTH builds
    /// (SPEC-LIT §118.1).
    scalar_partials: DevBuf<Acc>,
    /// `[nblocks * EXACT_WORDS]` this rank's stage-one limb partials,
    /// `EXACT_K` per block.
    limb_partials: DevBuf<i64>,
    /// `[1]` the slot the rank's stage two writes into: the allgather's
    /// source, as `scalar_slot` is the in-process copy's source.
    scalar_slot: DevBuf<Scalar>,
    /// `[EXACT_WORDS]` the same, for this rank's limb total.
    limb_slot: DevBuf<i64>,
    /// `[n_ranks]` the gathered per-rank scalars, `Acc` in BOTH builds
    /// (SPEC-LIT §118.1) - the stage two that combines them adds in `ofacc`.
    part_scalar: DevBuf<Acc>,
    /// `[n_ranks * EXACT_WORDS]` the gathered per-rank limb totals.
    part_limbs: DevBuf<i64>,
    /// `[1]` the global anchor, `max|t|`.
    amax: DevBuf<Scalar>,
    /// `[EXACT_WORDS]` the global limb total.
    total: DevBuf<i64>,
    /// `[1]` the answer.
    out: DevBuf<Scalar>,
}

impl SpmdReduction {
    /// Build the rank's reduction over the `n` terms the rank owns. `new`
    /// with a world of one rank is not a special case: one allgather of one
    /// count, one combine over one value - the serial path IS the parallel
    /// path (§72.4).
    ///
    /// The counts are gathered FIRST and checked AFTERWARDS, on every rank
    /// alike (§138.4): every rank sees every count, so every rank reaches
    /// the same refusal and none is left waiting in a collective. Both
    /// checks run before any device allocation.
    pub fn new(gpu: &Gpu, comm: &dyn Comm, n: usize) -> Result<Self> {
        let rank = comm.rank();
        let n_ranks = comm.size();
        let mut counts_bytes = vec![0u8; 8 * n_ranks];
        comm.allgather(&(n as u64).to_ne_bytes(), &mut counts_bytes)?;
        let allgathers = 1u64;
        let mut counts = Vec::with_capacity(n_ranks);
        for q in 0..n_ranks {
            let mut b = [0u8; 8];
            b.copy_from_slice(&counts_bytes[8 * q..8 * q + 8]);
            counts.push(u64::from_ne_bytes(b) as usize);
        }
        for (q, &c) in counts.iter().enumerate() {
            if c > Label::MAX as usize {
                return Err(Error::Config(format!(
                    "exactsum: rank {q}'s count of {c} terms does not fit a \
                     Label, which every stage-one launch passes as its term \
                     count (SPEC-LIT §72.2); no device memory was touched"
                )));
            }
        }
        let total_terms: usize = counts.iter().sum();
        if total_terms > EXACT_MAX_TERMS {
            return Err(Error::Config(format!(
                "exactsum: {total_terms} terms over {n_ranks} rank(s) \
                 exceeds the {EXACT_MAX_TERMS}-term limit of a {EXACT_W}-bit \
                 limb held in an i64 (SPEC-LIT §72.2). Narrow EXACT_W and \
                 widen EXACT_K, or reduce in groups"
            )));
        }
        let blocks = reduce_geometry(n.max(1)).1;
        Ok(Self {
            k: ExactKernels::new(gpu)?,
            rank,
            n_ranks,
            n,
            counts,
            allgathers,
            scalar_partials: gpu.zeros(blocks)?,
            limb_partials: gpu.zeros(blocks * EXACT_WORDS)?,
            scalar_slot: gpu.zeros(1)?,
            limb_slot: gpu.zeros(EXACT_WORDS)?,
            part_scalar: gpu.zeros(n_ranks)?,
            part_limbs: gpu.zeros(n_ranks * EXACT_WORDS)?,
            amax: gpu.zeros(1)?,
            total: gpu.zeros(EXACT_WORDS)?,
            out: gpu.zeros(1)?,
        })
    }

    /// The rank this handle belongs to - the rank `new` ran on.
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// The rank count `new` saw through the `Comm`.
    pub fn n_ranks(&self) -> usize {
        self.n_ranks
    }

    /// This rank's term count.
    pub fn n(&self) -> usize {
        self.n
    }

    /// Every rank's term count, in rank order, as `new` gathered them.
    pub fn counts(&self) -> &[usize] {
        &self.counts
    }

    /// The `Comm::allgather` calls this handle has made, `new`'s included.
    pub fn n_allgathers(&self) -> u64 {
        self.allgathers
    }

    /// The answer, still on the device.
    pub fn out(&self) -> &DevBuf<Scalar> {
        &self.out
    }

    /// The answer, on the host. One 8-byte transfer.
    pub fn value(&self, gpu: &Gpu) -> Result<Scalar> {
        Ok(gpu.download(&self.out)?[0])
    }

    /// The global anchor `max|t|` the last call used.
    pub fn anchor(&self, gpu: &Gpu) -> Result<Scalar> {
        Ok(gpu.download(&self.amax)?[0])
    }
    /// The global limb total the last exact call produced - `EXACT_WORDS`
    /// values, the last of which is the non-finite term count. Exposed
    /// because the limbs, not the converted float, are the object that is
    /// order-free (§72).
    pub fn limbs(&self, gpu: &Gpu) -> Result<Vec<i64>> {
        gpu.download(&self.total)
    }

    /// How many terms of the last exact call were an infinity or a NaN.
    ///
    /// Non-zero means the answer is a NaN by construction, and that the
    /// field was already poisoned before this module saw it.
    pub fn non_finite_terms(&self, gpu: &Gpu) -> Result<i64> {
        Ok(self.limbs(gpu)?[EXACT_K])
    }

    // ----------------------------------------------------------------------
    //  The refusals of §138.4's table, in the table's order
    // ----------------------------------------------------------------------

    /// The four run-time checks, all before anything is launched or sent:
    /// the capture, the world size, the rank, then every buffer's length.
    /// A short buffer is one rank's fault and the other ranks cannot see
    /// it: after any `Err` the world is spent (R10).
    fn check(&self, gpu: &Gpu, comm: &dyn Comm, bufs: &[(&str, usize)]) -> Result<()> {
        if gpu.is_capturing() {
            return Err(Error::Config(
                "exactsum: an SPMD exact reduction cannot run inside a \
                 CUDA-graph capture: every collective copies the rank's \
                 value to the host, hands it to a Comm::allgather and \
                 writes the gathered values back - host round trips, one \
                 per collective, and there is nothing to record. \
                 Alternative: ExactReduction, whose gather is a \
                 device-to-device copy and is the one-process form \
                 (SPEC-LIT 72.4, 138.4)"
                    .to_string(),
            ));
        }
        let size = comm.size();
        if size != self.n_ranks {
            return Err(Error::Config(format!(
                "exactsum: this world has {size} ranks, but the reduction \
                 was built for {} ranks; a gathered combine over the wrong \
                 rank count would silently answer the wrong sum",
                self.n_ranks
            )));
        }
        let rank = comm.rank();
        if rank != self.rank {
            return Err(Error::Config(format!(
                "exactsum: this handle is rank {rank}, but the reduction \
                 belongs to rank {}, and the gathered slots are ordered by \
                 rank",
                self.rank
            )));
        }
        for (name, len) in bufs {
            if *len < self.n {
                return Err(Error::Config(format!(
                    "exactsum: `{name}` holds {len} values and this rank \
                     owns {} terms. A buffer may be LONGER than the rank's \
                     term count - a decomposed field is n_cells + n_halo \
                     long (SPEC-LIT §71.1) - but never shorter",
                    self.n
                )));
            }
        }
        Ok(())
    }

    // ----------------------------------------------------------------------
    //  The wire, and the pieces every operation shares
    // ----------------------------------------------------------------------

    /// The collective for one scalar: download the rank's `Scalar` result,
    /// widen it on the host to `Acc`, allgather the 8 bytes, and write the
    /// `n_ranks` doubles that come back into `part_scalar` in rank order.
    /// The in-process form moves the same bits device to device; neither
    /// move does arithmetic, which is what makes both exact (§72.4).
    fn allgather_scalar(&mut self, gpu: &Gpu, comm: &dyn Comm) -> Result<()> {
        let mine = widen(gpu.download(&self.scalar_slot)?[0]).to_ne_bytes();
        let mut all = vec![0u8; 8 * self.n_ranks];
        comm.allgather(&mine, &mut all)?;
        self.allgathers += 1;
        let mut vals = Vec::with_capacity(self.n_ranks);
        for q in 0..self.n_ranks {
            let mut b = [0u8; 8];
            b.copy_from_slice(&all[8 * q..8 * q + 8]);
            vals.push(Acc::from_ne_bytes(b));
        }
        gpu.write(&mut self.part_scalar, &vals)
    }

    /// The collective for one limb set: download the rank's `EXACT_WORDS`
    /// `i64` words, allgather the 40 bytes, and write the `n_ranks` limb
    /// sets that come back into `part_limbs` in rank order.
    fn allgather_limbs(&mut self, gpu: &Gpu, comm: &dyn Comm) -> Result<()> {
        let got = gpu.download(&self.limb_slot)?;
        let mut mine = vec![0u8; EXACT_WORDS * 8];
        for (dst, v) in mine.chunks_exact_mut(8).zip(got.iter()) {
            dst.copy_from_slice(&v.to_ne_bytes());
        }
        let mut all = vec![0u8; EXACT_WORDS * 8 * self.n_ranks];
        comm.allgather(&mine, &mut all)?;
        self.allgathers += 1;
        let mut vals = Vec::with_capacity(self.n_ranks * EXACT_WORDS);
        for q in 0..self.n_ranks * EXACT_WORDS {
            let mut b = [0u8; 8];
            b.copy_from_slice(&all[8 * q..8 * q + 8]);
            vals.push(i64::from_ne_bytes(b));
        }
        gpu.write(&mut self.part_limbs, &vals)
    }

    /// This rank's launch geometry, or `None` when it owns nothing. A
    /// zero-block grid is an illegal launch configuration, so a rank with
    /// no terms is answered with a memset rather than a kernel - the same
    /// rule `device_sum` follows for an empty reduction, and the same
    /// contribution an empty part makes in §72.
    fn geometry(&self) -> Option<(LaunchConfig, usize, Label)> {
        if self.n == 0 {
            return None;
        }
        let (cfg, nparts) = reduce_geometry(self.n);
        // `new` proved every rank's count fits a label, so this cannot fail.
        Some((cfg, nparts, self.n as Label))
    }

    /// The rank's own maximum of |t| - `solMaxMagStage1` and its stage two,
    /// exactly what `anchor_of_magnitude` does for one part. A rank owning
    /// nothing launches nothing and its `scalar_slot` reads zero.
    fn max_mag_rank(
        &mut self,
        gpu: &Gpu,
        sol: &SolverKernels,
        x: &DevBuf<Scalar>,
    ) -> Result<()> {
        let Self { scalar_slot, scalar_partials, n, .. } = self;
        solver::device_max_mag(gpu, sol, scalar_slot, x, scalar_partials, *n)
    }

    /// Stage two of this rank's own maximum - `solMaxStage2`, unchanged -
    /// into `scalar_slot`. The in-process form runs the same kernel per
    /// part and then moves the bytes; the move is what `allgather_scalar`
    /// replaces.
    fn stage_two_max(&mut self, gpu: &Gpu, sol: &SolverKernels, nparts: usize) -> Result<()> {
        let npl = to_label(nparts)?;
        let Self { scalar_slot, scalar_partials, .. } = self;
        unsafe {
            gpu.stream()
                .launch_builder(&sol.max2)
                .arg(&mut *scalar_slot)
                .arg(&*scalar_partials)
                .arg(&npl)
                .launch(one_block())?;
        }
        Ok(())
    }

    /// Stage two of this rank's own limb sum - `exCombine`, the same kernel
    /// the cross-rank combine runs; only the length differs (§72.4).
    fn combine_rank_limbs(&mut self, gpu: &Gpu, nparts: usize) -> Result<()> {
        let npl = to_label(nparts)?;
        let Self { k, limb_slot, limb_partials, .. } = self;
        unsafe {
            gpu.stream()
                .launch_builder(&k.combine)
                .arg(&mut *limb_slot)
                .arg(&*limb_partials)
                .arg(&npl)
                .launch(one_block())?;
        }
        Ok(())
    }

    /// The gathered per-rank maxima -> the one global anchor. `solMaxStage2`,
    /// one block, fixed shape - and exact whatever the order, because a
    /// maximum is (§72.3).
    fn finish_anchor(&mut self, gpu: &Gpu, sol: &SolverKernels) -> Result<()> {
        let np = to_label(self.n_ranks)?;
        let Self { amax, part_scalar, .. } = self;
        unsafe {
            gpu.stream()
                .launch_builder(&sol.max2)
                .arg(&mut *amax)
                .arg(&*part_scalar)
                .arg(&np)
                .launch(one_block())?;
        }
        Ok(())
    }

    /// The gathered per-rank limb totals -> one limb total -> one float.
    /// The same `exCombine` that finished each rank, over `P` values instead
    /// of over the blocks, then `exToScalar` with `offset` added once - the
    /// in-process form's `finish`, over `n_ranks` slots instead of parts.
    fn finish(&mut self, gpu: &Gpu, offset: Scalar) -> Result<()> {
        let np = to_label(self.n_ranks)?;
        {
            let Self { k, total, part_limbs, .. } = self;
            unsafe {
                gpu.stream()
                    .launch_builder(&k.combine)
                    .arg(&mut *total)
                    .arg(&*part_limbs)
                    .arg(&np)
                    .launch(one_block())?;
            }
        }
        let Self { k, out, total, amax, .. } = self;
        unsafe {
            gpu.stream()
                .launch_builder(&k.to_scalar)
                .arg(&mut *out)
                .arg(&*total)
                .arg(&*amax)
                .arg(&offset)
                .launch(one_thread())?;
        }
        Ok(())
    }

    /// The gathered-partial combine: `solSumStage2` over `P` doubles,
    /// `offset` added once - the in-process form's `finish_gathered`, over
    /// `n_ranks` slots. The gathered buffer is `Acc` in BOTH builds, so in
    /// the f32 build the cross-rank sum does not re-round through float.
    fn finish_gathered(&mut self, gpu: &Gpu, sol: &SolverKernels, offset: Scalar) -> Result<()> {
        let np = to_label(self.n_ranks)?;
        let Self { out, part_scalar, .. } = self;
        unsafe {
            gpu.stream()
                .launch_builder(&sol.sum2)
                .arg(&mut *out)
                .arg(&*part_scalar)
                .arg(&np)
                .arg(&offset)
                .launch(one_block())?;
        }
        Ok(())
    }

    /// The anchor from `max|x|`, one allgather: the rank's own maximum,
    /// then the global combine. §72's `anchor_of_magnitude` for one part.
    fn anchor_of_magnitude(
        &mut self,
        gpu: &Gpu,
        sol: &SolverKernels,
        comm: &dyn Comm,
        x: &DevBuf<Scalar>,
    ) -> Result<()> {
        self.max_mag_rank(gpu, sol, x)?;
        self.allgather_scalar(gpu, comm)?;
        self.finish_anchor(gpu, sol)
    }


    // ----------------------------------------------------------------------
    //  Partition-invariant: the exact accumulator
    // ----------------------------------------------------------------------

    /// `out` = the sum of every rank's `sum_{i < n_r} x_r[i]` - bit for bit
    /// `ExactReduction::sum` over the same parts, whatever rank holds
    /// which. Two allgathers: the anchor, then the limbs.
    pub fn sum(
        &mut self,
        gpu: &Gpu,
        sol: &SolverKernels,
        comm: &dyn Comm,
        x: &DevBuf<Scalar>,
    ) -> Result<()> {
        self.check(gpu, comm, &[("x", x.len())])?;
        self.anchor_of_magnitude(gpu, sol, comm, x)?;

        // ---- this rank's limbs against the global anchor -------------------
        match self.geometry() {
            Some((cfg, nparts, nl)) => {
                {
                    let Self { k, limb_partials, amax, .. } = self;
                    unsafe {
                        gpu.stream()
                            .launch_builder(&k.sum1)
                            .arg(limb_partials)
                            .arg(x)
                            .arg(&nl)
                            .arg(&*amax)
                            .launch(cfg)?;
                    }
                }
                self.combine_rank_limbs(gpu, nparts)?;
            }
            None => gpu.fill_zero(&mut self.limb_slot)?,
        }
        self.allgather_limbs(gpu, comm)?;
        self.finish(gpu, 0.0)
    }

    /// `out` = the sum of every rank's `sum_{i < n_r} |x_r[i]|` - the
    /// residual measure of SPEC-LIT §8.4.
    pub fn sum_mag(
        &mut self,
        gpu: &Gpu,
        sol: &SolverKernels,
        comm: &dyn Comm,
        x: &DevBuf<Scalar>,
    ) -> Result<()> {
        self.check(gpu, comm, &[("x", x.len())])?;
        self.anchor_of_magnitude(gpu, sol, comm, x)?;

        // ---- this rank's limbs against the global anchor -------------------
        match self.geometry() {
            Some((cfg, nparts, nl)) => {
                {
                    let Self { k, limb_partials, amax, .. } = self;
                    unsafe {
                        gpu.stream()
                            .launch_builder(&k.sum_mag1)
                            .arg(limb_partials)
                            .arg(x)
                            .arg(&nl)
                            .arg(&*amax)
                            .launch(cfg)?;
                    }
                }
                self.combine_rank_limbs(gpu, nparts)?;
            }
            None => gpu.fill_zero(&mut self.limb_slot)?,
        }
        self.allgather_limbs(gpu, comm)?;
        self.finish(gpu, 0.0)
    }

    /// `out` = the dot product over every rank, exactly: `sum_r (a_r, b_r)`.
    pub fn dot(
        &mut self,
        gpu: &Gpu,
        sol: &SolverKernels,
        comm: &dyn Comm,
        a: &DevBuf<Scalar>,
        b: &DevBuf<Scalar>,
    ) -> Result<()> {
        self.check(gpu, comm, &[("a", a.len()), ("b", b.len())])?;

        // ---- the anchor: this rank's max|a b|, gathered -------------------
        match self.geometry() {
            Some((cfg, nparts, nl)) => {
                {
                    let Self { k, scalar_partials, .. } = self;
                    unsafe {
                        gpu.stream()
                            .launch_builder(&k.dot_max1)
                            .arg(scalar_partials)
                            .arg(a)
                            .arg(b)
                            .arg(&nl)
                            .launch(cfg)?;
                    }
                }
                self.stage_two_max(gpu, sol, nparts)?;
            }
            None => gpu.fill_zero(&mut self.scalar_slot)?,
        }
        self.allgather_scalar(gpu, comm)?;
        self.finish_anchor(gpu, sol)?;

        // ---- the limbs ----------------------------------------------------
        match self.geometry() {
            Some((cfg, nparts, nl)) => {
                {
                    let Self { k, limb_partials, amax, .. } = self;
                    unsafe {
                        gpu.stream()
                            .launch_builder(&k.dot1)
                            .arg(limb_partials)
                            .arg(a)
                            .arg(b)
                            .arg(&nl)
                            .arg(&*amax)
                            .launch(cfg)?;
                    }
                }
                self.combine_rank_limbs(gpu, nparts)?;
            }
            None => gpu.fill_zero(&mut self.limb_slot)?,
        }
        self.allgather_limbs(gpu, comm)?;
        self.finish(gpu, 0.0)
    }

    /// SPEC-LIT §8.4's normalisation factor,
    /// `sum ( |A.psi - A.xRef| + |b - A.xRef| ) + eps`, exactly.
    ///
    /// `eps` is added once, after the accumulator has been converted,
    /// exactly as the in-process form adds it - by each rank's own stage
    /// two it is added never, or the factor would be a function of the rank
    /// count (§72.5).
    pub fn norm_factor(
        &mut self,
        gpu: &Gpu,
        sol: &SolverKernels,
        comm: &dyn Comm,
        apsi: &DevBuf<Scalar>,
        b: &DevBuf<Scalar>,
        ax_ref: &DevBuf<Scalar>,
        eps: Scalar,
    ) -> Result<()> {
        self.check(
            gpu,
            comm,
            &[("Apsi", apsi.len()), ("b", b.len()), ("AxRef", ax_ref.len())],
        )?;

        // ---- the anchor: this rank's own stage one, gathered --------------
        match self.geometry() {
            Some((cfg, nparts, nl)) => {
                {
                    let Self { k, scalar_partials, .. } = self;
                    unsafe {
                        gpu.stream()
                            .launch_builder(&k.norm_factor_max1)
                            .arg(scalar_partials)
                            .arg(apsi)
                            .arg(b)
                            .arg(ax_ref)
                            .arg(&nl)
                            .launch(cfg)?;
                    }
                }
                self.stage_two_max(gpu, sol, nparts)?;
            }
            None => gpu.fill_zero(&mut self.scalar_slot)?,
        }
        self.allgather_scalar(gpu, comm)?;
        self.finish_anchor(gpu, sol)?;

        // ---- the limbs ----------------------------------------------------
        match self.geometry() {
            Some((cfg, nparts, nl)) => {
                {
                    let Self { k, limb_partials, amax, .. } = self;
                    unsafe {
                        gpu.stream()
                            .launch_builder(&k.norm_factor1)
                            .arg(limb_partials)
                            .arg(apsi)
                            .arg(b)
                            .arg(ax_ref)
                            .arg(&nl)
                            .arg(&*amax)
                            .launch(cfg)?;
                    }
                }
                self.combine_rank_limbs(gpu, nparts)?;
            }
            None => gpu.fill_zero(&mut self.limb_slot)?,
        }
        self.allgather_limbs(gpu, comm)?;
        self.finish(gpu, eps)
    }

    /// `out` = `max_r max_i |x_r[i]|`. A maximum over non-NaN operands is
    /// exactly order-independent, so every measure built on one is already
    /// partition-invariant; what it needs from here is only the gather.
    /// One allgather, and the anchor is the answer.
    pub fn max_mag(
        &mut self,
        gpu: &Gpu,
        sol: &SolverKernels,
        comm: &dyn Comm,
        x: &DevBuf<Scalar>,
    ) -> Result<()> {
        self.check(gpu, comm, &[("x", x.len())])?;
        self.anchor_of_magnitude(gpu, sol, comm, x)?;
        let Self { out, amax, .. } = self;
        gpu.stream()
            .memcpy_dtod(&amax.slice(0..1), &mut out.slice_mut(0..1))?;
        Ok(())
    }

    // ----------------------------------------------------------------------
    //  Run-invariant only: the gathered partial
    // ----------------------------------------------------------------------

    /// `out = fl(sigma_0 + ... + sigma_{P-1})` where `sigma_r` is rank r's
    /// own [`crate::solver::device_sum`] - §72.5's gathered partial, one
    /// allgather of 8 bytes. Reproducible run to run for a fixed world and
    /// **not** partition-invariant: `sigma_r` still depends on which terms
    /// landed on rank r.
    pub fn gathered_sum(
        &mut self,
        gpu: &Gpu,
        sol: &SolverKernels,
        comm: &dyn Comm,
        x: &DevBuf<Scalar>,
    ) -> Result<()> {
        self.check(gpu, comm, &[("x", x.len())])?;
        {
            let Self { scalar_slot, scalar_partials, n, .. } = self;
            solver::device_sum(gpu, sol, scalar_slot, x, scalar_partials, *n)?;
        }
        self.allgather_scalar(gpu, comm)?;
        self.finish_gathered(gpu, sol, 0.0)
    }

    /// The same construction for a dot product.
    pub fn gathered_dot(
        &mut self,
        gpu: &Gpu,
        sol: &SolverKernels,
        comm: &dyn Comm,
        a: &DevBuf<Scalar>,
        b: &DevBuf<Scalar>,
    ) -> Result<()> {
        self.check(gpu, comm, &[("a", a.len()), ("b", b.len())])?;
        {
            let Self { scalar_slot, scalar_partials, n, .. } = self;
            solver::device_dot(gpu, sol, scalar_slot, a, b, scalar_partials, *n)?;
        }
        self.allgather_scalar(gpu, comm)?;
        self.finish_gathered(gpu, sol, 0.0)
    }

    /// The same construction for a magnitude sum - the residual measure a
    /// Krylov solve tests convergence on.
    pub fn gathered_sum_mag(
        &mut self,
        gpu: &Gpu,
        sol: &SolverKernels,
        comm: &dyn Comm,
        x: &DevBuf<Scalar>,
    ) -> Result<()> {
        self.check(gpu, comm, &[("x", x.len())])?;
        {
            let Self { scalar_slot, scalar_partials, n, .. } = self;
            solver::device_sum_mag(gpu, sol, scalar_slot, x, scalar_partials, *n)?;
        }
        self.allgather_scalar(gpu, comm)?;
        self.finish_gathered(gpu, sol, 0.0)
    }

    /// The same construction for SPEC-LIT §8.4's normalisation factor.
    ///
    /// `eps` is added **once**, by the cross-rank combine, and not by each
    /// rank's own stage two - adding it `P` times would make the factor a
    /// function of the rank count (§72.5).
    pub fn gathered_norm_factor(
        &mut self,
        gpu: &Gpu,
        sol: &SolverKernels,
        comm: &dyn Comm,
        apsi: &DevBuf<Scalar>,
        b: &DevBuf<Scalar>,
        ax_ref: &DevBuf<Scalar>,
        eps: Scalar,
    ) -> Result<()> {
        self.check(
            gpu,
            comm,
            &[("Apsi", apsi.len()), ("b", b.len()), ("AxRef", ax_ref.len())],
        )?;
        match self.geometry() {
            Some((cfg, nparts, nl)) => {
                {
                    let Self { scalar_slot, scalar_partials, .. } = self;
                    unsafe {
                        gpu.stream()
                            .launch_builder(&sol.norm_factor1)
                            .arg(&mut *scalar_partials)
                            .arg(apsi)
                            .arg(b)
                            .arg(ax_ref)
                            .arg(&nl)
                            .launch(cfg)?;
                    }
                    solver::finish_sum(gpu, sol, scalar_slot, scalar_partials, nparts, 0.0)?;
                }
            }
            None => gpu.fill_zero(&mut self.scalar_slot)?,
        }
        self.allgather_scalar(gpu, comm)?;
        self.finish_gathered(gpu, sol, eps)
    }
}

