// meteor-cfd - Copyright (c) 2026 주식회사 메테오시뮬레이션 (Meteo Simulation Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

//! Cross-part reductions that do not move when the mesh is cut.
//!
//! SPEC-LIT §72. `cuda/solver.cu`'s reductions are bitwise reproducible for a
//! **fixed** length and a **fixed** grid; §71 cut the mesh, and the moment it
//! did, every dot product, norm, volume mean and patch total in the crate
//! became a sum over a set that a partition splits differently every time.
//! Floating-point addition is not associative, so those are different numbers.
//!
//! This module supplies two constructions, and the difference between them is
//! the whole point of the section:
//!
//! * [`ExactReduction::gathered_sum`] and its dot twin run each part's
//!   *existing* two-stage reduction and then **gather** the `P` partials —
//!   moved, never all-reduced — into `solSumStage2`. That is four lines of
//!   glue and it buys the same answer run to run for one fixed partition.
//!   It does **not** buy the same answer for another partition, and it does
//!   not reproduce the whole mesh's answer.
//! * [`ExactReduction::sum`], [`ExactReduction::sum_mag`],
//!   [`ExactReduction::dot`] and [`ExactReduction::norm_factor`] split every
//!   term exactly into four 30-bit integer limbs against one **global**
//!   anchor, sum the limbs as `i64`, and convert back once. Integer addition
//!   is associative, so the answer is a function of the multiset of terms:
//!   identical for every part count, every partition map and every
//!   relabelling of the parts, and equal to the one-part answer.
//!
//! Both are here because the claim that the cheap one is not enough is worth
//! more with the cheap one sitting next to it being measured —
//! `the_gathered_partial_construction_moves_and_the_exact_one_does_not` runs
//! the two side by side on the same data and reports which moved.
//!
//! # The collective is a gather, never a reduce
//!
//! Every cross-part step here is one of exactly two things: a `memcpy_dtod` —
//! the stand-in for `ncclAllGather`, which moves bytes, performs no
//! arithmetic, and is therefore exact — or a **one-block kernel of fixed
//! shape** over the gathered values. There is no `ncclAllReduce` and there
//! must never be one: NVIDIA documents `NCCL_ALGO`, `NCCL_PROTO` and
//! `NCCL_MAX_CTAS` as performance knobs and promises nothing about reduction
//! order, so an all-reduce would make the answer a property of the fabric.
//!
//! # What is reused rather than rewritten
//!
//! `max|x|` is `solMaxMagStage1` and every stage-two maximum is
//! `solMaxStage2`, both from `cuda/solver.cu` and both unchanged. A maximum
//! over non-NaN operands is *exactly* order-independent, so maxima needed
//! nothing from this section but the gather. `cuda/exactsum.cu` adds a
//! stage-one maximum only for the two term expressions that did not already
//! have one.
//!
//! Provenance: the limb decomposition is DERIVED from the binned / indexed
//! floating-point literature (Demmel & Nguyen 2013, 2015; Ahrens, Demmel &
//! Nguyen 2020; Collange et al. 2015; Kulisch 2013) — the header of
//! `cuda/exactsum.cu` carries the full citations and `PROVENANCE.md` records
//! them. The limb width, the limb count, the single global anchor and the
//! gather-not-reduce collective are ORIGINAL.
//! No GPL-licensed source was consulted.

use cudarc::driver::{CudaFunction, LaunchConfig, PushKernelArg};

use crate::device::{DevBuf, Gpu, KernelSet};
use crate::error::{Error, Result};
use crate::solver::{self, one_block, reduce_geometry, to_label, SolverKernels};
use crate::Scalar;

// ==========================================================================
//  The limb layout
// ==========================================================================

/// Limb width in bits. Mirrors `OFGPU_EX_W` in `cuda/exactsum.cu`;
/// `the_limb_layout_matches_the_device` pins the two together by measurement
/// rather than by comment.
pub const EXACT_W: u32 = 30;

/// Limbs per accumulator. Mirrors `OFGPU_EX_K`.
///
/// `EXACT_K * EXACT_W` is 120 binades of coverage below the largest term, so a
/// term is dropped only when it is more than 120 binades below the biggest
/// one — 67 binades below the last bit a double can hold.
pub const EXACT_K: usize = 4;

/// Words per accumulator: the [`EXACT_K`] value limbs plus **one more holding
/// the count of non-finite terms**. Mirrors `OFGPU_EX_WORDS`.
///
/// The extra word exists because the anchor comes from a maximum, and
/// `max(a, NaN)` in the device's `a > b ? a : b` form *discards* the NaN — so
/// a single NaN term would leave the anchor finite, the limbs formed from
/// garbage, and the reduction answering a plausible number. `device_sum` does
/// not have that problem (`acc += NaN` is NaN and propagates), and replacing
/// it with something that hides a poisoned field would be a regression. So the
/// accumulator counts what it cannot represent, and the conversion answers
/// NaN. `a_non_finite_term_poisons_the_answer_visibly` is the test that found
/// this, after the first version of the guard checked only the anchor and
/// caught infinities but not NaNs.
pub const EXACT_WORDS: usize = EXACT_K + 1;

/// Most terms one accumulator may sum before an `i64` limb could overflow.
///
/// Each term contributes at most `2^EXACT_W - 1` to a limb, so `n` terms need
/// `n (2^30 - 1) < 2^63`, i.e. `n < 2^33`. The bound below keeps a whole
/// binade in hand. It is stated and enforced rather than assumed, because the
/// one way this accumulator can be wrong is silently, by wrapping.
pub const EXACT_MAX_TERMS: usize = 1usize << 32;

// ==========================================================================
//  Kernels
// ==========================================================================

/// Every entry point in `cuda/exactsum.cu`, resolved once.
struct ExactKernels {
    dot_max1: CudaFunction,
    norm_factor_max1: CudaFunction,
    sum1: CudaFunction,
    sum_mag1: CudaFunction,
    dot1: CudaFunction,
    norm_factor1: CudaFunction,
    combine: CudaFunction,
    to_scalar: CudaFunction,
}

impl ExactKernels {
    fn new(gpu: &Gpu) -> Result<Self> {
        let k = KernelSet::new(gpu, crate::kernels::EXACTSUM)?;
        Ok(Self {
            dot_max1: k.func("exDotMaxStage1")?,
            norm_factor_max1: k.func("exNormFactorMaxStage1")?,
            sum1: k.func("exSumStage1")?,
            sum_mag1: k.func("exSumMagStage1")?,
            dot1: k.func("exDotStage1")?,
            norm_factor1: k.func("exNormFactorStage1")?,
            combine: k.func("exCombine")?,
            to_scalar: k.func("exToScalar")?,
        })
    }
}

/// One thread: the limb-to-float conversion, which is not a reduction.
fn one_thread() -> LaunchConfig {
    LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    }
}

// ==========================================================================
//  The reduction
// ==========================================================================

/// A cross-part reduction over `P` parts with fixed term counts.
///
/// The counts are fixed at construction because in a distributed run they are:
/// `n_r` is how many cells part `r` owns and it does not change. Passing them
/// once means every call can check the buffers it is handed against the
/// lengths it was promised, which is the difference between a refusal and a
/// read past the end of a field.
///
/// A part's buffer may be **longer** than its term count — every field in a
/// decomposed run is `n_cells + n_halo` long (§71.1) and only the owned half
/// is summed. It may never be shorter.
pub struct ExactReduction {
    k: ExactKernels,
    /// `[n_parts]` terms each part contributes.
    n: Vec<usize>,
    /// `[n_parts]` per-part stage-one scalar partials — the maxima, and the
    /// plain sums of the gathered-partial construction.
    scalar_partials: Vec<DevBuf<Scalar>>,
    /// `[n_parts]` per-part stage-one limb partials, `EXACT_K` per block.
    limb_partials: Vec<DevBuf<i64>>,

    /// `[1]` the slot every per-part stage two writes into, before it is
    /// *moved* into the gathered buffer. This is the `ncclAllGather` source
    /// buffer and the copy out of it is the collective.
    scalar_slot: DevBuf<Scalar>,
    /// `[EXACT_WORDS]` the same, for a part's limb total.
    limb_slot: DevBuf<i64>,

    /// `[n_parts]` the gathered per-part scalars.
    part_scalar: DevBuf<Scalar>,
    /// `[n_parts * EXACT_WORDS]` the gathered per-part limb totals.
    part_limbs: DevBuf<i64>,

    /// `[1]` the global anchor, `max|t|`.
    amax: DevBuf<Scalar>,
    /// `[EXACT_WORDS]` the global limb total.
    total: DevBuf<i64>,
    /// `[1]` the answer.
    out: DevBuf<Scalar>,
}

impl ExactReduction {
    /// Allocate for parts owning `n[r]` terms each.
    ///
    /// `n` may have one entry, and that is not a special case: a one-part
    /// reduction runs the identical code with `P = 1`, so the serial path is
    /// the parallel path and cannot drift away from it untested.
    pub fn new(gpu: &Gpu, n: &[usize]) -> Result<Self> {
        if n.is_empty() {
            return Err(Error::Config(
                "exactsum: a reduction over zero parts has no answer; pass at \
                 least one part, owning zero terms if that is what it owns"
                    .to_string(),
            ));
        }
        let total_terms: usize = n.iter().sum();
        if total_terms > EXACT_MAX_TERMS {
            return Err(Error::Config(format!(
                "exactsum: {total_terms} terms over {} part(s) exceeds the \
                 {EXACT_MAX_TERMS}-term limit of a {EXACT_W}-bit limb held in \
                 an i64 (SPEC-LIT §72.2). Narrow EXACT_W and widen EXACT_K, or \
                 reduce in groups",
                n.len()
            )));
        }
        for &ni in n {
            to_label(ni)?;
        }

        let np = n.len();
        let mut scalar_partials = Vec::with_capacity(np);
        let mut limb_partials = Vec::with_capacity(np);
        for &ni in n {
            let parts = reduce_geometry(ni.max(1)).1;
            scalar_partials.push(gpu.zeros(parts)?);
            limb_partials.push(gpu.zeros(parts * EXACT_WORDS)?);
        }

        Ok(Self {
            k: ExactKernels::new(gpu)?,
            n: n.to_vec(),
            scalar_partials,
            limb_partials,
            scalar_slot: gpu.zeros(1)?,
            limb_slot: gpu.zeros(EXACT_WORDS)?,
            part_scalar: gpu.zeros(np)?,
            part_limbs: gpu.zeros(np * EXACT_WORDS)?,
            amax: gpu.zeros(1)?,
            total: gpu.zeros(EXACT_WORDS)?,
            out: gpu.zeros(1)?,
        })
    }

    /// How many parts this reduction was built for.
    pub fn n_parts(&self) -> usize {
        self.n.len()
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

    /// The global limb total the last exact call produced — `EXACT_WORDS`
    /// values, the last of which is the non-finite term count.
    ///
    /// Exposed because it, and not the converted float, is the object that is
    /// order-free: a test that compares limbs compares the accumulator itself
    /// rather than a rounded view of it.
    pub fn limbs(&self, gpu: &Gpu) -> Result<Vec<i64>> {
        gpu.download(&self.total)
    }

    /// How many terms of the last exact call were an infinity or a NaN.
    ///
    /// Non-zero means the answer is a NaN by construction, and that the field
    /// was already poisoned before this module saw it.
    pub fn non_finite_terms(&self, gpu: &Gpu) -> Result<i64> {
        Ok(self.limbs(gpu)?[EXACT_K])
    }

    // ----------------------------------------------------------------------
    //  Partition-invariant: the exact accumulator
    // ----------------------------------------------------------------------

    /// `out = sum_r sum_{i < n_r} x_r[i]`.
    pub fn sum(&mut self, gpu: &Gpu, sol: &SolverKernels, xs: &[DevBuf<Scalar>]) -> Result<()> {
        self.check(xs, "sum", "x")?;
        self.anchor_of_magnitude(gpu, sol, xs)?;
        for p in 0..self.n.len() {
            let Some((cfg, nparts, nl)) = self.geometry(p) else {
                self.zero_part_limbs(gpu, p)?;
                continue;
            };
            {
                let Self { k, limb_partials, amax, .. } = self;
                unsafe {
                    gpu.stream()
                        .launch_builder(&k.sum1)
                        .arg(&mut limb_partials[p])
                        .arg(&xs[p])
                        .arg(&nl)
                        .arg(&*amax)
                        .launch(cfg)?;
                }
            }
            self.gather_part_limbs(gpu, p, nparts)?;
        }
        self.finish(gpu, 0.0)
    }

    /// `out = sum_r sum_{i < n_r} |x_r[i]|` — the residual measure of
    /// `SPEC-LIT` §8.4.
    pub fn sum_mag(&mut self, gpu: &Gpu, sol: &SolverKernels, xs: &[DevBuf<Scalar>]) -> Result<()> {
        self.check(xs, "sum_mag", "x")?;
        self.anchor_of_magnitude(gpu, sol, xs)?;
        for p in 0..self.n.len() {
            let Some((cfg, nparts, nl)) = self.geometry(p) else {
                self.zero_part_limbs(gpu, p)?;
                continue;
            };
            {
                let Self { k, limb_partials, amax, .. } = self;
                unsafe {
                    gpu.stream()
                        .launch_builder(&k.sum_mag1)
                        .arg(&mut limb_partials[p])
                        .arg(&xs[p])
                        .arg(&nl)
                        .arg(&*amax)
                        .launch(cfg)?;
                }
            }
            self.gather_part_limbs(gpu, p, nparts)?;
        }
        self.finish(gpu, 0.0)
    }

    /// `out = sum_r (a_r, b_r)`.
    pub fn dot(
        &mut self,
        gpu: &Gpu,
        sol: &SolverKernels,
        a: &[DevBuf<Scalar>],
        b: &[DevBuf<Scalar>],
    ) -> Result<()> {
        self.check(a, "dot", "a")?;
        self.check(b, "dot", "b")?;

        // ---- the anchor: max|a b| over every part, then the gather --------
        for p in 0..self.n.len() {
            let Some((cfg, nparts, nl)) = self.geometry(p) else {
                self.zero_part_scalar(gpu, p)?;
                continue;
            };
            {
                let Self { k, scalar_partials, .. } = self;
                unsafe {
                    gpu.stream()
                        .launch_builder(&k.dot_max1)
                        .arg(&mut scalar_partials[p])
                        .arg(&a[p])
                        .arg(&b[p])
                        .arg(&nl)
                        .launch(cfg)?;
                }
            }
            self.gather_part_max(gpu, sol, p, nparts)?;
        }
        self.finish_anchor(gpu, sol)?;

        // ---- the limbs ----------------------------------------------------
        for p in 0..self.n.len() {
            let Some((cfg, nparts, nl)) = self.geometry(p) else {
                self.zero_part_limbs(gpu, p)?;
                continue;
            };
            {
                let Self { k, limb_partials, amax, .. } = self;
                unsafe {
                    gpu.stream()
                        .launch_builder(&k.dot1)
                        .arg(&mut limb_partials[p])
                        .arg(&a[p])
                        .arg(&b[p])
                        .arg(&nl)
                        .arg(&*amax)
                        .launch(cfg)?;
                }
            }
            self.gather_part_limbs(gpu, p, nparts)?;
        }
        self.finish(gpu, 0.0)
    }

    /// `SPEC-LIT` §8.4's normalisation factor:
    /// `sum ( |A.psi - A.xRef| + |b - A.xRef| ) + eps`.
    ///
    /// `eps` is added once, after the accumulator has been converted, exactly
    /// as `solSumStage2`'s `offset` is — so the exact twin takes the same
    /// argument in the same place and a reader can line the two up.
    pub fn norm_factor(
        &mut self,
        gpu: &Gpu,
        sol: &SolverKernels,
        apsi: &[DevBuf<Scalar>],
        b: &[DevBuf<Scalar>],
        ax_ref: &[DevBuf<Scalar>],
        eps: Scalar,
    ) -> Result<()> {
        self.check(apsi, "norm_factor", "Apsi")?;
        self.check(b, "norm_factor", "b")?;
        self.check(ax_ref, "norm_factor", "AxRef")?;

        for p in 0..self.n.len() {
            let Some((cfg, nparts, nl)) = self.geometry(p) else {
                self.zero_part_scalar(gpu, p)?;
                continue;
            };
            {
                let Self { k, scalar_partials, .. } = self;
                unsafe {
                    gpu.stream()
                        .launch_builder(&k.norm_factor_max1)
                        .arg(&mut scalar_partials[p])
                        .arg(&apsi[p])
                        .arg(&b[p])
                        .arg(&ax_ref[p])
                        .arg(&nl)
                        .launch(cfg)?;
                }
            }
            self.gather_part_max(gpu, sol, p, nparts)?;
        }
        self.finish_anchor(gpu, sol)?;

        for p in 0..self.n.len() {
            let Some((cfg, nparts, nl)) = self.geometry(p) else {
                self.zero_part_limbs(gpu, p)?;
                continue;
            };
            {
                let Self { k, limb_partials, amax, .. } = self;
                unsafe {
                    gpu.stream()
                        .launch_builder(&k.norm_factor1)
                        .arg(&mut limb_partials[p])
                        .arg(&apsi[p])
                        .arg(&b[p])
                        .arg(&ax_ref[p])
                        .arg(&nl)
                        .arg(&*amax)
                        .launch(cfg)?;
                }
            }
            self.gather_part_limbs(gpu, p, nparts)?;
        }
        self.finish(gpu, eps)
    }

    // ----------------------------------------------------------------------
    //  Order-free without any of the above
    // ----------------------------------------------------------------------

    /// `out = max_r max_i |x_r[i]|`.
    ///
    /// Here for completeness and for the section's honesty. A maximum over
    /// non-NaN operands is **exactly** order-independent, so `contErr`, the
    /// Courant number and every adaptive-`dt` measure in the crate are already
    /// partition-invariant and need nothing from the accumulator. What they do
    /// need is the gather, which is this method.
    pub fn max_mag(&mut self, gpu: &Gpu, sol: &SolverKernels, xs: &[DevBuf<Scalar>]) -> Result<()> {
        self.check(xs, "max_mag", "x")?;
        self.anchor_of_magnitude(gpu, sol, xs)?;
        let Self { out, amax, .. } = self;
        gpu.stream().memcpy_dtod(&amax.slice(0..1), &mut out.slice_mut(0..1))?;
        Ok(())
    }

    // ----------------------------------------------------------------------
    //  Run-invariant only: the gathered partial
    // ----------------------------------------------------------------------

    /// `out = fl(sigma_0 + ... + sigma_{P-1})` where `sigma_r` is part `r`'s
    /// own [`crate::solver::device_sum`].
    ///
    /// **This is not partition-invariant, and it is here to show that it is
    /// not.** `sigma_r` is a rounded number that depends on which cells part
    /// `r` owns, so no care taken over the order of the combine can recover
    /// the whole mesh's answer. It is the cheapest construction that is
    /// reproducible *run to run for a fixed partition*, which is a real
    /// guarantee and a strictly weaker one.
    pub fn gathered_sum(
        &mut self,
        gpu: &Gpu,
        sol: &SolverKernels,
        xs: &[DevBuf<Scalar>],
    ) -> Result<()> {
        self.check(xs, "gathered_sum", "x")?;
        for p in 0..self.n.len() {
            {
                let Self { scalar_slot, scalar_partials, n, .. } = self;
                solver::device_sum(gpu, sol, scalar_slot, &xs[p], &mut scalar_partials[p], n[p])?;
            }
            self.gather_scalar_slot(gpu, p)?;
        }
        self.finish_gathered(gpu, sol, 0.0)
    }

    /// The same construction for a dot product.
    pub fn gathered_dot(
        &mut self,
        gpu: &Gpu,
        sol: &SolverKernels,
        a: &[DevBuf<Scalar>],
        b: &[DevBuf<Scalar>],
    ) -> Result<()> {
        self.check(a, "gathered_dot", "a")?;
        self.check(b, "gathered_dot", "b")?;
        for p in 0..self.n.len() {
            {
                let Self { scalar_slot, scalar_partials, n, .. } = self;
                solver::device_dot(
                    gpu,
                    sol,
                    scalar_slot,
                    &a[p],
                    &b[p],
                    &mut scalar_partials[p],
                    n[p],
                )?;
            }
            self.gather_scalar_slot(gpu, p)?;
        }
        self.finish_gathered(gpu, sol, 0.0)
    }

    /// The same construction for a magnitude sum — the residual measure a
    /// Krylov solve tests convergence on.
    pub fn gathered_sum_mag(
        &mut self,
        gpu: &Gpu,
        sol: &SolverKernels,
        xs: &[DevBuf<Scalar>],
    ) -> Result<()> {
        self.check(xs, "gathered_sum_mag", "x")?;
        for p in 0..self.n.len() {
            {
                let Self { scalar_slot, scalar_partials, n, .. } = self;
                solver::device_sum_mag(
                    gpu,
                    sol,
                    scalar_slot,
                    &xs[p],
                    &mut scalar_partials[p],
                    n[p],
                )?;
            }
            self.gather_scalar_slot(gpu, p)?;
        }
        self.finish_gathered(gpu, sol, 0.0)
    }

    /// The same construction for `SPEC-LIT` §8.4's normalisation factor.
    ///
    /// `eps` is added **once**, by the cross-part combine, and not by each
    /// part's own stage two — adding it `P` times would make the factor a
    /// function of the part count even before the summation order did, which
    /// is the sort of arithmetic that hides inside a "cheap" construction and
    /// is worth being explicit about.
    pub fn gathered_norm_factor(
        &mut self,
        gpu: &Gpu,
        sol: &SolverKernels,
        apsi: &[DevBuf<Scalar>],
        b: &[DevBuf<Scalar>],
        ax_ref: &[DevBuf<Scalar>],
        eps: Scalar,
    ) -> Result<()> {
        self.check(apsi, "gathered_norm_factor", "Apsi")?;
        self.check(b, "gathered_norm_factor", "b")?;
        self.check(ax_ref, "gathered_norm_factor", "AxRef")?;
        for p in 0..self.n.len() {
            let Some((cfg, nparts, nl)) = self.geometry(p) else {
                self.zero_part_scalar(gpu, p)?;
                continue;
            };
            {
                let Self { scalar_slot, scalar_partials, .. } = self;
                unsafe {
                    gpu.stream()
                        .launch_builder(&sol.norm_factor1)
                        .arg(&mut scalar_partials[p])
                        .arg(&apsi[p])
                        .arg(&b[p])
                        .arg(&ax_ref[p])
                        .arg(&nl)
                        .launch(cfg)?;
                }
                solver::finish_sum(gpu, sol, scalar_slot, &scalar_partials[p], nparts, 0.0)?;
            }
            self.gather_scalar_slot(gpu, p)?;
        }
        self.finish_gathered(gpu, sol, eps)
    }

    // ----------------------------------------------------------------------
    //  Internals
    // ----------------------------------------------------------------------

    /// Launch geometry for part `p`, or `None` when it owns nothing. A
    /// zero-block grid is an illegal launch configuration, so a part with no
    /// cells is answered with a memset rather than a kernel — the same rule
    /// `device_sum` follows for an empty reduction.
    fn geometry(&self, p: usize) -> Option<(LaunchConfig, usize, crate::Label)> {
        let np = self.n[p];
        if np == 0 {
            return None;
        }
        let (cfg, nparts) = reduce_geometry(np);
        // `new` proved every count fits a label, so this cannot fail here.
        Some((cfg, nparts, np as crate::Label))
    }

    /// Every buffer must be at least as long as its part's term count.
    fn check(&self, xs: &[DevBuf<Scalar>], what: &str, name: &str) -> Result<()> {
        if xs.len() != self.n.len() {
            return Err(Error::Config(format!(
                "exactsum: {what} was given {} `{name}` buffer(s) for {} part(s)",
                xs.len(),
                self.n.len()
            )));
        }
        for (p, (x, &ni)) in xs.iter().zip(&self.n).enumerate() {
            if x.len() < ni {
                return Err(Error::Config(format!(
                    "exactsum: {what}: part {p}'s `{name}` buffer holds {} values \
                     and the part owns {ni} cells. A decomposed field must be \
                     n_cells + n_halo long (SPEC-LIT §71.1)",
                    x.len()
                )));
            }
        }
        Ok(())
    }

    /// The anchor from `max|x|`, reusing `solMaxMagStage1` unchanged.
    fn anchor_of_magnitude(
        &mut self,
        gpu: &Gpu,
        sol: &SolverKernels,
        xs: &[DevBuf<Scalar>],
    ) -> Result<()> {
        for p in 0..self.n.len() {
            {
                let Self { scalar_slot, scalar_partials, n, .. } = self;
                solver::device_max_mag(
                    gpu,
                    sol,
                    scalar_slot,
                    &xs[p],
                    &mut scalar_partials[p],
                    n[p],
                )?;
            }
            self.gather_scalar_slot(gpu, p)?;
        }
        self.finish_anchor(gpu, sol)
    }

    /// Stage two of a part's own maximum — `solMaxStage2`, unchanged — and the
    /// move into the gathered buffer.
    fn gather_part_max(
        &mut self,
        gpu: &Gpu,
        sol: &SolverKernels,
        p: usize,
        nparts: usize,
    ) -> Result<()> {
        let npl = to_label(nparts)?;
        {
            let Self { scalar_slot, scalar_partials, .. } = self;
            unsafe {
                gpu.stream()
                    .launch_builder(&sol.max2)
                    .arg(&mut *scalar_slot)
                    .arg(&scalar_partials[p])
                    .arg(&npl)
                    .launch(one_block())?;
            }
        }
        self.gather_scalar_slot(gpu, p)
    }

    /// The collective, for one scalar: a copy, and therefore exact.
    fn gather_scalar_slot(&mut self, gpu: &Gpu, p: usize) -> Result<()> {
        let Self { scalar_slot, part_scalar, .. } = self;
        gpu.stream()
            .memcpy_dtod(&scalar_slot.slice(0..1), &mut part_scalar.slice_mut(p..p + 1))?;
        Ok(())
    }

    /// A part that owns nothing contributes a zero to the gathered maxima.
    fn zero_part_scalar(&mut self, gpu: &Gpu, p: usize) -> Result<()> {
        let Self { scalar_slot, .. } = self;
        gpu.fill_zero(scalar_slot)?;
        self.gather_scalar_slot(gpu, p)
    }

    /// Stage two of a part's own limb sum, and the move into the gathered
    /// buffer. `exCombine` here is the same kernel the cross-part combine
    /// runs; only the length differs.
    fn gather_part_limbs(&mut self, gpu: &Gpu, p: usize, nparts: usize) -> Result<()> {
        let npl = to_label(nparts)?;
        {
            let Self { k, limb_slot, limb_partials, .. } = self;
            unsafe {
                gpu.stream()
                    .launch_builder(&k.combine)
                    .arg(&mut *limb_slot)
                    .arg(&limb_partials[p])
                    .arg(&npl)
                    .launch(one_block())?;
            }
        }
        self.move_limb_slot(gpu, p)
    }

    /// A part that owns nothing contributes a zero limb set.
    fn zero_part_limbs(&mut self, gpu: &Gpu, p: usize) -> Result<()> {
        let Self { limb_slot, .. } = self;
        gpu.fill_zero(limb_slot)?;
        self.move_limb_slot(gpu, p)
    }

    /// The collective, for one part's `EXACT_K` limbs.
    fn move_limb_slot(&mut self, gpu: &Gpu, p: usize) -> Result<()> {
        let Self { limb_slot, part_limbs, .. } = self;
        let (a, b) = (p * EXACT_WORDS, (p + 1) * EXACT_WORDS);
        gpu.stream()
            .memcpy_dtod(&limb_slot.slice(0..EXACT_WORDS), &mut part_limbs.slice_mut(a..b))?;
        Ok(())
    }

    /// The gathered per-part maxima -> the one global anchor. `solMaxStage2`,
    /// one block, fixed shape — and exact whatever the order, because a
    /// maximum is.
    fn finish_anchor(&mut self, gpu: &Gpu, sol: &SolverKernels) -> Result<()> {
        let np = to_label(self.n.len())?;
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

    /// The gathered per-part limb totals -> one limb total -> one float. The
    /// same `exCombine` that finished each part, over `P` values instead of
    /// over the blocks: the cross-part combine is not a new kernel.
    fn finish(&mut self, gpu: &Gpu, offset: Scalar) -> Result<()> {
        let np = to_label(self.n.len())?;
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

    /// The gathered-partial combine: `solSumStage2` over `P` doubles.
    fn finish_gathered(&mut self, gpu: &Gpu, sol: &SolverKernels, offset: Scalar) -> Result<()> {
        let np = to_label(self.n.len())?;
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
}

// ==========================================================================
//  A host twin, so the device is checked against something and not itself
// ==========================================================================

/// `x * 2^e`, exactly, without an intermediate that could overflow.
///
/// Rust's `std` has no `ldexp`. Multiplying by `2^e` in chunks of at most 500
/// binades is exact — scaling by a power of two only moves the exponent — and
/// cannot overflow on the way whenever the *result* is in range, because every
/// chunk moves the running value monotonically toward it.
pub fn ldexp(x: Scalar, e: i32) -> Scalar {
    let mut r = x;
    let mut e = e;
    while e > 500 {
        r *= pow2(500);
        e -= 500;
    }
    while e < -500 {
        r *= pow2(-500);
        e += 500;
    }
    r * pow2(e)
}

/// `2^e` for `|e| <= 1000`, built by repeated exact multiplication.
fn pow2(e: i32) -> Scalar {
    let (n, step) = if e >= 0 { (e, 2.0 as Scalar) } else { (-e, 0.5 as Scalar) };
    let mut r = 1.0 as Scalar;
    for _ in 0..n {
        r *= step;
    }
    r
}

/// The anchor exponent `M`: an integer with `|t| < 2^M` for every term.
///
/// The host half of `exAnchor_`, which is `ilogb(amax) + 1`. `ilogb` is not in
/// Rust's `std`, and going through the bit pattern would need a subnormal
/// case; scaling into `[0.5, 1)` has none.
//
// `!(amax > 0.0)` rather than `amax <= 0.0`: the negation is the point. A NaN
// compares false against everything, so this form catches it and the tidier
// one does not, and the device kernel is written the same way for the same
// reason.
#[allow(clippy::neg_cmp_op_on_partial_ord)]
pub fn anchor_exponent(amax: Scalar) -> i32 {
    if !(amax > 0.0) {
        return 0;
    }
    let mut e = 0i32;
    let mut f = amax;
    while f >= 1.0 {
        f *= 0.5;
        e += 1;
    }
    while f < 0.5 {
        f *= 2.0;
        e -= 1;
    }
    e
}

/// The same accumulator, on the host, in `i128`.
///
/// Written out independently and used only by the tests: two implementations
/// that agree bit for bit are evidence, one implementation agreeing with
/// itself is not. `i128` rather than `i64` so the host twin does not lean on
/// the same overflow argument the device one does — if `EXACT_MAX_TERMS` were
/// wrong the two would disagree instead of wrapping together.
// See `anchor_exponent` for why the comparison is negated.
#[allow(clippy::neg_cmp_op_on_partial_ord)]
pub fn host_exact_sum(terms: &[Scalar]) -> Scalar {
    // The device counts non-finite terms in the extra word and answers NaN;
    // the twin has to answer the same thing or the comparison is meaningless.
    if terms.iter().any(|t| !t.is_finite()) {
        return Scalar::NAN;
    }
    let amax = terms.iter().fold(0.0 as Scalar, |m, t| m.max(t.abs()));
    if !(amax > 0.0) {
        return 0.0;
    }
    let mexp = anchor_exponent(amax);
    let radix = pow2(EXACT_W as i32);
    let mut m = [0i128; EXACT_K];
    for &t in terms {
        let mut s = ldexp(t, EXACT_W as i32 - mexp);
        for mk in m.iter_mut() {
            let q = s.trunc();
            *mk += q as i128;
            s = (s - q) * radix;
        }
    }
    for k in (1..EXACT_K).rev() {
        let carry = m[k] >> EXACT_W;
        m[k] -= carry << EXACT_W;
        m[k - 1] += carry;
    }
    let inv = pow2(-(EXACT_W as i32));
    let mut v = 0.0 as Scalar;
    for k in (1..EXACT_K).rev() {
        v = (m[k] as Scalar + v) * inv;
    }
    ldexp(m[0] as Scalar + v, mexp - EXACT_W as i32)
}

#[cfg(test)]
mod tests;
