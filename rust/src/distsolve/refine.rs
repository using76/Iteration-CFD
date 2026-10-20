// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! The distributed Krylov under `single`, refined: SPEC-LIT §118.4's
//! mixed-precision iterative refinement around §73's `dist_pcg` /
//! `dist_pbicgstab` and §138's `spmd_pcg` / `spmd_pbicgstab` (SPEC-LIT
//! §118.12, F32-07), and since AMG-06a (§141.9) around §141.11's stationary
//! `spmd_gamg`. `single` only; the f64 build does not compile this file.
//!
//! The outer loop is (118.4.1)-(118.4.9) on every part at once: the
//! solution `x` in double on each part, the double residual and §8.4 factor
//! row by row, and the unchanged float method solving only the scaled
//! correction. Two things are distributed and nothing else:
//!
//! * **the halo of `x`** - a cut face's row reads its neighbour's `x`. The
//!   double crosses the float exchange as three floats, `hi = fl32(x)`,
//!   `mid = fl32(x - hi)`, `lo = fl32(x - hi - mid)` (118.12.1), joined back
//!   as `(hi + mid) + lo` (118.12.2): three exchanges, bit for bit the
//!   owner's double for `2^-97 <= |x| < 2^128` and for zero;
//! * **the sums** - since PAR-14 run 2 (§139.10) each world sum is §72's
//!   exact accumulator over the part's DOUBLE terms
//!   ([`crate::exactsum::ExactSum64`]): the anchor is the world maximum
//!   (order-free), the limbs are exact integer arithmetic, the gather moves
//!   bytes (one `Comm::allgather` under SPMD), and the ONE rounding is the
//!   host finish [`crate::exactsum::exact_sum64`]. The sums are
//!   partition-invariant to the bit - the same multiset of per-cell terms
//!   lands on the same limbs whatever the cut - which is what lets a
//!   decomposed run answer the whole mesh's solve bit for bit.
//!
//! Provenance: ORIGINAL - the distributed placement of §118.4's refinement
//! (Moler 1967; Carson & Higham 2018) is this project's own; the three-float
//! split is the classical error-free transformation of a double into
//! non-overlapping floats (Dekker 1971; Hida, Li & Bailey 2001), cited in
//! SPEC-LIT §118.12. No GPL-licensed source was consulted.

use crate::comm::Comm;
use crate::device::{DevBuf, Gpu};
use crate::error::{Error, Result};
use crate::exactsum::{exact_sum64, ExactSum64, EXACT_WORDS};
use crate::halo::{HaloExchange, SpmdHalo};
use crate::precision::{ir_inner_tolerance, ir_target, ir_verdict, IrVerdict};
use crate::solver::{self, SolverControls, SolverKernels};
use crate::{Label, Scalar};

use super::spmd::SpmdWorkspace;
use super::{DistPerformance, DistSystem, DistWorkspace};

/// The float method an outer step drives.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Method {
    Pcg,
    Pbicgstab,
    /// The stationary distributed V-cycle, [`super::spmd::spmd_gamg`]
    /// (SPEC-LIT §141.11), refined since AMG-06a (§141.9). SPMD only: the
    /// in-process `dist_solve` refuses GAMG before it gets here.
    Gamg,
}

/// The refinement's state, one entry per part this process holds (every
/// part in process, the rank's one part under SPMD). Every field vector is
/// the part's `n_local` long, so the halo of `x` has somewhere to land.
pub(super) struct DistIr {
    /// `x^k` in double (118.4.1).
    x_hi: Vec<DevBuf<f64>>,
    /// The correction the float method computes (118.4.6).
    d: Vec<DevBuf<Scalar>>,
    /// `btilde`, the unit-norm right-hand side (118.4.5).
    b_in: Vec<DevBuf<Scalar>>,
    /// The three floats of `x` while its halo crosses the float exchange.
    hi: Vec<DevBuf<Scalar>>,
    mid: Vec<DevBuf<Scalar>>,
    lo: Vec<DevBuf<Scalar>>,
    /// Three double slots per part: `[0]` the WORLD `sum x` (written back
    /// for the §8.4 kernel's `x_ref`), `[1]` the part's `N_p`, `[2]` its
    /// `R_p`.
    scal: Vec<DevBuf<f64>>,
    /// The exact world sums (§139.10): the anchor and limb passes over the
    /// part's double terms.
    exact: ExactSum64,
    /// The part's per-cell residual terms `|b - A.x|`, owned cells only.
    r_terms: Vec<DevBuf<f64>>,
    /// The part's per-cell §8.4 terms `|A.x - A.x_ref| + |b - A.x_ref|`.
    n_terms: Vec<DevBuf<f64>>,
    /// Outer steps of the LAST solve; 0 when it was not refined.
    pub(super) outer_steps: usize,
}

impl DistIr {
    /// Allocate for parts of `n` owned and `n_local` owned + halo cells.
    pub(super) fn new(gpu: &Gpu, n: &[usize], n_local: &[usize]) -> Result<Self> {
        let f32s = || -> Result<Vec<DevBuf<Scalar>>> {
            n_local.iter().map(|&k| gpu.zeros(k.max(1))).collect()
        };
        Ok(Self {
            x_hi: n_local
                .iter()
                .map(|&k| gpu.zeros(k.max(1)))
                .collect::<Result<_>>()?,
            d: f32s()?,
            b_in: f32s()?,
            hi: f32s()?,
            mid: f32s()?,
            lo: f32s()?,
            scal: n.iter().map(|_| gpu.zeros(3)).collect::<Result<_>>()?,
            exact: ExactSum64::new(gpu, n)?,
            r_terms: n
                .iter()
                .map(|&k| gpu.zeros(k.max(1)))
                .collect::<Result<_>>()?,
            n_terms: n
                .iter()
                .map(|&k| gpu.zeros(k.max(1)))
                .collect::<Result<_>>()?,
            outer_steps: 0,
        })
    }

    /// `[parts]` the double solution of the last refined solve, `n_local`
    /// long; the owned cells are the answer, the halo the neighbours'.
    pub(super) fn x_hi(&self) -> &[DevBuf<f64>] {
        &self.x_hi
    }
}

/// Whether a solve is refined: every solve but a `fixed_iters` one (the
/// kind a CUDA graph may capture, §113.2) and an empty system (§118.4).
pub(super) fn refines(ctrl: &SolverControls, n_global_cells: usize) -> bool {
    !ctrl.fixed_iters && n_global_cells > 0
}

/// What differs between the in-process and the SPMD forms: how a halo is
/// filled, how this process's part values become the world's, and which
/// float body solves the correction.
trait Parts {
    /// Fill the halo of one float field on every part held here.
    fn exchange(&mut self, gpu: &Gpu, f: &mut [DevBuf<Scalar>]) -> Result<()>;
    /// The world maximum of the parts' magnitudes. Exact whatever the
    /// order - a maximum has none (§72.3).
    fn world_max(&mut self, mine: &[f64]) -> Result<f64>;
    /// The world sum of the parts' exact limb sets (`n_parts`
    /// concatenated `EXACT_WORDS` sets, gathered in part = rank order),
    /// against the anchor `amax` they were formed with. Exact whatever the
    /// order; the ONE rounding is [`exact_sum64`]'s host finish (§139.10).
    fn world_sum(&mut self, mine: &[i64], amax: f64, offset: f64) -> Result<f64>;
    /// Every rank's outer verdict agrees: the SPMD flag vote, taken on
    /// `stop` when the workspace's vote is on; nothing in process.
    fn vote(&mut self, stop: bool) -> Result<()>;
    /// The unchanged float method on `A d = b` (118.4.6).
    fn inner(
        &mut self,
        gpu: &Gpu,
        d: &mut [DevBuf<Scalar>],
        b: &[DevBuf<Scalar>],
        ctrl: &SolverControls,
    ) -> Result<DistPerformance>;
}

/// Every part in one process.
struct InProcess<'a> {
    sol: &'a SolverKernels,
    sys: &'a DistSystem,
    ex: &'a mut HaloExchange,
    w: &'a mut DistWorkspace,
    method: Method,
}

impl Parts for InProcess<'_> {
    fn exchange(&mut self, gpu: &Gpu, f: &mut [DevBuf<Scalar>]) -> Result<()> {
        self.ex.scalar(gpu, f)
    }

    fn world_max(&mut self, mine: &[f64]) -> Result<f64> {
        Ok(mine.iter().copied().fold(0.0f64, f64::max))
    }

    fn world_sum(&mut self, mine: &[i64], amax: f64, offset: f64) -> Result<f64> {
        // The parts are all here; the exact i64 combine is local.
        Ok(exact_sum64(mine, amax, offset))
    }

    fn vote(&mut self, _stop: bool) -> Result<()> {
        Ok(())
    }

    fn inner(
        &mut self,
        gpu: &Gpu,
        d: &mut [DevBuf<Scalar>],
        b: &[DevBuf<Scalar>],
        ctrl: &SolverControls,
    ) -> Result<DistPerformance> {
        match self.method {
            Method::Pcg => {
                super::dist_pcg_float(gpu, self.sol, d, Some(b), self.sys, self.ex, self.w, ctrl)
            }
            Method::Pbicgstab => super::dist_pbicgstab_float(
                gpu,
                self.sol,
                d,
                Some(b),
                self.sys,
                self.ex,
                self.w,
                ctrl,
            ),
            Method::Gamg => Err(Error::Config(
                "dist_solve: the stationary GAMG has no in-process form; it runs \
                 on the SPMD path, spmd_solve with SpmdWorkspace::set_gamg \
                 (SPEC-LIT 141.11)"
                    .to_string(),
            )),
        }
    }
}

/// One part per rank.
struct Spmd<'a> {
    sol: &'a SolverKernels,
    comm: &'a dyn Comm,
    sys: &'a DistSystem,
    halo: &'a mut SpmdHalo,
    w: &'a mut SpmdWorkspace,
    method: Method,
}

impl Parts for Spmd<'_> {
    fn exchange(&mut self, gpu: &Gpu, f: &mut [DevBuf<Scalar>]) -> Result<()> {
        self.halo.scalar(gpu, self.comm, &mut f[0])
    }

    fn world_max(&mut self, mine: &[f64]) -> Result<f64> {
        // One double per rank: allgather moves bytes, the host maximum is
        // exact whatever the order it arrives in (§72.3).
        let bytes: Vec<u8> = mine
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let size = self.comm.size();
        let mut all = vec![0u8; 8 * size];
        self.comm.allgather(&bytes, &mut all)?;
        Ok(all
            .chunks_exact(8)
            .map(|b| f64::from_le_bytes(b.try_into().expect("8 bytes")))
            .fold(0.0f64, f64::max))
    }

    fn world_sum(&mut self, mine: &[i64], amax: f64, offset: f64) -> Result<f64> {
        // One limb set per rank: the allgather moves the bytes, and
        // exact_sum64's i64 combine over them is exact whatever the order.
        let bytes: Vec<u8> = mine
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let size = self.comm.size();
        let mut all = vec![0u8; 8 * EXACT_WORDS * size];
        self.comm.allgather(&bytes, &mut all)?;
        let sets: Vec<i64> = all
            .chunks_exact(8)
            .map(|b| i64::from_le_bytes(b.try_into().expect("8 bytes")))
            .collect();
        Ok(exact_sum64(&sets, amax, offset))
    }

    fn vote(&mut self, stop: bool) -> Result<()> {
        let who = match self.method {
            Method::Pcg => "spmd_pcg",
            Method::Pbicgstab => "spmd_pbicgstab",
            Method::Gamg => "spmd_gamg",
        };
        super::spmd::vote_verdict(self.comm, self.w, stop, who)
    }

    fn inner(
        &mut self,
        gpu: &Gpu,
        d: &mut [DevBuf<Scalar>],
        b: &[DevBuf<Scalar>],
        ctrl: &SolverControls,
    ) -> Result<DistPerformance> {
        let (sol, comm, sys) = (self.sol, self.comm, self.sys);
        match self.method {
            Method::Pcg => super::spmd::spmd_pcg_float(
                gpu,
                sol,
                comm,
                &mut d[0],
                Some(&b[0]),
                sys,
                self.halo,
                self.w,
                ctrl,
            ),
            Method::Pbicgstab => super::spmd::spmd_pbicgstab_float(
                gpu,
                sol,
                comm,
                &mut d[0],
                Some(&b[0]),
                sys,
                self.halo,
                self.w,
                ctrl,
            ),
            Method::Gamg => super::spmd::spmd_gamg_float(
                gpu,
                sol,
                comm,
                &mut d[0],
                Some(&b[0]),
                sys,
                self.halo,
                self.w,
                ctrl,
            ),
        }
    }
}

/// The in-process refined solve: [`super::dist_pcg`] / [`super::dist_pbicgstab`]
/// under `single`. The workspace's [`DistIr`] is taken out for the duration
/// and put back on EVERY return path.
#[allow(clippy::too_many_arguments)]
pub(super) fn dist_refined(
    gpu: &Gpu,
    sol: &SolverKernels,
    psi: &mut [DevBuf<Scalar>],
    sys: &DistSystem,
    ex: &mut HaloExchange,
    w: &mut DistWorkspace,
    ctrl: &SolverControls,
    method: Method,
) -> Result<DistPerformance> {
    let mut ir = w.ir.take().ok_or_else(no_state)?;
    let mut perf = DistPerformance::default();
    let out = {
        let mut parts = InProcess {
            sol,
            sys,
            ex,
            w: &mut *w,
            method,
        };
        refine(gpu, sol, &mut parts, sys, psi, &mut ir, ctrl, &mut perf)
    };
    w.ir = Some(ir);
    out.map(|_| perf)
}

/// The SPMD refined solve: [`super::spmd_pcg`] / [`super::spmd_pbicgstab`]
/// / [`super::spmd_gamg`] under `single`, on the rank's one part. The caller has run the SPMD
/// preflight, so every refusal came before the first collective.
#[allow(clippy::too_many_arguments)]
pub(super) fn spmd_refined(
    gpu: &Gpu,
    sol: &SolverKernels,
    comm: &dyn Comm,
    psi: &mut DevBuf<Scalar>,
    sys: &DistSystem,
    halo: &mut SpmdHalo,
    w: &mut SpmdWorkspace,
    ctrl: &SolverControls,
    method: Method,
) -> Result<DistPerformance> {
    let mut ir = w.ir.take().ok_or_else(no_state)?;
    let mut perf = DistPerformance::default();
    let out = {
        let mut parts = Spmd {
            sol,
            comm,
            sys,
            halo,
            w: &mut *w,
            method,
        };
        refine(
            gpu,
            sol,
            &mut parts,
            sys,
            std::slice::from_mut(psi),
            &mut ir,
            ctrl,
            &mut perf,
        )
    };
    w.ir = Some(ir);
    out.map(|_| perf)
}

fn no_state() -> Error {
    Error::Config("distsolve: no iterative-refinement state on this workspace".to_string())
}

/// (118.12.1)-(118.12.2) the halo of every part's `x` from its owners:
/// split, three float exchanges, join on the halo cells.
fn fill_halo<P: Parts>(
    gpu: &Gpu,
    sol: &SolverKernels,
    parts: &mut P,
    sys: &DistSystem,
    ir: &mut DistIr,
    perf: &mut DistPerformance,
) -> Result<()> {
    let (own, loc) = (sys.owned(), sys.local());
    for p in 0..own.len() {
        solver::split3(
            gpu,
            sol,
            &mut ir.hi[p],
            &mut ir.mid[p],
            &mut ir.lo[p],
            &ir.x_hi[p],
            own[p],
        )?;
    }
    parts.exchange(gpu, &mut ir.hi)?;
    parts.exchange(gpu, &mut ir.mid)?;
    parts.exchange(gpu, &mut ir.lo)?;
    perf.n_exchanges += 3;
    for p in 0..own.len() {
        solver::join3(
            gpu,
            sol,
            &mut ir.x_hi[p],
            &ir.hi[p],
            &ir.mid[p],
            &ir.lo[p],
            own[p],
            loc[p],
        )?;
    }
    Ok(())
}

/// (118.4.3) every part's per-cell residual terms `|b - A.x|` into
/// `r_terms[p]`, OWNED cells only (§139.10): a halo cell's row is its
/// owner's, and summing it twice would make the world sum depend on the
/// cut. The world sum itself is [`world_exact`]'s.
fn residual_terms(
    gpu: &Gpu,
    sol: &SolverKernels,
    sys: &DistSystem,
    ir: &mut DistIr,
) -> Result<()> {
    for p in 0..sys.n_parts() {
        let own = sys.owned()[p];
        if own == 0 {
            continue;
        }
        solver::residual64_terms(
            gpu,
            sol,
            &mut ir.r_terms[p],
            &ir.x_hi[p],
            &sys.a[p],
            &sys.mesh[p],
            own,
        )?;
    }
    Ok(())
}

/// (118.4.2) the same for the §8.4 factor's per-cell terms, into
/// `n_terms[p]`, with `x_ref = sum x / n_global` from `scal[p][0]`.
#[allow(clippy::too_many_arguments)]
fn norm_factor_terms(
    gpu: &Gpu,
    sol: &SolverKernels,
    sys: &DistSystem,
    ir: &mut DistIr,
    inv_n: f64,
) -> Result<()> {
    for p in 0..sys.n_parts() {
        let own = sys.owned()[p];
        if own == 0 {
            continue;
        }
        solver::norm_factor64_terms(
            gpu,
            sol,
            &mut ir.n_terms[p],
            &ir.x_hi[p],
            &ir.scal[p],
            inv_n,
            &sys.a[p],
            &sys.mesh[p],
            own,
        )?;
    }
    Ok(())
}

/// Which buffer one exact world sum runs over.
#[derive(Clone, Copy)]
enum Terms {
    /// The solution itself, `x_hi` (the `sum x` reduction).
    X,
    /// The materialised §8.4 terms, `n_terms`.
    N,
    /// The materialised residual terms, `r_terms`.
    R,
}

/// One exact world sum (§139.10): the anchor pass over the `Terms` buffer,
/// then the limb pass, then the Parts seam's gather and the host finish. A
/// part without cells contributes zeros to both gathers, so the
/// collectives keep their shape.
fn world_exact<P: Parts>(
    gpu: &Gpu,
    parts: &mut P,
    sys: &DistSystem,
    ir: &mut DistIr,
    which: Terms,
    offset: f64,
) -> Result<f64> {
    let own = sys.owned();
    let mut maxima = Vec::with_capacity(own.len());
    for p in 0..own.len() {
        maxima.push(if own[p] == 0 {
            0.0
        } else {
            let t = match which {
                Terms::X => &ir.x_hi[p],
                Terms::N => &ir.n_terms[p],
                Terms::R => &ir.r_terms[p],
            };
            ir.exact.max_part(gpu, p, t, own[p])?
        });
    }
    let amax = parts.world_max(&maxima)?;
    ir.exact.set_anchor(gpu, amax)?;
    let mut mine: Vec<i64> = Vec::with_capacity(own.len() * EXACT_WORDS);
    for p in 0..own.len() {
        if own[p] == 0 {
            mine.extend(std::iter::repeat(0).take(EXACT_WORDS));
            continue;
        }
        let t = match which {
            Terms::X => &ir.x_hi[p],
            Terms::N => &ir.n_terms[p],
            Terms::R => &ir.r_terms[p],
        };
        mine.extend(ir.exact.part_limbs(gpu, p, t, own[p])?);
    }
    Ok(parts.world_sum(&mine, amax, offset)?)
}

/// SPEC-LIT (118.4.1)-(118.4.9) over every part, the halo and the sums as
/// the module header says.
#[allow(clippy::too_many_arguments)]
fn refine<P: Parts>(
    gpu: &Gpu,
    sol: &SolverKernels,
    parts: &mut P,
    sys: &DistSystem,
    psi: &mut [DevBuf<Scalar>],
    ir: &mut DistIr,
    ctrl: &SolverControls,
    perf: &mut DistPerformance,
) -> Result<()> {
    let np = sys.n_parts();
    let own = sys.owned().to_vec();
    let loc = sys.local().to_vec();

    // (118.4.1) x^0 = fl64(psi) on the owned cells, then the halo from the
    // owners - so a stale halo in `psi` never enters a row.
    for p in 0..np {
        if own[p] > 0 {
            solver::widen(gpu, sol, &mut ir.x_hi[p], &psi[p], own[p])?;
        }
    }
    fill_halo(gpu, sol, parts, sys, ir, perf)?;

    // (118.4.2) the world `sum x`, exact (§139.10), written back to every
    // part's slot 0 for the §8.4 kernel's `x_ref = sum x / n_global`.
    let sum_x = world_exact(gpu, parts, sys, ir, Terms::X, 0.0)?;
    perf.n_reductions += 1;
    let inv_n = 1.0 / (sys.n_global_cells() as f64);
    for p in 0..np {
        gpu.write(&mut ir.scal[p], &[sum_x, 0.0, 0.0])?;
    }
    norm_factor_terms(gpu, sol, sys, ir, inv_n)?;
    // (118.4.2) N, and (118.4.3) R_0, each an exact world sum.
    let n_sum = world_exact(gpu, parts, sys, ir, Terms::N, 0.0)?;
    residual_terms(gpu, sol, sys, ir)?;
    let r_0 = world_exact(gpu, parts, sys, ir, Terms::R, 0.0)?;
    perf.n_reductions += 3;
    let norm = n_sum + f64::MIN_POSITIVE;
    let mut r_k = r_0;

    // (118.4.4) T = max(tol·N, relTol·R_0).
    let target = ir_target(ctrl.tolerance as f64, ctrl.rel_tol as f64, norm, r_0);
    let max_iter = ctrl.max_iter.max(0) as usize;
    let min_iter = ctrl.min_iter.max(0) as usize;
    let mut r_prev: Option<f64> = None;
    let mut sweeps: usize = 0;
    let mut outer: usize = 0;

    // (118.4.8) the stop rules on the freshly computed world R_k - the same
    // bits on every rank, so every rank stops on the same step, and the SPMD
    // vote checks that instead of trusting it (ranks with different controls
    // would part here).
    let verdict = loop {
        let v = ir_verdict(r_k, r_prev, target, sweeps, min_iter, max_iter, outer);
        parts.vote(v != IrVerdict::Continue)?;
        if v != IrVerdict::Continue {
            break v;
        }

        // (118.4.5) btilde = fl32(r^k / R_k); d = 0 to start, halo included.
        let inv_sigma = 1.0 / r_k;
        for p in 0..np {
            solver::scaled_residual64(
                gpu,
                sol,
                &mut ir.b_in[p],
                &ir.x_hi[p],
                &sys.a[p],
                &sys.mesh[p],
                inv_sigma,
            )?;
            gpu.fill_zero(&mut ir.d[p])?;
        }

        // (118.4.6) the unchanged float method on the scaled system.
        let inner_ctrl = SolverControls {
            tolerance: ir_inner_tolerance(target, r_k) as Scalar,
            rel_tol: 0.0,
            max_iter: (max_iter - sweeps) as Label,
            min_iter: 0,
            fixed_iters: false,
            report_residuals: false,
            ..*ctrl
        };
        let done = parts.inner(gpu, &mut ir.d, &ir.b_in, &inner_ctrl)?;
        sweeps += done.n_iterations;
        perf.n_exchanges += done.n_exchanges;
        perf.n_reductions += done.n_reductions;

        // (118.4.7) x^{k+1} = x^k + R_k·fl64(d) on the owned cells; the
        // halo follows from the owners.
        for p in 0..np {
            solver::axpy_widen64(gpu, sol, &mut ir.x_hi[p], &ir.d[p], r_k, own[p])?;
        }
        outer += 1;
        fill_halo(gpu, sol, parts, sys, ir, perf)?;

        // (118.4.3) again, and its exact world sum.
        residual_terms(gpu, sol, sys, ir)?;
        r_prev = Some(r_k);
        r_k = world_exact(gpu, parts, sys, ir, Terms::R, 0.0)?;
        perf.n_reductions += 1;
    };

    // (118.4.9) psi = fl32(x^final) on owned AND halo cells: the halo holds
    // the owners' doubles, so the stored field agrees across every cut.
    for p in 0..np {
        if loc[p] > 0 {
            solver::narrow(gpu, sol, &mut psi[p], &ir.x_hi[p], loc[p])?;
        }
    }

    perf.n_iterations = sweeps;
    perf.converged = verdict == IrVerdict::Converged;
    if ctrl.report_residuals {
        perf.initial_residual = (r_0 / norm) as Scalar;
        perf.final_residual = (r_k / norm) as Scalar;
    }
    ir.outer_steps = outer;
    Ok(())
}
