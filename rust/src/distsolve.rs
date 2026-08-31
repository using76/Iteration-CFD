// meteor-cfd - Copyright (c) 2026 주식회사 메테오시뮬레이션 (Meteo Simulation Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

//! PCG and PBiCGStab over a decomposed mesh.
//!
//! SPEC-LIT §73. §71 cut the mesh and moved data across the cuts; §72 made a
//! reduction a function of the multiset of terms rather than of the partition.
//! Neither of them ran a solver, because a Krylov method is exactly the thing
//! that needs both at once: a matrix product that reaches across a cut and a
//! dot product that sums over every part. This module puts the two together
//! and runs the two Krylov methods the crate already has.
//!
//! # The two guarantees, kept apart on purpose
//!
//! [`DistReduce`] selects which cross-part reduction a solve uses, and the
//! choice is the difference between two claims that are often conflated:
//!
//! * [`DistReduce::Gathered`] - each part's own `device_dot`, gathered and
//!   combined by the fixed one-block kernel. **Run-invariant**: the same
//!   binary, case, part count and partition give the same bits. It is not
//!   partition-invariant, and SPEC-LIT §72.5 measures it moving.
//! * [`DistReduce::Exact`] - SPEC-LIT §72's limb accumulator.
//!   **Partition-invariant**: the same bits at every part count, under every
//!   partition map and every relabelling, and equal to the one-part answer.
//!
//! The gathered mode is not a straw man. It is what a distributed run would
//! ship by default if reproducibility across decompositions were not the
//! claim, and it is here because at `P = 1` it makes this module's solvers
//! **bitwise the existing `solver::solve_pcg` and `solver::solve_pbicgstab`** -
//! which is the test that pins every kernel argument in the recurrences below
//! against solvers that are already gated. See
//! `a_one_part_gathered_solve_is_the_serial_solver`.
//!
//! # What the halo costs a Krylov method, exactly
//!
//! One exchange per matrix product, and nothing else. PCG has one product per
//! iteration, PBiCGStab two. Every other step of both recurrences - `axpy`,
//! `axmy`, the BiCGStab `p`/`s`/`x`/`r` updates, the Jacobi preconditioner -
//! is elementwise on a cell's own values and reads nothing across a cut, so no
//! exchange is owed and none is made. The scalar recurrences (`alpha`,
//! `beta`, `omega`) are one-thread kernels on values that are already global.
//!
//! # The preconditioner is where the honesty is owed
//!
//! `Diagonal` is `z_i = r_i / diag_i`. It is elementwise, and a part's `diag`
//! is bitwise the whole mesh's `diag` (§71.6), so it is partition-invariant
//! **for free** - no exchange, no colouring, nothing.
//!
//! `Dic` and `Dilu` are not, and cannot be made so cheaply. The factorisation
//! `Dt_v = A_vv - SUM_{u < v} A_vu A_uv / Dt_u` is sequential in the colour
//! order and the sequence crosses cuts. What this module runs is the
//! factorisation of each part's **own** submatrix, with the couplings to other
//! parts dropped. That is a different preconditioner for every partition, so
//! the iterate sequence is a function of the cut and the answer is only
//! run-invariant. §73.5 measures what it costs in iterations, and this module
//! refuses to pretend otherwise: [`DistWorkspace::partition_invariant`]
//! answers `false` for it, and the gate asserts the answer really does move.
//!
//! **It is block Jacobi, not restricted additive Schwarz.** RAS (Cai &
//! Sarkis 1999) is additive Schwarz on *overlapping* subdomains with the
//! overlap discarded on the update; with zero overlap - which is what a
//! one-cell halo used only for the matrix product gives - it degenerates to
//! block Jacobi (Saad 2003, ch. 14). The halo is read by `lduAmul` and by
//! nothing in `precon.rs`, so there is no overlap to restrict. Calling it RAS
//! would predict the wrong iteration count: RAS improves with the overlap
//! width and this has none.
//!
//! # Provenance
//!
//! ORIGINAL. The recurrences are Hestenes & Stiefel (1952) and van der Vorst
//! (1992) exactly as `solver.rs` already implements them - every kernel this
//! module launches is one `cuda/solver.cu` or `cuda/precon.cu` already owns,
//! unchanged, so the arithmetic is literally the serial solver's. What is new
//! is the placement of the exchange and the substitution of the cross-part
//! reduction. Block Jacobi and additive Schwarz are Saad (2003) ch. 14; the
//! distinction from Cai & Sarkis (1999) is drawn above.
//! No GPL-licensed source was consulted.

use cudarc::driver::PushKernelArg;

use crate::decompose::Decomposition;
use crate::device::{cfg_for, DevBuf, Gpu};
use crate::error::{Error, Result};
use crate::exactsum::ExactReduction;
use crate::halo::HaloExchange;
use crate::ldu::{GpuLduMatrix, HostLduMatrix};
use crate::mesh::{GpuMesh, HostMesh};
use crate::precon::MultiColour;
use crate::solver::{self, LinearSolverKind, Preconditioner, SolverControls, SolverKernels};
use crate::{Label, Scalar};

// ==========================================================================
//  Which cross-part reduction
// ==========================================================================

/// Which construction a distributed solve reduces with.
///
/// The names are the guarantees, not the implementations: see the module
/// header and SPEC-LIT §72.5.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DistReduce {
    /// SPEC-LIT §72's limb accumulator. Partition-invariant, and four to five
    /// times the cost of the plain reduction (§72.6).
    Exact,
    /// Each part's own `device_dot`, gathered and combined. Run-invariant
    /// only.
    Gathered,
}

impl DistReduce {
    pub fn as_str(&self) -> &'static str {
        match self {
            DistReduce::Exact => "exact",
            DistReduce::Gathered => "gathered",
        }
    }
}

// ==========================================================================
//  The distributed system
// ==========================================================================

/// Every part's mesh and matrix, on the device.
///
/// Built by [`DistSystem::split`], which distributes a matrix **assembled on
/// the whole mesh**. Nothing here re-assembles: SPEC-LIT §70.5 names the
/// sixteen gather kernels that are not yet partition-invariant, and a
/// decomposed run that called them would be testing a claim this section does
/// not make. §71.6 is the argument for why every coefficient of a split matrix
/// is a copy or an exact negation of a whole-mesh coefficient.
pub struct DistSystem {
    /// `[P]` the parts' meshes.
    pub mesh: Vec<GpuMesh>,
    /// `[P]` the parts' matrices.
    pub a: Vec<GpuLduMatrix>,
    /// `[P]` owned cells.
    n: Vec<usize>,
    /// `[P]` owned + halo - the length every field buffer must have.
    n_local: Vec<usize>,
    /// Cells in the whole mesh. `normFactor`'s mean divides by this and not by
    /// a part's own count, which is the `sum(b)/n` hazard SPEC-LIT §72.7 named
    /// and left open.
    n_global_cells: usize,
}

impl DistSystem {
    /// Split `a` over `dec` and upload every part.
    ///
    /// `is_fixed` and `fixed_value` reach the halo through the **exchange**,
    /// not from the host's whole-mesh copy: a genuinely distributed build has
    /// no such copy, and `lduSetValues` reads both at a coupled face's
    /// neighbour, which on a part is a ghost cell.
    pub fn split(
        gpu: &Gpu,
        ex: &mut HaloExchange,
        m: &HostMesh,
        dec: &Decomposition,
        a: &HostLduMatrix,
    ) -> Result<Self> {
        if ex.n_parts() != dec.n_parts {
            return Err(Error::Config(format!(
                "distsolve: the exchange was built for {} part(s), the \
                 decomposition has {}",
                ex.n_parts(),
                dec.n_parts
            )));
        }
        let np = dec.n_parts;
        let split: Vec<HostLduMatrix> = (0..np)
            .map(|p| dec.split_matrix(m, p, a))
            .collect::<Result<_>>()?;

        let mut mesh = Vec::with_capacity(np);
        let mut mats = Vec::with_capacity(np);
        let mut n = Vec::with_capacity(np);
        let mut n_local = Vec::with_capacity(np);
        for p in 0..np {
            mesh.push(GpuMesh::upload(gpu, &dec.parts[p].mesh)?);
            mats.push(split[p].upload(gpu)?);
            n.push(dec.parts[p].mesh.n_cells);
            n_local.push(dec.parts[p].n_local());
        }

        // The two constraint arrays are the only matrix state a kernel reads
        // at a halo index, so they are the only ones the exchange has to fill.
        let mut isf: Vec<DevBuf<Label>> = (0..np)
            .map(|p| gpu.upload(&split[p].is_fixed))
            .collect::<Result<_>>()?;
        let mut fv: Vec<DevBuf<Scalar>> = (0..np)
            .map(|p| gpu.upload(&split[p].fixed_value))
            .collect::<Result<_>>()?;
        ex.label(gpu, &mut isf)?;
        ex.scalar(gpu, &mut fv)?;
        for (mat, (i, v)) in mats.iter_mut().zip(isf.into_iter().zip(fv)) {
            mat.is_fixed = i;
            mat.fixed_value = v;
        }

        Ok(Self {
            mesh,
            a: mats,
            n,
            n_local,
            n_global_cells: dec.n_global_cells,
        })
    }

    pub fn n_parts(&self) -> usize {
        self.n.len()
    }

    /// `[P]` owned cells per part - what a reduction sums over and what every
    /// kernel launches threads for.
    pub fn owned(&self) -> &[usize] {
        &self.n
    }

    /// `[P]` owned + halo - what every field buffer must be long enough for.
    pub fn local(&self) -> &[usize] {
        &self.n_local
    }

    pub fn n_global_cells(&self) -> usize {
        self.n_global_cells
    }

    /// Allocate one halo-length scalar field per part.
    pub fn zeros(&self, gpu: &Gpu) -> Result<Vec<DevBuf<Scalar>>> {
        self.n_local.iter().map(|&k| gpu.zeros(k)).collect()
    }
}

// ==========================================================================
//  The workspace
// ==========================================================================

/// What a distributed solve needs that is not the system or the solution.
///
/// Struct of arrays, indexed by part - because that is the shape both
/// [`HaloExchange::scalar`] (`&mut [DevBuf]`) and [`ExactReduction::dot`]
/// (`&[DevBuf]`) take. An array of per-part structs would have to be
/// re-gathered into a slice at every call site, and `DevBuf: Clone` is a
/// device-to-device **copy** in `cudarc`, so that gather would allocate and
/// copy inside the iteration loop.
pub struct DistWorkspace {
    n: Vec<usize>,
    n_local: Vec<usize>,
    n_global_cells: usize,
    reduce: DistReduce,

    // ---- Krylov vectors, one per part, each `n_local` long ---------------
    r: Vec<DevBuf<Scalar>>,
    r0: Vec<DevBuf<Scalar>>,
    p: Vec<DevBuf<Scalar>>,
    v: Vec<DevBuf<Scalar>>,
    s: Vec<DevBuf<Scalar>>,
    t: Vec<DevBuf<Scalar>>,
    p_hat: Vec<DevBuf<Scalar>>,
    s_hat: Vec<DevBuf<Scalar>>,
    apsi: Vec<DevBuf<Scalar>>,
    tmp: Vec<DevBuf<Scalar>>,
    y: Vec<DevBuf<Scalar>>,
    /// The right-hand side, copied out of `a[p].source` at the top of a solve.
    /// A copy and not a borrow because the reduction API takes a slice of
    /// buffers and the sources live one inside each matrix.
    b: Vec<DevBuf<Scalar>>,
    /// `1/Dt` (DIC/DILU) or `1/diag` (Jacobi), per part.
    r_diag: Vec<DevBuf<Scalar>>,
    /// `[P]` the part-local colouring, when one was built.
    multicolour: Vec<Option<MultiColour>>,

    // ---- the cross-part reduction ----------------------------------------
    red: ExactReduction,

    // ---- scalars, one value each, shared by every part -------------------
    rho: DevBuf<Scalar>,
    rho_old: DevBuf<Scalar>,
    alpha: DevBuf<Scalar>,
    omega: DevBuf<Scalar>,
    beta: DevBuf<Scalar>,
    num: DevBuf<Scalar>,
    den: DevBuf<Scalar>,
    x_ref: DevBuf<Scalar>,
    norm_factor: DevBuf<Scalar>,
    initial_res: DevBuf<Scalar>,
    final_res: DevBuf<Scalar>,
    flag: DevBuf<Label>,
}

/// What a distributed solve did.
#[derive(Debug, Clone, Default)]
pub struct DistPerformance {
    pub initial_residual: Scalar,
    pub final_residual: Scalar,
    pub n_iterations: usize,
    pub converged: bool,
    /// Halo exchanges performed - one per matrix product, and no others.
    pub n_exchanges: usize,
    /// Cross-part reductions performed. Each is one collective in a
    /// distributed build, and the count is what a strong-scaling model needs.
    pub n_reductions: usize,
}

impl DistWorkspace {
    /// Allocate for `sys`, reducing the way `reduce` says.
    pub fn new(gpu: &Gpu, sys: &DistSystem, reduce: DistReduce) -> Result<Self> {
        let np = sys.n_parts();
        if np == 0 {
            return Err(Error::Config(
                "distsolve: a system with no parts has nothing to solve".to_string(),
            ));
        }
        Ok(Self {
            n: sys.n.clone(),
            n_local: sys.n_local.clone(),
            n_global_cells: sys.n_global_cells,
            reduce,
            r: sys.zeros(gpu)?,
            r0: sys.zeros(gpu)?,
            p: sys.zeros(gpu)?,
            v: sys.zeros(gpu)?,
            s: sys.zeros(gpu)?,
            t: sys.zeros(gpu)?,
            p_hat: sys.zeros(gpu)?,
            s_hat: sys.zeros(gpu)?,
            apsi: sys.zeros(gpu)?,
            tmp: sys.zeros(gpu)?,
            y: sys.zeros(gpu)?,
            b: sys.zeros(gpu)?,
            r_diag: sys.zeros(gpu)?,
            multicolour: (0..np).map(|_| None).collect(),
            red: ExactReduction::new(gpu, &sys.n)?,
            rho: gpu.zeros(1)?,
            rho_old: gpu.zeros(1)?,
            alpha: gpu.zeros(1)?,
            omega: gpu.zeros(1)?,
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

    pub fn n_parts(&self) -> usize {
        self.n.len()
    }

    pub fn reduce_mode(&self) -> DistReduce {
        self.reduce
    }

    /// Colour every part's own graph, so `Dic` and `Dilu` become available.
    ///
    /// The colouring is greedy in each part's **local** cell order, which a
    /// different partition changes. That is one of the two reasons the
    /// incomplete factorisations are not partition-invariant here; the other,
    /// and the larger one, is that the couplings across a cut are dropped
    /// altogether. Neither is hidden: [`Self::partition_invariant`] answers
    /// `false`.
    pub fn colour(&mut self, gpu: &Gpu, sys: &DistSystem) -> Result<()> {
        for p in 0..self.n_parts() {
            self.multicolour[p] = Some(MultiColour::new(gpu, &sys.mesh[p])?);
        }
        Ok(())
    }

    /// `[P]` colours per part, or zeros if nothing has been coloured. The
    /// measurement SPEC-LIT §73.5 reports alongside the iteration counts.
    pub fn colours(&self) -> Vec<usize> {
        self.multicolour
            .iter()
            .map(|c| c.as_ref().map_or(0, |m| m.n_colours()))
            .collect()
    }

    /// Whether a solve with this workspace and `precon` produces bits that do
    /// not depend on how the mesh was cut.
    ///
    /// Both halves must hold: the reduction must be the exact accumulator, and
    /// the preconditioner must be one that a cut does not change. This is the
    /// function the gate consults rather than a comment, so that adding a
    /// preconditioner without thinking about it fails loudly.
    pub fn partition_invariant(&self, precon: Preconditioner) -> bool {
        let reduction_ok = self.reduce == DistReduce::Exact;
        let precon_ok = match precon {
            // Elementwise on a cell's own values, and a part's `diag` is
            // bitwise the whole mesh's.
            Preconditioner::None | Preconditioner::Diagonal => true,
            // Block-local: the factorisation drops the couplings across every
            // cut, and it is coloured in local order.
            Preconditioner::Dic | Preconditioner::Dilu => false,
        };
        reduction_ok && precon_ok
    }

    /// Which preconditioner a solve will really run, under the SPEC-LIT §13.4
    /// rule: supported, or refused by name with the alternative.
    pub fn effective_preconditioner(&self, requested: Preconditioner) -> Result<Preconditioner> {
        match requested {
            Preconditioner::None | Preconditioner::Diagonal => Ok(requested),
            Preconditioner::Dic | Preconditioner::Dilu => {
                if self.multicolour.iter().all(|c| c.is_some()) {
                    Ok(requested)
                } else {
                    crate::io::contract::unsupported(
                        "solvers/<var>/preconditioner",
                        requested.name(),
                        &["none", "diagonal"],
                        "diagonal (Jacobi): this distributed workspace has not \
                         been coloured, so there is no part-local LDU graph to \
                         factorise. Call DistWorkspace::colour first",
                        Preconditioner::Diagonal,
                    )
                }
            }
        }
    }

    fn check(&self, sys: &DistSystem, psi: &[DevBuf<Scalar>], who: &str) -> Result<()> {
        if sys.n_parts() != self.n_parts() {
            return Err(Error::Config(format!(
                "{who}: the workspace holds {} part(s), the system {}",
                self.n_parts(),
                sys.n_parts()
            )));
        }
        if psi.len() != self.n_parts() {
            return Err(Error::Config(format!(
                "{who}: {} solution buffer(s) for {} part(s)",
                psi.len(),
                self.n_parts()
            )));
        }
        for p in 0..self.n_parts() {
            if sys.n[p] != self.n[p] || sys.n_local[p] != self.n_local[p] {
                return Err(Error::Config(format!(
                    "{who}: part {p} is {}+{} cells in the system and {}+{} in \
                     the workspace",
                    sys.n[p],
                    sys.n_local[p] - sys.n[p],
                    self.n[p],
                    self.n_local[p] - self.n[p]
                )));
            }
            // The halo is not optional. A buffer only `n_cells` long is read
            // past its end by `lduAmul` at every cut face, which SPEC-LIT
            // §71.7 named as the highest-volume risk in the whole distributed
            // effort - so it is refused here by name, with both numbers,
            // rather than left to produce a plausible wrong answer.
            if psi[p].len() < self.n_local[p] {
                return Err(Error::Config(format!(
                    "{who}: part {p}'s solution buffer holds {} values; it must \
                     hold {} - {} owned cells and {} halo cells, because \
                     lduAmul reads psi at a cut face's neighbour",
                    psi[p].len(),
                    self.n_local[p],
                    self.n[p],
                    self.n_local[p] - self.n[p]
                )));
            }
        }
        Ok(())
    }
}

// ==========================================================================
//  The three distributed primitives
// ==========================================================================

/// `y = A x`, on every part, with `x`'s halo filled first.
///
/// **This is the only place an exchange happens.** One per matrix product,
/// because the product is the only operator in either recurrence whose stencil
/// leaves a cell.
fn dist_amul(
    gpu: &Gpu,
    sol: &SolverKernels,
    ex: &mut HaloExchange,
    sys: &DistSystem,
    y: &mut [DevBuf<Scalar>],
    x: &mut [DevBuf<Scalar>],
    perf: &mut DistPerformance,
) -> Result<()> {
    ex.scalar(gpu, x)?;
    perf.n_exchanges += 1;
    for p in 0..sys.n_parts() {
        solver::amul(gpu, sol, &mut y[p], &x[p], &sys.a[p], &sys.mesh[p])?;
    }
    Ok(())
}

/// The four reduction shapes a Krylov solve needs.
enum Reduction<'a> {
    Sum(&'a [DevBuf<Scalar>]),
    SumMag(&'a [DevBuf<Scalar>]),
    Dot(&'a [DevBuf<Scalar>], &'a [DevBuf<Scalar>]),
    /// `A psi`, `b`, `A x_ref`, and the epsilon added once at the end.
    NormFactor(
        &'a [DevBuf<Scalar>],
        &'a [DevBuf<Scalar>],
        &'a [DevBuf<Scalar>],
        Scalar,
    ),
}

/// The cross-part reduction, whichever construction was chosen, into `out`.
///
/// One function for all four shapes so that the choice between the two
/// guarantees is made in exactly one place and cannot be made differently for
/// `sum` than for `dot`.
fn reduce_into(
    gpu: &Gpu,
    sol: &SolverKernels,
    red: &mut ExactReduction,
    mode: DistReduce,
    what: Reduction<'_>,
    out: &mut DevBuf<Scalar>,
    perf: &mut DistPerformance,
) -> Result<()> {
    match (mode, what) {
        (DistReduce::Exact, Reduction::Sum(x)) => red.sum(gpu, sol, x)?,
        (DistReduce::Exact, Reduction::SumMag(x)) => red.sum_mag(gpu, sol, x)?,
        (DistReduce::Exact, Reduction::Dot(a, b)) => red.dot(gpu, sol, a, b)?,
        (DistReduce::Exact, Reduction::NormFactor(ap, b, ax, e)) => {
            red.norm_factor(gpu, sol, ap, b, ax, e)?
        }
        (DistReduce::Gathered, Reduction::Sum(x)) => red.gathered_sum(gpu, sol, x)?,
        (DistReduce::Gathered, Reduction::SumMag(x)) => red.gathered_sum_mag(gpu, sol, x)?,
        (DistReduce::Gathered, Reduction::Dot(a, b)) => red.gathered_dot(gpu, sol, a, b)?,
        (DistReduce::Gathered, Reduction::NormFactor(ap, b, ax, e)) => {
            red.gathered_norm_factor(gpu, sol, ap, b, ax, e)?
        }
    }
    perf.n_reductions += 1;
    solver::copy_scalar(gpu, sol, out, red.out())
}

/// `y = M^-1 x` on every part.
///
/// With `Dic`/`Dilu` this is the part's own factorisation and nothing crosses
/// a cut - the sweeps walk `cf_offset`, which on a part holds that part's
/// interior faces only, so a cut face is simply absent from the recurrence.
/// That absence *is* the block-local approximation; it is not a bug, it is not
/// free, and §73.5 measures what it costs.
#[allow(clippy::too_many_arguments)]
fn dist_precondition(
    gpu: &Gpu,
    sol: &SolverKernels,
    sys: &DistSystem,
    dst: &mut [DevBuf<Scalar>],
    src: &[DevBuf<Scalar>],
    r_diag: &[DevBuf<Scalar>],
    multicolour: &[Option<MultiColour>],
    precon: Preconditioner,
    n: &[usize],
) -> Result<()> {
    for p in 0..sys.n_parts() {
        solver::precondition_parts(
            gpu,
            sol,
            &mut dst[p],
            &src[p],
            &r_diag[p],
            multicolour[p].as_ref(),
            &sys.a[p],
            &sys.mesh[p],
            precon,
            n[p],
        )?;
    }
    Ok(())
}

// ==========================================================================
//  Setup shared by both solvers
// ==========================================================================

/// Build the preconditioner on every part.
fn dist_build_preconditioner(
    gpu: &Gpu,
    sol: &SolverKernels,
    sys: &DistSystem,
    w: &mut DistWorkspace,
    precon: Preconditioner,
) -> Result<()> {
    if precon == Preconditioner::None {
        return Ok(());
    }
    for p in 0..sys.n_parts() {
        let n = w.n[p];
        if n == 0 {
            continue;
        }
        if precon == Preconditioner::Diagonal {
            let nl = solver::to_label(n)?;
            unsafe {
                gpu.stream()
                    .launch_builder(&sol.invert_diag)
                    .arg(&mut w.r_diag[p])
                    .arg(&sys.a[p].diag)
                    .arg(&nl)
                    .launch(cfg_for(n))?;
            }
            continue;
        }
        let DistWorkspace { r_diag, multicolour, .. } = &mut *w;
        let Some(mc) = multicolour[p].as_ref() else {
            return Err(Error::Config(
                "distsolve: DIC/DILU was selected without a colouring; \
                 DistWorkspace::effective_preconditioner should have rejected \
                 this already"
                    .to_string(),
            ));
        };
        mc.factorise(
            gpu,
            &mut r_diag[p],
            &sys.a[p],
            &sys.mesh[p],
            precon == Preconditioner::Dic,
        )?;
    }
    Ok(())
}

/// `w.b[p] = a[p].source`, and the SPEC-LIT §8.4 normalisation factor.
///
/// Leaves `A psi` in `w.apsi`, exactly as `solver::device_norm_factor` does,
/// so the residual is one subtraction and not a second product.
///
/// **The mean divides by the whole mesh's cell count**, not by the part's.
/// `x_ref = mean(psi)` is a property of the field, and a part's own `1/n_r`
/// would make the normalisation - and therefore the convergence test, and
/// therefore the iteration count - a function of the cut. SPEC-LIT §72.7
/// listed this as open; it is closed here.
fn dist_norm_factor(
    gpu: &Gpu,
    sol: &SolverKernels,
    ex: &mut HaloExchange,
    sys: &DistSystem,
    w: &mut DistWorkspace,
    psi: &mut [DevBuf<Scalar>],
    perf: &mut DistPerformance,
) -> Result<()> {
    for p in 0..sys.n_parts() {
        let DistWorkspace { b, n, .. } = &mut *w;
        solver::vec_copy(gpu, sol, &mut b[p], &sys.a[p].source, n[p])?;
    }

    {
        let DistWorkspace { red, x_ref, reduce, .. } = &mut *w;
        reduce_into(gpu, sol, red, *reduce, Reduction::Sum(psi), x_ref, perf)?;
    }

    let inv_n = 1.0 / (w.n_global_cells as Scalar);
    for p in 0..sys.n_parts() {
        let n = w.n[p];
        if n == 0 {
            continue;
        }
        let nl = solver::to_label(n)?;
        unsafe {
            gpu.stream()
                .launch_builder(&sol.broadcast_scaled)
                .arg(&mut w.tmp[p])
                .arg(&w.x_ref)
                .arg(&inv_n)
                .arg(&nl)
                .launch(cfg_for(n))?;
        }
    }

    {
        let DistWorkspace { y, tmp, .. } = &mut *w;
        dist_amul(gpu, sol, ex, sys, y, tmp, perf)?;
    }
    {
        let DistWorkspace { apsi, .. } = &mut *w;
        dist_amul(gpu, sol, ex, sys, apsi, psi, perf)?;
    }

    let DistWorkspace { red, apsi, b, y, norm_factor, reduce, .. } = &mut *w;
    reduce_into(
        gpu,
        sol,
        red,
        *reduce,
        Reduction::NormFactor(&apsi[..], &b[..], &y[..], solver::NORM_EPS),
        norm_factor,
        perf,
    )
}

/// Read the sticky convergence flag back.
///
/// One 4-byte transfer, and the only host round trip in a distributed solve -
/// the same one `solver::read_flag` makes, without the pinned landing pad,
/// because this path is not the one a CUDA graph captures.
fn dist_read_flag(gpu: &Gpu, flag: &DevBuf<Label>) -> Result<bool> {
    Ok(gpu.download(flag)?.first().copied().unwrap_or(0) != 0)
}

/// `r = b - A psi` on every part.
fn dist_residual(gpu: &Gpu, sol: &SolverKernels, w: &mut DistWorkspace) -> Result<()> {
    let DistWorkspace { r, b, apsi, n, .. } = &mut *w;
    for p in 0..n.len() {
        solver::vec_sub(gpu, sol, &mut r[p], &b[p], &apsi[p], n[p])?;
    }
    Ok(())
}

/// The end of both solvers: the true residual, recomputed rather than taken
/// from the recurrence, and the report brought back to the host.
#[allow(clippy::too_many_arguments)]
fn dist_finish(
    gpu: &Gpu,
    sol: &SolverKernels,
    ex: &mut HaloExchange,
    sys: &DistSystem,
    w: &mut DistWorkspace,
    psi: &mut [DevBuf<Scalar>],
    ctrl: &SolverControls,
    perf: &mut DistPerformance,
) -> Result<()> {
    if !ctrl.report_residuals {
        return Ok(());
    }
    {
        let DistWorkspace { apsi, .. } = &mut *w;
        dist_amul(gpu, sol, ex, sys, apsi, psi, perf)?;
    }
    {
        let DistWorkspace { tmp, b, apsi, n, .. } = &mut *w;
        for p in 0..n.len() {
            solver::vec_sub(gpu, sol, &mut tmp[p], &b[p], &apsi[p], n[p])?;
        }
    }
    {
        let DistWorkspace { red, tmp, final_res, reduce, .. } = &mut *w;
        reduce_into(
            gpu,
            sol,
            red,
            *reduce,
            Reduction::SumMag(&tmp[..]),
            final_res,
            perf,
        )?;
    }

    gpu.sync()?;
    perf.initial_residual = gpu.download(&w.initial_res)?[0];
    perf.final_residual = gpu.download(&w.final_res)?[0];
    let nf = gpu.download(&w.norm_factor)?[0];
    if nf > 0.0 {
        perf.initial_residual /= nf;
        perf.final_residual /= nf;
    }
    if ctrl.fixed_iters {
        let abs = perf.final_residual <= ctrl.tolerance;
        let rel = ctrl.rel_tol > 0.0 && perf.final_residual <= ctrl.rel_tol * perf.initial_residual;
        perf.converged = abs || rel;
    }
    Ok(())
}

// ==========================================================================
//  Preconditioned conjugate gradient, distributed
// ==========================================================================

/// PCG over a decomposed mesh.
///
/// The recurrence is Hestenes & Stiefel (1952), Saad §6.7 Algorithm 6.18 -
/// character for character what [`solver::solve_pcg`] runs, on the same
/// kernels:
///
/// ```text
/// r = b - A psi ;  z = M^-1 r ;  p = z ;  rho = (r,z)
/// repeat
///     exchange p ;  q = A p
///     alpha  = rho / (p,q)
///     psi   += alpha p
///     r     -= alpha q
///     z      = M^-1 r
///     rho'   = (r,z) ;  beta = rho'/rho ;  p = z + beta p ;  rho = rho'
/// ```
///
/// **One exchange and two reductions per iteration.** The two reductions are
/// the synchronisation points of the method and no reordering removes them;
/// SPEC-LIT §73.7 refuses the pipelined variants by name and says why.
///
/// **Symmetric positive definite only**, and nothing here checks it - the same
/// rule and the same reason as the serial solver. With `Dic` the part-local
/// blocks must each be SPD too, and they are whenever the whole matrix is:
/// dropping a cut's off-diagonal couplings while keeping the diagonal share
/// they contributed leaves each block strictly more diagonally dominant than
/// the matrix it came from.
pub fn dist_pcg(
    gpu: &Gpu,
    sol: &SolverKernels,
    psi: &mut [DevBuf<Scalar>],
    sys: &DistSystem,
    ex: &mut HaloExchange,
    w: &mut DistWorkspace,
    ctrl: &SolverControls,
) -> Result<DistPerformance> {
    let mut perf = DistPerformance {
        converged: true,
        ..Default::default()
    };
    w.check(sys, psi, "dist_pcg")?;
    if w.n_global_cells == 0 {
        return Ok(perf);
    }
    perf.converged = false;

    let precon = w.effective_preconditioner(ctrl.precon)?;
    dist_build_preconditioner(gpu, sol, sys, w, precon)?;

    dist_norm_factor(gpu, sol, ex, sys, w, psi, &mut perf)?;
    dist_residual(gpu, sol, w)?;
    {
        let DistWorkspace { red, r, initial_res, reduce, .. } = &mut *w;
        reduce_into(
            gpu,
            sol,
            red,
            *reduce,
            Reduction::SumMag(&r[..]),
            initial_res,
            &mut perf,
        )?;
    }
    solver::copy_scalar(gpu, sol, &mut w.final_res, &w.initial_res)?;
    gpu.fill_zero(&mut w.flag)?;

    // z = M^-1 r ; p = z ; rho = (r,z)
    {
        let DistWorkspace { p_hat, r, r_diag, multicolour, n, .. } = &mut *w;
        dist_precondition(
            gpu,
            sol,
            sys,
            &mut p_hat[..],
            &r[..],
            &r_diag[..],
            &multicolour[..],
            precon,
            &n[..],
        )?;
    }
    {
        let DistWorkspace { p, p_hat, n, .. } = &mut *w;
        for i in 0..n.len() {
            solver::vec_copy(gpu, sol, &mut p[i], &p_hat[i], n[i])?;
        }
    }
    {
        let DistWorkspace { red, r, p_hat, rho, reduce, .. } = &mut *w;
        reduce_into(
            gpu,
            sol,
            red,
            *reduce,
            Reduction::Dot(&r[..], &p_hat[..]),
            rho,
            &mut perf,
        )?;
    }

    let max_iter = ctrl.max_iter.max(0) as usize;
    let interval = ctrl.check_interval.max(1) as usize;
    let checking = !ctrl.fixed_iters;

    if checking {
        solver::convergence_test(
            gpu,
            sol,
            &mut w.flag,
            &w.initial_res,
            &w.initial_res,
            &w.norm_factor,
            ctrl,
            0,
        )?;
        perf.converged = dist_read_flag(gpu, &w.flag)?;
    }

    if !perf.converged {
        for it in 0..max_iter {
            let iters = it + 1;

            {
                let DistWorkspace { v, p, .. } = &mut *w;
                dist_amul(gpu, sol, ex, sys, v, p, &mut perf)?;
            }
            {
                let DistWorkspace { red, p, v, den, reduce, .. } = &mut *w;
                reduce_into(
                    gpu,
                    sol,
                    red,
                    *reduce,
                    Reduction::Dot(&p[..], &v[..]),
                    den,
                    &mut perf,
                )?;
            }
            solver::divide_scalar(gpu, sol, &mut w.alpha, &w.rho, &w.den)?;

            for i in 0..sys.n_parts() {
                let n = w.n[i];
                if n == 0 {
                    continue;
                }
                let nl = solver::to_label(n)?;
                unsafe {
                    gpu.stream()
                        .launch_builder(&sol.axpy)
                        .arg(&mut psi[i])
                        .arg(&w.p[i])
                        .arg(&w.alpha)
                        .arg(&nl)
                        .launch(cfg_for(n))?;
                    gpu.stream()
                        .launch_builder(&sol.axmy)
                        .arg(&mut w.r[i])
                        .arg(&w.v[i])
                        .arg(&w.alpha)
                        .arg(&nl)
                        .launch(cfg_for(n))?;
                }
            }

            perf.n_iterations = iters;

            if checking && iters % interval == 0 {
                {
                    let DistWorkspace { red, r, final_res, reduce, .. } = &mut *w;
                    reduce_into(
                        gpu,
                        sol,
                        red,
                        *reduce,
                        Reduction::SumMag(&r[..]),
                        final_res,
                        &mut perf,
                    )?;
                }
                let itl = solver::to_label(iters)?;
                solver::convergence_test(
                    gpu,
                    sol,
                    &mut w.flag,
                    &w.final_res,
                    &w.initial_res,
                    &w.norm_factor,
                    ctrl,
                    itl,
                )?;
                if dist_read_flag(gpu, &w.flag)? {
                    perf.converged = true;
                    break;
                }
            }

            {
                let DistWorkspace { p_hat, r, r_diag, multicolour, n, .. } = &mut *w;
                dist_precondition(
                    gpu,
                    sol,
                    sys,
                    &mut p_hat[..],
                    &r[..],
                    &r_diag[..],
                    &multicolour[..],
                    precon,
                    &n[..],
                )?;
            }
            {
                let DistWorkspace { red, r, p_hat, num, reduce, .. } = &mut *w;
                reduce_into(
                    gpu,
                    sol,
                    red,
                    *reduce,
                    Reduction::Dot(&r[..], &p_hat[..]),
                    num,
                    &mut perf,
                )?;
            }
            solver::divide_scalar(gpu, sol, &mut w.beta, &w.num, &w.rho)?;
            solver::copy_scalar(gpu, sol, &mut w.rho, &w.num)?;

            for i in 0..sys.n_parts() {
                let n = w.n[i];
                if n == 0 {
                    continue;
                }
                let nl = solver::to_label(n)?;
                unsafe {
                    gpu.stream()
                        .launch_builder(&sol.p_update_cg)
                        .arg(&mut w.p[i])
                        .arg(&w.p_hat[i])
                        .arg(&w.beta)
                        .arg(&nl)
                        .launch(cfg_for(n))?;
                }
            }
        }
    }

    dist_finish(gpu, sol, ex, sys, w, psi, ctrl, &mut perf)?;
    Ok(perf)
}

// ==========================================================================
//  PBiCGStab, distributed
// ==========================================================================

/// PBiCGStab over a decomposed mesh.
///
/// van der Vorst (1992); the same recurrence [`solver::solve_pbicgstab`] runs.
/// **Two exchanges and four reductions per iteration** - twice PCG's, because
/// the method has two matrix products and four inner products.
///
/// One deliberate difference from the serial solver. `solve_pbicgstab` fuses
/// `(t,s)` and `(t,t)` into `device_dot2`, which reads `t` once for both; this
/// takes them as two reductions. The fused kernel `solDot2Stage1` accumulates
/// `ab` and `aa` with the identical grid-stride walk and the identical
/// `blockSum_`, and `solSum2Stage2` combines each with the identical loop
/// `solSumStage2` uses, so **the two constructions are bitwise equal** -
/// `a_one_part_gathered_solve_is_the_serial_solver` proves it on both solvers.
/// The cost of not fusing is one extra pass over `t` per iteration, and the
/// reason not to fuse is that the exact accumulator's dot needs two passes
/// anyway (the anchor must precede the limbs, SPEC-LIT §72.3), so a fused
/// exact `dot2` would save nothing on the path that matters.
pub fn dist_pbicgstab(
    gpu: &Gpu,
    sol: &SolverKernels,
    psi: &mut [DevBuf<Scalar>],
    sys: &DistSystem,
    ex: &mut HaloExchange,
    w: &mut DistWorkspace,
    ctrl: &SolverControls,
) -> Result<DistPerformance> {
    let mut perf = DistPerformance {
        converged: true,
        ..Default::default()
    };
    w.check(sys, psi, "dist_pbicgstab")?;
    if w.n_global_cells == 0 {
        return Ok(perf);
    }
    perf.converged = false;

    let precon = w.effective_preconditioner(ctrl.precon)?;
    dist_build_preconditioner(gpu, sol, sys, w, precon)?;

    dist_norm_factor(gpu, sol, ex, sys, w, psi, &mut perf)?;
    dist_residual(gpu, sol, w)?;
    {
        let DistWorkspace { r0, r, n, .. } = &mut *w;
        for i in 0..n.len() {
            solver::vec_copy(gpu, sol, &mut r0[i], &r[i], n[i])?;
        }
    }
    {
        let DistWorkspace { red, r, initial_res, reduce, .. } = &mut *w;
        reduce_into(
            gpu,
            sol,
            red,
            *reduce,
            Reduction::SumMag(&r[..]),
            initial_res,
            &mut perf,
        )?;
    }
    solver::copy_scalar(gpu, sol, &mut w.final_res, &w.initial_res)?;

    for i in 0..sys.n_parts() {
        gpu.fill_zero(&mut w.p[i])?;
        gpu.fill_zero(&mut w.v[i])?;
    }
    gpu.fill_zero(&mut w.flag)?;
    solver::set_scalar(gpu, sol, &mut w.rho_old, 1.0)?;
    solver::set_scalar(gpu, sol, &mut w.alpha, 1.0)?;
    solver::set_scalar(gpu, sol, &mut w.omega, 1.0)?;

    let max_iter = ctrl.max_iter.max(0) as usize;
    let interval = ctrl.check_interval.max(1) as usize;
    let checking = !ctrl.fixed_iters;

    if checking {
        solver::convergence_test(
            gpu,
            sol,
            &mut w.flag,
            &w.initial_res,
            &w.initial_res,
            &w.norm_factor,
            ctrl,
            0,
        )?;
        perf.converged = dist_read_flag(gpu, &w.flag)?;
    }

    if !perf.converged {
        for it in 0..max_iter {
            let iters = it + 1;

            {
                let DistWorkspace { red, r0, r, rho, reduce, .. } = &mut *w;
                reduce_into(
                    gpu,
                    sol,
                    red,
                    *reduce,
                    Reduction::Dot(&r0[..], &r[..]),
                    rho,
                    &mut perf,
                )?;
            }
            unsafe {
                gpu.stream()
                    .launch_builder(&sol.beta_bicg)
                    .arg(&mut w.beta)
                    .arg(&w.rho)
                    .arg(&w.rho_old)
                    .arg(&w.alpha)
                    .arg(&w.omega)
                    .launch(solver::one_thread())?;
            }

            // p = r + beta (p - omega v)
            for i in 0..sys.n_parts() {
                let n = w.n[i];
                if n == 0 {
                    continue;
                }
                let nl = solver::to_label(n)?;
                unsafe {
                    gpu.stream()
                        .launch_builder(&sol.p_update)
                        .arg(&mut w.p[i])
                        .arg(&w.r[i])
                        .arg(&w.v[i])
                        .arg(&w.beta)
                        .arg(&w.omega)
                        .arg(&nl)
                        .launch(cfg_for(n))?;
                }
            }

            {
                let DistWorkspace { p_hat, p, r_diag, multicolour, n, .. } = &mut *w;
                dist_precondition(
                    gpu,
                    sol,
                    sys,
                    &mut p_hat[..],
                    &p[..],
                    &r_diag[..],
                    &multicolour[..],
                    precon,
                    &n[..],
                )?;
            }
            {
                let DistWorkspace { v, p_hat, .. } = &mut *w;
                dist_amul(gpu, sol, ex, sys, v, p_hat, &mut perf)?;
            }
            {
                let DistWorkspace { red, r0, v, den, reduce, .. } = &mut *w;
                reduce_into(
                    gpu,
                    sol,
                    red,
                    *reduce,
                    Reduction::Dot(&r0[..], &v[..]),
                    den,
                    &mut perf,
                )?;
            }
            solver::divide_scalar(gpu, sol, &mut w.alpha, &w.rho, &w.den)?;

            // s = r - alpha v
            for i in 0..sys.n_parts() {
                let n = w.n[i];
                if n == 0 {
                    continue;
                }
                let nl = solver::to_label(n)?;
                unsafe {
                    gpu.stream()
                        .launch_builder(&sol.s_update)
                        .arg(&mut w.s[i])
                        .arg(&w.r[i])
                        .arg(&w.v[i])
                        .arg(&w.alpha)
                        .arg(&nl)
                        .launch(cfg_for(n))?;
                }
            }

            {
                let DistWorkspace { s_hat, s, r_diag, multicolour, n, .. } = &mut *w;
                dist_precondition(
                    gpu,
                    sol,
                    sys,
                    &mut s_hat[..],
                    &s[..],
                    &r_diag[..],
                    &multicolour[..],
                    precon,
                    &n[..],
                )?;
            }
            {
                let DistWorkspace { t, s_hat, .. } = &mut *w;
                dist_amul(gpu, sol, ex, sys, t, s_hat, &mut perf)?;
            }
            {
                let DistWorkspace { red, t, s, num, reduce, .. } = &mut *w;
                reduce_into(
                    gpu,
                    sol,
                    red,
                    *reduce,
                    Reduction::Dot(&t[..], &s[..]),
                    num,
                    &mut perf,
                )?;
            }
            {
                let DistWorkspace { red, t, den, reduce, .. } = &mut *w;
                reduce_into(
                    gpu,
                    sol,
                    red,
                    *reduce,
                    Reduction::Dot(&t[..], &t[..]),
                    den,
                    &mut perf,
                )?;
            }
            solver::divide_scalar(gpu, sol, &mut w.omega, &w.num, &w.den)?;

            // psi += alpha pHat + omega sHat ;  r = s - omega t
            for i in 0..sys.n_parts() {
                let n = w.n[i];
                if n == 0 {
                    continue;
                }
                let nl = solver::to_label(n)?;
                unsafe {
                    gpu.stream()
                        .launch_builder(&sol.x_update)
                        .arg(&mut psi[i])
                        .arg(&w.p_hat[i])
                        .arg(&w.s_hat[i])
                        .arg(&w.alpha)
                        .arg(&w.omega)
                        .arg(&nl)
                        .launch(cfg_for(n))?;
                    gpu.stream()
                        .launch_builder(&sol.r_update)
                        .arg(&mut w.r[i])
                        .arg(&w.s[i])
                        .arg(&w.t[i])
                        .arg(&w.omega)
                        .arg(&nl)
                        .launch(cfg_for(n))?;
                }
            }
            solver::copy_scalar(gpu, sol, &mut w.rho_old, &w.rho)?;

            perf.n_iterations = iters;

            if checking && iters % interval == 0 {
                {
                    let DistWorkspace { red, r, final_res, reduce, .. } = &mut *w;
                    reduce_into(
                        gpu,
                        sol,
                        red,
                        *reduce,
                        Reduction::SumMag(&r[..]),
                        final_res,
                        &mut perf,
                    )?;
                }
                let itl = solver::to_label(iters)?;
                solver::convergence_test(
                    gpu,
                    sol,
                    &mut w.flag,
                    &w.final_res,
                    &w.initial_res,
                    &w.norm_factor,
                    ctrl,
                    itl,
                )?;
                if dist_read_flag(gpu, &w.flag)? {
                    perf.converged = true;
                    break;
                }
            }
        }
    }

    dist_finish(gpu, sol, ex, sys, w, psi, ctrl, &mut perf)?;
    Ok(perf)
}

// ==========================================================================
//  Dispatch - SPEC-LIT section 13.4 applied to the distributed path
// ==========================================================================

/// Solve `A psi = b` over a decomposition with the method the case asked for.
///
/// `GAMG` is refused by name for the same reason [`solver::solve`] refuses
/// it - it is a pressure *backend* here, not a Krylov method - and the
/// distributed path adds a second reason: no pressure backend has a decomposed
/// form at all (SPEC-LIT §73.7).
pub fn dist_solve(
    gpu: &Gpu,
    sol: &SolverKernels,
    psi: &mut [DevBuf<Scalar>],
    sys: &DistSystem,
    ex: &mut HaloExchange,
    w: &mut DistWorkspace,
    ctrl: &SolverControls,
) -> Result<DistPerformance> {
    match ctrl.solver {
        LinearSolverKind::PBiCGStab => dist_pbicgstab(gpu, sol, psi, sys, ex, w, ctrl),
        LinearSolverKind::PCG => dist_pcg(gpu, sol, psi, sys, ex, w, ctrl),
        LinearSolverKind::Gamg => {
            crate::io::contract::unsupported::<()>(
                "solvers/<var>/solver",
                "GAMG",
                &["PCG", "PBiCGStab"],
                "PBiCGStab: GAMG is a pressure backend in this code, not a \
                 Krylov method, and no pressure backend has a decomposed form",
                (),
            )?;
            dist_pbicgstab(gpu, sol, psi, sys, ex, w, ctrl)
        }
    }
}

#[cfg(test)]
mod tests;
