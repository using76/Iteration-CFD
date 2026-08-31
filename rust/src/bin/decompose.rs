// meteor-cfd - Copyright (c) 2026 주식회사 메테오시뮬레이션 (Meteo Simulation Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

//! `ofgpu-decompose` - cut a real case into parts, run it, and say whether the
//! answer moved.
//!
//! SPEC-LIT §71.8. The library test
//! `halo::tests::a_decomposed_run_is_bitwise_the_undecomposed_run` runs this
//! same pipeline on boxes written out by hand; this runs it on the meshes in
//! `cases/`, which is where the awkward geometry lives - graded spacing,
//! cyclic couples, patches of every kind, and a face count large enough that a
//! partition has real interior.
//!
//! ```text
//! ofgpu-decompose <case> [-parts 2,3,4] [-method hilbert|linear|roundrobin]
//!                        [-sweeps N] [-quiet]
//! ```
//!
//! What it runs is a fixed pipeline over one matrix:
//!
//! ```text
//! A  = laplacian(mesh) + ddt          assembled ONCE, on the host
//! relax(alpha) -> set_values -> add_boundary_contributions
//! N x  [ exchange halo ; psi += (b - A psi)/diag ]
//! ```
//!
//! and it compares the whole-mesh answer with the decomposed one **bit for
//! bit**.
//!
//! It then runs the third gate, SPEC-LIT §73.8: a real PCG and a real
//! PBiCGStab **over the decomposition**, solved to a tolerance rather than for
//! a fixed count, must produce the whole mesh's field bit for bit AND stop on
//! the same iteration - and then must do it again under every rotation of the
//! part labels, which is the same cells in the same groups owned by a
//! different rank and is the permutation a rank-indexed reduction would fail
//! while passing everything else. After that it measures what the block-local
//! DIC/DILU costs in iterations, which is the number §73.5 publishes.
//!
//! It also runs the second gate, SPEC-LIT §72.8: the same cut, and every
//! relabelling of its parts, must reduce to the same bits. Three reductions
//! over two fields - `ExactReduction::sum`, `::dot` and, for contrast, the
//! gathered-per-part-partial `::gathered_sum` - and the report says which
//! moved. The exact ones must not move; the gathered one is expected to, and
//! how often it does is the measurement §72.9 publishes.
//!
//! The matrix is assembled on the host and distributed by
//! `Decomposition::split_matrix` rather than re-assembled on each part,
//! because the assembly kernels are not yet partition-invariant - SPEC-LIT
//! §70.5 names the sixteen that are not and §71.9 says what is owed before
//! they can be. A run that re-assembled would be testing a claim this section
//! does not make.
//!
//! Provenance: ORIGINAL - the driver and its gate. No external source
//! (`PROVENANCE.md`, `src/bin/*`). No GPL-licensed source was consulted.

use std::path::PathBuf;

use ofgpu::decompose::{Decomposition, PartitionMethod};
use ofgpu::device::cfg_for;
use ofgpu::distsolve::{
    dist_solve, DistPerformance, DistReduce, DistSystem, DistWorkspace,
};
use ofgpu::exactsum::{ldexp, ExactReduction};
use ofgpu::halo::HaloExchange;
use ofgpu::ldu::{GpuLduMatrix, HostLduMatrix};
use ofgpu::ldu_ops::{self, LduKernels};
use ofgpu::mesh::GpuMesh;
use ofgpu::solver::{
    self, LinearSolverKind, Preconditioner, SolverControls, SolverKernels,
};
use ofgpu::{DevBuf, Error, Gpu, HostMesh, Label, Result, Scalar};

use cudarc::driver::PushKernelArg;

mod common;

const USAGE: &str = "\
usage: ofgpu-decompose <case> [options]

  -parts   2,3,4        part counts to try (default 2,3,4)
  -method  NAME         hilbert (default) | linear | roundrobin | all
  -sweeps  N            Jacobi sweeps per run (default 8)
  -alpha   A            under-relaxation factor (default 0.5)
  -quiet                print only the verdict
";

struct Args {
    case: PathBuf,
    parts: Vec<usize>,
    methods: Vec<String>,
    sweeps: usize,
    alpha: Scalar,
    quiet: bool,
}

fn parse() -> Result<Args> {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    if argv.is_empty() || argv[0] == "-h" || argv[0] == "--help" {
        println!("{USAGE}");
        std::process::exit(if argv.is_empty() { 2 } else { 0 });
    }
    let mut a = Args {
        case: PathBuf::from(&argv[0]),
        parts: vec![2, 3, 4],
        methods: vec!["hilbert".to_string()],
        sweeps: 8,
        alpha: 0.5,
        quiet: false,
    };
    let mut i = 1;
    while i < argv.len() {
        let need = |i: usize| -> Result<String> {
            argv.get(i + 1)
                .cloned()
                .ok_or_else(|| Error::Config(format!("{} needs a value", argv[i])))
        };
        match argv[i].as_str() {
            "-parts" => {
                a.parts = need(i)?
                    .split(',')
                    .map(|s| s.trim().parse::<usize>())
                    .collect::<std::result::Result<_, _>>()
                    .map_err(|e| Error::Config(format!("-parts: {e}")))?;
                i += 2;
            }
            "-method" => {
                let v = need(i)?;
                a.methods = if v == "all" {
                    vec![
                        "hilbert".to_string(),
                        "linear".to_string(),
                        "roundrobin".to_string(),
                    ]
                } else {
                    vec![v]
                };
                i += 2;
            }
            "-sweeps" => {
                a.sweeps = need(i)?
                    .parse()
                    .map_err(|e| Error::Config(format!("-sweeps: {e}")))?;
                i += 2;
            }
            "-alpha" => {
                a.alpha = need(i)?
                    .parse()
                    .map_err(|e| Error::Config(format!("-alpha: {e}")))?;
                i += 2;
            }
            "-quiet" => {
                a.quiet = true;
                i += 1;
            }
            other => {
                return Err(Error::Config(format!(
                    "unknown option '{other}'\n\n{USAGE}"
                )))
            }
        }
    }
    Ok(a)
}

fn method_named(name: &str, m: &HostMesh, np: usize) -> Result<PartitionMethod> {
    match name {
        "hilbert" => Ok(PartitionMethod::Hilbert),
        "linear" => Ok(PartitionMethod::Linear),
        // The worst partition there is, and therefore the most searching test:
        // cell c to part c % P cuts nearly every face in the mesh.
        "roundrobin" => Ok(PartitionMethod::Explicit(
            (0..m.n_cells).map(|c| (c % np) as Label).collect(),
        )),
        other => Err(Error::Config(format!(
            "unknown partition method '{other}'; known: hilbert, linear, \
             roundrobin, all"
        ))),
    }
}

/// A deterministic field of the GLOBAL cell id, so that distributing it is a
/// permutation and never a computation.
fn field(c: usize) -> Scalar {
    0.25 + 0.0625 * ((c * 37) % 23) as Scalar
}

/// A Laplacian plus a `ddt`, from the mesh's own metrics, with a scattering of
/// pinned cells chosen by GLOBAL id so the same cells are pinned however the
/// mesh is cut.
fn laplacian(m: &HostMesh) -> HostLduMatrix {
    let rdt = 1.5;
    let mut a = HostLduMatrix::zeros(m);
    for f in 0..m.n_internal_faces {
        let g = m.mag_sf[f] * m.delta_coeffs[f];
        a.upper[f] = g;
        a.lower[f] = g;
        a.diag[m.owner[f] as usize] -= g;
        a.diag[m.neighbour[f] as usize] -= g;
    }
    for bf in 0..m.n_boundary_faces {
        let g = m.b_mag_sf[bf] * m.b_delta_coeffs[bf];
        a.internal_coeffs[bf] = -g;
        a.boundary_coeffs[bf] = if m.b_nbr_cell[bf] >= 0 { -g } else { -g * 0.375 };
    }
    for c in 0..m.n_cells {
        let vd = m.v[c] * rdt;
        a.diag[c] -= vd;
        a.source[c] -= vd * field(c);
    }
    let stride = (m.n_cells / 64).max(7);
    for c in (0..m.n_cells).step_by(stride) {
        a.is_fixed[c] = 1;
        a.fixed_value[c] = 0.5 + 0.125 * (c % 5) as Scalar;
    }
    a
}

/// One part's device state. `psi` lives outside, in the vector the exchange
/// takes, which is the shape a distributed run has.
struct Rig {
    mesh: GpuMesh,
    a: GpuLduMatrix,
    n: usize,
    apsi: DevBuf<Scalar>,
    r: DevBuf<Scalar>,
    z: DevBuf<Scalar>,
    rdiag: DevBuf<Scalar>,
    one: DevBuf<Scalar>,
}

impl Rig {
    fn new(gpu: &Gpu, hm: &HostMesh, ha: &HostLduMatrix) -> Result<Self> {
        Ok(Self {
            mesh: GpuMesh::upload(gpu, hm)?,
            a: ha.upload(gpu)?,
            n: hm.n_cells,
            apsi: gpu.zeros(hm.n_cells)?,
            r: gpu.zeros(hm.n_cells)?,
            z: gpu.zeros(hm.n_cells)?,
            rdiag: gpu.zeros(hm.n_cells)?,
            one: gpu.upload(&[1.0 as Scalar])?,
        })
    }
}

/// `relax`, `set_values`, the fold, and `1/diag` - the order `src/ldu_ops.rs`
/// documents.
fn prepare(
    gpu: &Gpu,
    lk: &LduKernels,
    sk: &SolverKernels,
    rig: &mut Rig,
    psi: &DevBuf<Scalar>,
    alpha: Scalar,
) -> Result<()> {
    ldu_ops::relax(gpu, lk, &mut rig.a, &rig.mesh, psi, alpha)?;
    ldu_ops::set_values(gpu, lk, &mut rig.a, &rig.mesh)?;
    ldu_ops::add_boundary_contributions(gpu, lk, &mut rig.a, &rig.mesh)?;
    let n = rig.n as Label;
    unsafe {
        gpu.stream()
            .launch_builder(&sk.invert_diag)
            .arg(&mut rig.rdiag)
            .arg(&rig.a.diag)
            .arg(&n)
            .launch(cfg_for(rig.n))?;
    }
    Ok(())
}

/// `psi += (b - A psi)/diag`. Every step but the product is elementwise on one
/// cell's own values, so the product is the only thing a cut can move.
fn sweep(gpu: &Gpu, sk: &SolverKernels, rig: &mut Rig, psi: &mut DevBuf<Scalar>) -> Result<()> {
    let n = rig.n;
    let nl = n as Label;
    solver::amul(gpu, sk, &mut rig.apsi, psi, &rig.a, &rig.mesh)?;
    solver::vec_sub(gpu, sk, &mut rig.r, &rig.a.source, &rig.apsi, n)?;
    unsafe {
        gpu.stream()
            .launch_builder(&sk.precond_jacobi)
            .arg(&mut rig.z)
            .arg(&rig.r)
            .arg(&rig.rdiag)
            .arg(&nl)
            .launch(cfg_for(n))?;
        gpu.stream()
            .launch_builder(&sk.axpy)
            .arg(&mut *psi)
            .arg(&rig.z)
            .arg(&rig.one)
            .arg(&nl)
            .launch(cfg_for(n))?;
    }
    Ok(())
}


// ==========================================================================
//  SPEC-LIT §72 - the reductions
// ==========================================================================

/// A term with a full-width mantissa, a hundred-binade exponent range and
/// mixed signs, from one round of a 64-bit multiply-xorshift on the index.
///
/// The realistic field below - the solved `psi`, reduced against the cell
/// volumes - is what a solver actually reduces, and on a uniform box its
/// partial sums are nearly exact, so it barely distinguishes the two
/// constructions. This one does: a 52-bit mantissa spread over a hundred
/// binades puts a rounding into essentially every addition. Both are reported,
/// because "the cheap construction survives on easy data" is a fact about the
/// data and saying only the second would overstate the case.
fn nasty(i: usize) -> Scalar {
    let mut x = (i as u64)
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    x ^= x >> 33;
    x = x.wrapping_mul(0xff51_afd7_ed55_8ccd);
    x ^= x >> 33;

    let mant = 1.0 as Scalar + (x >> 12) as Scalar * ldexp(1.0, -52);
    let e = ((x >> 3) % 101) as i32 - 50;
    let s = if x & 1 == 0 { 1.0 } else { -1.0 as Scalar };
    s * ldexp(mant, e)
}

/// What one reduction of one field, over one decomposition, produced.
#[derive(Clone, Copy)]
struct Reduced {
    /// The exact accumulator: claimed partition-invariant.
    exact_sum: Scalar,
    exact_dot: Scalar,
    /// The gathered per-part partial: claimed run-invariant only.
    gathered: Scalar,
}

/// Run all three reductions over a set of already-split parts.
///
/// Every buffer is uploaded one value longer than the part owns and the extra
/// slot is poisoned, so a reduction that read past `n_cells` - which is what
/// summing a ghost cell would be - could not possibly agree.
fn reduce_over(
    gpu: &Gpu,
    sol: &SolverKernels,
    xs: &[Vec<Scalar>],
    ys: &[Vec<Scalar>],
    ns: &[usize],
) -> Result<Reduced> {
    let up = |v: &Vec<Scalar>| -> Result<DevBuf<Scalar>> {
        let mut d = v.clone();
        d.push(1.0e30);
        gpu.upload(&d)
    };
    let dx: Vec<DevBuf<Scalar>> = xs.iter().map(up).collect::<Result<_>>()?;
    let dy: Vec<DevBuf<Scalar>> = ys.iter().map(up).collect::<Result<_>>()?;

    let mut red = ExactReduction::new(gpu, ns)?;
    red.sum(gpu, sol, &dx)?;
    gpu.sync()?;
    let exact_sum = red.value(gpu)?;
    red.dot(gpu, sol, &dx, &dy)?;
    gpu.sync()?;
    let exact_dot = red.value(gpu)?;
    red.gathered_sum(gpu, sol, &dx)?;
    gpu.sync()?;
    let gathered = red.value(gpu)?;

    Ok(Reduced { exact_sum, exact_dot, gathered })
}

/// The whole mesh, as one part. Not a special case: `ExactReduction` runs the
/// identical code at `P = 1`.
fn reduce_whole(gpu: &Gpu, sol: &SolverKernels, x: &[Scalar], y: &[Scalar]) -> Result<Reduced> {
    reduce_over(gpu, sol, &[x.to_vec()], &[y.to_vec()], &[x.len()])
}

/// The same reduction over a decomposition, with the halo poisoned.
fn reduce_decomposed(
    gpu: &Gpu,
    sol: &SolverKernels,
    d: &Decomposition,
    x: &[Scalar],
    y: &[Scalar],
) -> Result<Reduced> {
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    let mut ns = Vec::new();
    for p in 0..d.n_parts {
        let owned = d.parts[p].mesh.n_cells;
        let mut fx = d.split_field(p, x)?;
        let mut fy = d.split_field(p, y)?;
        for h in fx.iter_mut().skip(owned) {
            *h = 1.0e30;
        }
        for h in fy.iter_mut().skip(owned) {
            *h = -1.0e30;
        }
        xs.push(fx);
        ys.push(fy);
        ns.push(owned);
    }
    reduce_over(gpu, sol, &xs, &ys, &ns)
}

/// `|got - want| / max(|want|, tiny)`, for reporting how far the gathered
/// construction moved. Never used as a pass criterion: the criterion is
/// `to_bits()` equality and nothing else.
fn rel(got: Scalar, want: Scalar) -> Scalar {
    let d = (got - want).abs();
    let w = want.abs();
    if w > 0.0 {
        d / w
    } else {
        d
    }
}

// ==========================================================================
//  SPEC-LIT §73 - the distributed Krylov solve
// ==========================================================================

/// A deterministic Dirichlet value per boundary cell, keyed on the GLOBAL cell
/// id so the same wall value is imposed however the mesh is cut.
fn wall(c: usize) -> Scalar {
    0.5 + 0.125 * ((c * 11) % 7) as Scalar
}

/// A **symmetric positive definite** Poisson operator plus a `ddt`, from the
/// mesh's own metrics, already folded on the host.
///
/// The §71 gate's `laplacian` above is deliberately not this: it is relaxed
/// and pinned, and its diagonal comes out of `relax` with the opposite sign,
/// which is fine for a Jacobi sweep and useless for PCG. A Krylov gate needs a
/// real SPD system - PCG minimises over a Krylov space in the `A` inner
/// product, and DIC is the Cholesky form - so this builds one.
///
/// Folded here rather than on each part because `split_matrix` sets
/// `internal_coeffs = 0` on a cut face, so the fold is a no-op there and
/// folding before or after the split gives the same matrix (SPEC-LIT §71.6).
fn poisson(m: &HostMesh) -> HostLduMatrix {
    let rdt = 1.5;
    let mut a = HostLduMatrix::zeros(m);
    for f in 0..m.n_internal_faces {
        let g = m.mag_sf[f] * m.delta_coeffs[f];
        a.upper[f] = -g;
        a.lower[f] = -g;
        a.diag[m.owner[f] as usize] += g;
        a.diag[m.neighbour[f] as usize] += g;
    }
    for bf in 0..m.n_boundary_faces {
        let g = m.b_mag_sf[bf] * m.b_delta_coeffs[bf];
        let c = m.b_face_cells[bf] as usize;
        a.diag[c] += g;
        if m.b_nbr_cell[bf] >= 0 {
            a.boundary_coeffs[bf] = g;
        } else {
            a.source[c] += g * wall(c);
        }
    }
    for c in 0..m.n_cells {
        let vd = m.v[c] * rdt;
        a.diag[c] += vd;
        a.source[c] += vd * field(c);
    }
    a
}

fn krylov_controls(
    solver: LinearSolverKind,
    precon: Preconditioner,
    fixed: Option<usize>,
) -> SolverControls {
    SolverControls {
        solver,
        precon,
        tolerance: 1.0e-10,
        rel_tol: 0.0,
        max_iter: fixed.map_or(600, |k| k as Label),
        min_iter: 0,
        check_interval: 1,
        fixed_iters: fixed.is_some(),
        report_residuals: !fixed.is_some(),
    }
}

/// Every part of a decomposition, uploaded once so a ladder of preconditioners
/// can be run over the same cut without re-splitting it.
struct DistRig {
    ex: HaloExchange,
    sys: DistSystem,
}

impl DistRig {
    fn new(gpu: &Gpu, m: &HostMesh, dec: &Decomposition, a: &HostLduMatrix) -> Result<Self> {
        let mut ex = HaloExchange::new(gpu, dec)?;
        let sys = DistSystem::split(gpu, &mut ex, m, dec, a)?;
        Ok(Self { ex, sys })
    }

    /// One distributed solve, gathered back into whole-mesh order.
    fn solve(
        &mut self,
        gpu: &Gpu,
        sk: &SolverKernels,
        dec: &Decomposition,
        initial: &[Scalar],
        ctrl: &SolverControls,
        reduce: DistReduce,
    ) -> Result<(Vec<Scalar>, DistPerformance)> {
        let mut w = DistWorkspace::new(gpu, &self.sys, reduce)?;
        if matches!(ctrl.precon, Preconditioner::Dic | Preconditioner::Dilu) {
            w.colour(gpu, &self.sys)?;
        }
        let mut psi: Vec<DevBuf<Scalar>> = (0..dec.n_parts)
            .map(|p| gpu.upload(&dec.split_field(p, initial)?))
            .collect::<Result<_>>()?;
        let perf = dist_solve(gpu, sk, &mut psi, &self.sys, &mut self.ex, &mut w, ctrl)?;
        gpu.sync()?;
        let per_part: Vec<Vec<Scalar>> =
            psi.iter().map(|b| gpu.download(b)).collect::<Result<_>>()?;
        Ok((dec.gather_field(&per_part)?, perf))
    }

    /// Seconds per iteration, with the residual test and the host round trip
    /// switched off so what is timed is the recurrence and nothing else.
    fn seconds_per_iteration(
        &mut self,
        gpu: &Gpu,
        sk: &SolverKernels,
        dec: &Decomposition,
        initial: &[Scalar],
        kind: LinearSolverKind,
        iters: usize,
    ) -> Result<f64> {
        let ctrl = krylov_controls(kind, Preconditioner::Diagonal, Some(iters));
        // Warm up: the first solve pays for module loads and the colouring.
        self.solve(gpu, sk, dec, initial, &ctrl, DistReduce::Exact)?;
        let t0 = std::time::Instant::now();
        self.solve(gpu, sk, dec, initial, &ctrl, DistReduce::Exact)?;
        gpu.sync()?;
        Ok(t0.elapsed().as_secs_f64() / iters as f64)
    }
}

fn main() -> Result<()> {
    let args = parse()?;
    let gpu = Gpu::new(0)?;
    let (m, _cc, _lowered) = common::load_case(&args.case)?;

    if !args.quiet {
        println!("case   : {}", args.case.display());
        println!("device : {}", gpu.name()?);
        println!(
            "mesh   : {} cells, {} internal faces, {} boundary faces, {} patches",
            m.n_cells,
            m.n_internal_faces,
            m.n_boundary_faces,
            m.patches.len()
        );
        println!("sweeps : {}   alpha {}", args.sweeps, args.alpha);
        println!();
    }

    let lk = LduKernels::new(&gpu)?;
    let sk = SolverKernels::new(&gpu)?;
    let a0 = laplacian(&m);
    let psi_in: Vec<Scalar> = (0..m.n_cells).map(field).collect();

    // ---- the undecomposed run, once ---------------------------------------
    let mut serial = Rig::new(&gpu, &m, &a0)?;
    let mut spsi = gpu.upload(&psi_in)?;
    prepare(&gpu, &lk, &sk, &mut serial, &spsi, args.alpha)?;
    for _ in 0..args.sweeps {
        sweep(&gpu, &sk, &mut serial, &mut spsi)?;
    }
    gpu.sync()?;
    let want = gpu.download(&spsi)?;

    // ---- SPEC-LIT §72: the reductions, on the whole mesh ------------------
    // Two fields. `want` is the run's own answer, reduced against the cell
    // volumes - what a solver actually reduces. `nasty` is adversarial, and it
    // is here because the realistic field on a well-conditioned box does not
    // distinguish the two constructions and reporting only it would flatter
    // the cheap one.
    let vol: Vec<Scalar> = m.v.clone();
    let hard: Vec<Scalar> = (0..m.n_cells).map(nasty).collect();
    let real_whole = reduce_whole(&gpu, &sk, &want, &vol)?;
    let hard_whole = reduce_whole(&gpu, &sk, &hard, &vol)?;

    let mut runs = 0usize;
    let mut failed = 0usize;
    let mut red_runs = 0usize;
    let mut red_failed = 0usize;
    let mut gathered_moved_real = 0usize;
    let mut gathered_moved_hard = 0usize;
    let mut worst_gathered = 0.0 as Scalar;
    let mut kry_runs = 0usize;
    let mut kry_failed = 0usize;
    let mut rel_runs = 0usize;
    let mut rel_failed = 0usize;

    // ---- SPEC-LIT §73: the whole-mesh reference solve ---------------------
    // A separate, symmetric positive definite operator: the Jacobi gate's
    // matrix is relaxed and pinned and its diagonal comes out of `relax` with
    // the opposite sign, which is fine for a sweep and useless for PCG.
    let a_spd = poisson(&m);
    let one = Decomposition::build(&m, 1, &PartitionMethod::Linear)?;
    let mut one_rig = DistRig::new(&gpu, &m, &one, &a_spd)?;
    let mut krylov_want: Vec<(LinearSolverKind, &'static str, Vec<Scalar>, usize)> = Vec::new();
    for (kind, label) in [
        (LinearSolverKind::PCG, "PCG"),
        (LinearSolverKind::PBiCGStab, "PBiCGStab"),
    ] {
        let ctrl = krylov_controls(kind, Preconditioner::Diagonal, None);
        let (want, perf) = one_rig.solve(&gpu, &sk, &one, &psi_in, &ctrl, DistReduce::Exact)?;
        if !perf.converged {
            return Err(Error::Config(format!(
                "{label} did not converge on the whole mesh in {} iterations; \
                 the Krylov gate has no reference to compare against",
                perf.n_iterations
            )));
        }
        println!(
            "{label:<10} whole mesh: {} iterations to 1e-10, final residual {:.3e}",
            perf.n_iterations, perf.final_residual
        );
        krylov_want.push((kind, label, want, perf.n_iterations));
    }
    println!();

    for &np in &args.parts {
        for name in &args.methods {
            let method = method_named(name, &m, np)?;
            let d = Decomposition::build(&m, np, &method)?;
            let rep = d.report();
            let mut ex = HaloExchange::new(&gpu, &d)?;

            let split: Vec<HostLduMatrix> = (0..np)
                .map(|p| d.split_matrix(&m, p, &a0))
                .collect::<Result<_>>()?;
            let mut rigs: Vec<Rig> = (0..np)
                .map(|p| Rig::new(&gpu, &d.parts[p].mesh, &split[p]))
                .collect::<Result<_>>()?;

            // `is_fixed` and `fixed_value` reach the halo through the
            // exchange, not from the host's whole-mesh copy: a distributed
            // build has no such copy.
            let mut isf: Vec<DevBuf<Label>> = (0..np)
                .map(|p| gpu.upload(&split[p].is_fixed))
                .collect::<Result<_>>()?;
            let mut fv: Vec<DevBuf<Scalar>> = (0..np)
                .map(|p| gpu.upload(&split[p].fixed_value))
                .collect::<Result<_>>()?;
            ex.label(&gpu, &mut isf)?;
            ex.scalar(&gpu, &mut fv)?;
            for (rig, (i, v)) in rigs.iter_mut().zip(isf.into_iter().zip(fv)) {
                rig.a.is_fixed = i;
                rig.a.fixed_value = v;
            }

            let mut psi: Vec<DevBuf<Scalar>> = (0..np)
                .map(|p| gpu.upload(&d.split_field(p, &psi_in)?))
                .collect::<Result<_>>()?;
            ex.scalar(&gpu, &mut psi)?;
            for (p, rig) in rigs.iter_mut().enumerate() {
                prepare(&gpu, &lk, &sk, rig, &psi[p], args.alpha)?;
            }
            for _ in 0..args.sweeps {
                ex.scalar(&gpu, &mut psi)?;
                for (p, rig) in rigs.iter_mut().enumerate() {
                    sweep(&gpu, &sk, rig, &mut psi[p])?;
                }
            }
            gpu.sync()?;

            let per_part: Vec<Vec<Scalar>> = psi
                .iter()
                .map(|b| gpu.download(b))
                .collect::<Result<_>>()?;
            let got = d.gather_field(&per_part)?;

            let mut bad = 0usize;
            let mut worst = 0.0 as Scalar;
            let mut first = usize::MAX;
            for c in 0..m.n_cells {
                if got[c].to_bits() != want[c].to_bits() {
                    bad += 1;
                    if first == usize::MAX {
                        first = c;
                    }
                    let e = (got[c] - want[c]).abs();
                    if e > worst {
                        worst = e;
                    }
                }
            }
            runs += 1;
            if bad > 0 {
                failed += 1;
            }

            if !args.quiet {
                println!(
                    "P = {np}  {name:<11} cells {:?}  halo {:?}  neighbours {:?}",
                    rep.cells, rep.halo, rep.nbrs
                );
                println!(
                    "            cut faces {}  cut couples {}  imbalance {:.3}  \
                     halo/cells {:.4}",
                    rep.n_cut_faces, rep.n_cut_couples, rep.imbalance, rep.halo_fraction
                );
            }
            if bad == 0 {
                println!("P = {np}  {name:<11} PASS - bitwise identical on all {} cells", m.n_cells);
            } else {
                println!(
                    "P = {np}  {name:<11} FAIL - {bad} of {} cells differ, first at {first}, \
                     worst |d| {worst:e}",
                    m.n_cells
                );
            }
            // ---- SPEC-LIT §72: the same cut, the reductions ------------
            // Every part-label rotation is run as well as the map itself: the
            // same cells in the same groups with only the names changed is the
            // purest permutation there is, and an accumulator that is merely
            // deterministic for one labelling would pass without it.
            let mut red_bad = 0usize;
            let mut moved_real = 0usize;
            let mut moved_hard = 0usize;
            let mut worst_here = 0.0 as Scalar;
            for shift in 0..np {
                let rotated: Vec<Label> = d
                    .cell_part
                    .iter()
                    .map(|&r| ((r as usize + shift) % np) as Label)
                    .collect();
                let dr = Decomposition::from_map(&m, np, rotated)?;
                let r = reduce_decomposed(&gpu, &sk, &dr, &want, &vol)?;
                let h = reduce_decomposed(&gpu, &sk, &dr, &hard, &vol)?;
                red_runs += 1;
                if r.exact_sum.to_bits() != real_whole.exact_sum.to_bits()
                    || r.exact_dot.to_bits() != real_whole.exact_dot.to_bits()
                    || h.exact_sum.to_bits() != hard_whole.exact_sum.to_bits()
                    || h.exact_dot.to_bits() != hard_whole.exact_dot.to_bits()
                {
                    red_bad += 1;
                    red_failed += 1;
                }
                if r.gathered.to_bits() != real_whole.gathered.to_bits() {
                    moved_real += 1;
                    gathered_moved_real += 1;
                    worst_here = worst_here.max(rel(r.gathered, real_whole.gathered));
                }
                if h.gathered.to_bits() != hard_whole.gathered.to_bits() {
                    moved_hard += 1;
                    gathered_moved_hard += 1;
                    worst_here = worst_here.max(rel(h.gathered, hard_whole.gathered));
                }
            }
            worst_gathered = worst_gathered.max(worst_here);
            if red_bad == 0 {
                println!(
                    "P = {np}  {name:<11} REDUCTIONS PASS - exact sum and dot \
                     bitwise identical over {np} relabelling(s)"
                );
            } else {
                println!(
                    "P = {np}  {name:<11} REDUCTIONS FAIL - {red_bad} of {np} relabelling(s) moved"
                );
            }
            if !args.quiet {
                println!(
                    "            gathered partial moved in {moved_real} of {np} \
                     (run field) and {moved_hard} of {np} (adversarial), \
                     worst |d|/|s| {worst_here:e}"
                );
            }

            // ---- SPEC-LIT §73: the same cut, a real Krylov solve -------
            // Solved to a tolerance, not for a fixed count, so the CONVERGENCE
            // TEST is gated too: it reads a residual that the exact
            // accumulator did not move, so it must stop on the same iteration.
            let mut krig = DistRig::new(&gpu, &m, &d, &a_spd)?;
            for (kind, label, want, want_iters) in &krylov_want {
                let ctrl = krylov_controls(*kind, Preconditioner::Diagonal, None);
                let (got, perf) = krig.solve(&gpu, &sk, &d, &psi_in, &ctrl, DistReduce::Exact)?;
                kry_runs += 1;
                let mut bad = 0usize;
                let mut worst_k = 0.0 as Scalar;
                for c in 0..m.n_cells {
                    if got[c].to_bits() != want[c].to_bits() {
                        bad += 1;
                        worst_k = worst_k.max((got[c] - want[c]).abs());
                    }
                }
                if bad == 0 && perf.n_iterations == *want_iters {
                    println!(
                        "P = {np}  {name:<11} {label:<10} KRYLOV PASS - bitwise \
                         identical, {} iterations, {} exchanges, {} reductions",
                        perf.n_iterations, perf.n_exchanges, perf.n_reductions
                    );
                } else {
                    kry_failed += 1;
                    println!(
                        "P = {np}  {name:<11} {label:<10} KRYLOV FAIL - {bad} of {} \
                         cells differ (worst |d| {worst_k:e}), {} iterations \
                         against the whole mesh's {want_iters}",
                        m.n_cells, perf.n_iterations
                    );
                }
            }

            // ---- SPEC-LIT §73: the same cut, RELABELLED ----------------
            // The gate above changes the cut. This changes only the NAMES of
            // the parts: the same cells, in the same groups, owned by a
            // different rank. It is the purest permutation there is, and it
            // is the one a rank-indexed reduction or a rank-ordered gather
            // would fail while passing everything above - which is exactly
            // what §72.5 measured the gathered construction doing. The whole
            // orbit is covered: `shift = 0` is the identity, which is the
            // gate immediately above, and the other `P - 1` rotations are
            // here.
            let rel_bad_before = rel_failed;
            for shift in 1..np {
                let rotated: Vec<Label> = d
                    .cell_part
                    .iter()
                    .map(|&r| ((r as usize + shift) % np) as Label)
                    .collect();
                let dr = Decomposition::from_map(&m, np, rotated)?;
                let mut rrig = DistRig::new(&gpu, &m, &dr, &a_spd)?;
                for (kind, label, want, want_iters) in &krylov_want {
                    let ctrl = krylov_controls(*kind, Preconditioner::Diagonal, None);
                    let (got, perf) =
                        rrig.solve(&gpu, &sk, &dr, &psi_in, &ctrl, DistReduce::Exact)?;
                    rel_runs += 1;
                    let bad = got
                        .iter()
                        .zip(want.iter())
                        .filter(|(g, w)| g.to_bits() != w.to_bits())
                        .count();
                    if bad > 0 || perf.n_iterations != *want_iters {
                        rel_failed += 1;
                        println!(
                            "P = {np}  {name:<11} {label:<10} RELABEL FAIL - \
                             rotation {shift}: {bad} of {} cells differ, {} \
                             iterations against {want_iters}",
                            m.n_cells, perf.n_iterations
                        );
                    }
                }
            }
            if np > 1 && rel_failed == rel_bad_before {
                println!(
                    "P = {np}  {name:<11} RELABEL PASS - the solve is bitwise \
                     unmoved by all {} rotation(s) of the part labels",
                    np - 1
                );
            }

            if !args.quiet {
                println!();
            }
        }
    }

    // ======================================================================
    //  SPEC-LIT §73.5 - what the block-local factorisation costs
    // ======================================================================
    let ladder: Vec<usize> = [1usize, 2, 4, 8, 16]
        .into_iter()
        .filter(|&p| p <= m.n_cells)
        .collect();
    let mut rigs: Vec<(usize, Decomposition, DistRig)> = Vec::new();
    for &p in &ladder {
        let d = Decomposition::build(&m, p, &PartitionMethod::Hilbert)?;
        let r = DistRig::new(&gpu, &m, &d, &a_spd)?;
        rigs.push((p, d, r));
    }

    println!();
    println!(
        "SPEC-LIT §73.5 - iterations to 1e-10, Hilbert cut, exact reduction.\n  \
         diagonal is elementwise and partition-invariant, so its count CANNOT \
         move.\n  DIC and DILU are factorised on each part's own submatrix \
         with the couplings\n  across every cut dropped: block Jacobi, not \
         restricted additive Schwarz."
    );
    print!("  {:<22}", "preconditioner");
    for &p in &ladder {
        print!("{:>10}", format!("P={p}"));
    }
    println!("{:>12}", "P=max/P=1");

    for (kind, klabel, precon, plabel) in [
        (LinearSolverKind::PCG, "PCG", Preconditioner::Diagonal, "diagonal"),
        (LinearSolverKind::PCG, "PCG", Preconditioner::Dic, "DIC block-local"),
        (
            LinearSolverKind::PBiCGStab,
            "BiCG",
            Preconditioner::Diagonal,
            "diagonal",
        ),
        (
            LinearSolverKind::PBiCGStab,
            "BiCG",
            Preconditioner::Dilu,
            "DILU block-local",
        ),
    ] {
        let ctrl = krylov_controls(kind, precon, None);
        let mut counts = Vec::with_capacity(ladder.len());
        let mut colours: Vec<usize> = Vec::new();
        for (p, d, rig) in rigs.iter_mut() {
            let (_, perf) = rig.solve(&gpu, &sk, d, &psi_in, &ctrl, DistReduce::Exact)?;
            if !perf.converged {
                println!(
                    "  {klabel} {plabel} at P = {p}: DID NOT CONVERGE in {} \
                     iterations",
                    perf.n_iterations
                );
            }
            counts.push(perf.n_iterations);
            if matches!(precon, Preconditioner::Dic | Preconditioner::Dilu) {
                let mut w = DistWorkspace::new(&gpu, &rig.sys, DistReduce::Exact)?;
                w.colour(&gpu, &rig.sys)?;
                colours.extend(w.colours());
            }
        }
        print!("  {:<22}", format!("{klabel} {plabel}"));
        for c in &counts {
            print!("{c:>10}");
        }
        let ratio = *counts.last().unwrap_or(&0) as Scalar / counts[0].max(1) as Scalar;
        println!("{:>12}", format!("{ratio:.2}x"));
        if !colours.is_empty() && !args.quiet {
            println!(
                "    colours per part over the ladder: {} .. {}",
                colours.iter().min().copied().unwrap_or(0),
                colours.iter().max().copied().unwrap_or(0)
            );
        }
    }

    // ======================================================================
    //  SPEC-LIT §73.6 - the strong-scaling inputs that CAN be measured here
    // ======================================================================
    // This machine has one GPU. A strong-scaling number cannot be measured on
    // it and is not invented here. What is measured is (a) the cost of one
    // iteration on the whole mesh, from which the per-cell cost follows, and
    // (b) the cost of running the same iteration as P parts on the SAME
    // device, which is the launch and halo overhead a decomposition adds
    // before any network exists. The one input that is NOT measurable here -
    // the latency of a collective between two GPUs - is left as a named free
    // parameter and the crossover is quoted for two plausible values.
    println!();
    println!("SPEC-LIT §73.6 - per-iteration cost, ONE device, all parts run in sequence");
    let mut t_one = 0.0f64;
    for (p, d, rig) in rigs.iter_mut() {
        let t = rig.seconds_per_iteration(&gpu, &sk, d, &psi_in, LinearSolverKind::PCG, 20)?;
        if *p == 1 {
            t_one = t;
        }
        println!(
            "  P = {p:<3} {:8.1} us/iteration   {:5.2}x the whole mesh's   \
             halo/cells {:.4}",
            t * 1e6,
            if t_one > 0.0 { t / t_one } else { 1.0 },
            d.report().halo_fraction
        );
    }
    let t_cell = t_one / m.n_cells as f64;
    println!(
        "  {:.4} ns per cell per PCG iteration on this device",
        t_cell * 1e9
    );
    println!(
        "  PCG spends 2 reductions and 1 exchange per iteration (PBiCGStab 4 and 2);\n  \
         at L us per collective, communication costs 3L us and the compute is \
         {:.4} ns x cells-per-GPU, so the two are equal at",
        t_cell * 1e9
    );
    for l in [3.0f64, 5.0, 10.0] {
        println!(
            "    L = {l:>4.0} us  ->  {:>10.0} cells per GPU",
            3.0 * l * 1e-6 / t_cell
        );
    }
    println!(
        "  That crossover is only meaningful where an iteration is BANDWIDTH \
         bound.\n  Below roughly half a million cells this device is launch-\
         overhead bound, so the\n  ns/cell above is an overhead figure and the \
         crossover it gives is not a limit."
    );
    println!(
        "  These are on-device numbers. The exchange here is a memcpy_dtod and \
         the gather\n  is a one-block kernel, so both are LOWER BOUNDS on what \
         a real fabric costs.\n  A measured strong-scaling number needs a \
         second GPU and is refused by name in §73.7."
    );

    // ---- SPEC-LIT §72.9: what the accumulator costs ----------------------
    // A dot product over the whole mesh, timed both ways on the same data.
    // The literature quotes 1.1-2x on the reduction kernel for binned schemes
    // and measures nothing about this one, so this is measured here rather
    // than quoted. Both loops are the identical shape - one launch of stage
    // one, one of stage two, one 8-byte read - so the ratio is the
    // accumulator's, not the harness's.
    {
        let reps = 200;
        let dx = gpu.upload(&hard)?;
        let dy = gpu.upload(&vol)?;
        let mut out: DevBuf<Scalar> = gpu.zeros(1)?;
        let mut partials: DevBuf<Scalar> = gpu.zeros(solver::reduce_partitions(m.n_cells))?;
        let mut red = ExactReduction::new(&gpu, &[m.n_cells])?;
        let xs = vec![dx.clone()];
        let ys = vec![dy.clone()];

        // Warm up both paths, then time both.
        solver::device_dot(&gpu, &sk, &mut out, &dx, &dy, &mut partials, m.n_cells)?;
        red.dot(&gpu, &sk, &xs, &ys)?;
        gpu.sync()?;

        let t0 = std::time::Instant::now();
        for _ in 0..reps {
            solver::device_dot(&gpu, &sk, &mut out, &dx, &dy, &mut partials, m.n_cells)?;
        }
        gpu.sync()?;
        let plain = t0.elapsed().as_secs_f64() / reps as f64;

        let t1 = std::time::Instant::now();
        for _ in 0..reps {
            red.dot(&gpu, &sk, &xs, &ys)?;
        }
        gpu.sync()?;
        let exact = t1.elapsed().as_secs_f64() / reps as f64;

        println!(
            "dot over {} cells: plain {:.1} us, exact {:.1} us, ratio {:.2}x \
             (SPEC-LIT §72.6)",
            m.n_cells,
            plain * 1e6,
            exact * 1e6,
            exact / plain
        );
    }

    println!();
    println!("{}/{} decompositions bitwise identical to the whole mesh", runs - failed, runs);
    println!(
        "{}/{} decomposed Krylov solves bitwise identical to the whole mesh, \
         same iteration count",
        kry_runs - kry_failed,
        kry_runs
    );
    println!(
        "{}/{} relabelled decompositions solve to the same bits and the same \
         iteration count",
        rel_runs - rel_failed,
        rel_runs
    );
    println!(
        "{}/{} relabelled decompositions reduce to the same bits with the exact accumulator",
        red_runs - red_failed,
        red_runs
    );
    println!(
        "the gathered-partial construction moved in {gathered_moved_real} of \
         {red_runs} (run field) and {gathered_moved_hard} of {red_runs} \
         (adversarial field), worst |d|/|s| {worst_gathered:e} - SPEC-LIT §72.9"
    );
    if failed > 0 || red_failed > 0 || kry_failed > 0 || rel_failed > 0 {
        std::process::exit(1);
    }
    Ok(())
}
