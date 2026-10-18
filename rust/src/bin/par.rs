// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! `ofgpu-par` - the parallel driver: run SPEC-LIT §71-§73's gates with one
//! rank per part, every rank on its own `Gpu`, talking only through a `Comm`.
//!
//! SPEC-LIT §138.8. `ofgpu-decompose` runs the same three gates with every
//! part in one process; this driver runs them SPMD, the way a decomposed
//! solve will run: rank `r` computes the partition map itself, hashes it,
//! builds its own decomposition and holds part `r` only. Every byte that
//! crosses ranks goes through the transport (§138.3, §138.4, §138.5).
//!
//! ```text
//! ofgpu-par <case> [-np 2,3,4] [-comm threads] [-method all]
//!                  [-sweeps N] [-alpha A] [-quiet]
//! ofgpu-par -solve simple [-np 1,2,4] [-n N] [-method NAME] [-quiet]
//! ```
//!
//! `-solve simple` is SPEC-LIT 139.4's gate (PAR-10c): the lid-driven cavity
//! of Ghia, Ghia & Shin (1982) at Re = 100, run serial on the whole mesh and
//! then decomposed per rank count with `Simple`'s exchange seam, every rank
//! printing its L_inf differences against the serial answer.
//!
//! For every rank count, every partitioner and every rotation of the part
//! labels, in one `ThreadComm` world per (rank count, partitioner):
//!
//! * MAP - each rank's FNV-1a hash of the map, allgathered, is one value,
//!   and the main thread's;
//! * HALO - four payloads exchanged through `SpmdHalo` equal the owner's
//!   values in every local cell, and §71.8's Jacobi pipeline equals the
//!   undecomposed run in every owned cell;
//! * REDUCTION - `SpmdReduction`'s exact sum and dot of two fields equal the
//!   whole mesh's `ExactReduction`;
//! * KRYLOV - `spmd_solve` PCG and PBiCGStab equal the whole mesh's solve in
//!   every owned cell and stop on the same iteration.
//!
//! Every comparison is `to_bits()` equality, so "worst == 0" means no cell
//! differs. The exit status is 0 only when every gate passes.
//!
//! `-comm mpi` and an `mpiexec` environment (`PMI_RANK` or `PMI_SIZE` set)
//! are refused by name: the MPI transport is PAR-06, deferred by the user.
//!
//! Provenance: ORIGINAL - the driver and its gates. The matrices, fields and
//! sweep pipeline are `ofgpu-decompose`'s, so the two drivers gate the same
//! systems. No external source. No GPL-licensed source was consulted.

use std::path::PathBuf;
use std::time::Instant;

use ofgpu::comm::{Comm, Placement, ThreadComm};
use ofgpu::decompose::{partition, Decomposition, PartMesh, PartitionMethod};
use ofgpu::device::cfg_for;
use ofgpu::distflow::{
    cavity_ghia_errors, cavity_mesh, cavity_part, cavity_serial, CavityRun, CAVITY_ITER_BAND,
    CAVITY_N, CAVITY_TOL_CENTRE, CAVITY_TOL_FIELD, CAVITY_U_TOL,
};
use ofgpu::distsolve::{
    dist_solve, spmd_solve, DistReduce, DistSystem, DistWorkspace, SpmdWorkspace,
};
use ofgpu::exactsum::{ldexp, ExactReduction, SpmdReduction};
use ofgpu::halo::{HaloExchange, SpmdHalo};
use ofgpu::ldu::{GpuLduMatrix, HostLduMatrix};
use ofgpu::ldu_ops::{self, LduKernels};
use ofgpu::mesh::GpuMesh;
use ofgpu::solver::{
    self, LinearSolverKind, Preconditioner, SolverControls, SolverKernels,
};
use ofgpu::{DevBuf, Error, Gpu, HostMesh, Label, Result, Scalar, Tensor, Vec3};

use cudarc::driver::PushKernelArg;

mod common;

const USAGE: &str = "\
usage: ofgpu-par <case> [options]

  -np      2,3,4        rank counts to run (default 2,3,4; solve mode 1,2,4)
  -comm    threads      the transport (default threads): one process, one
                        thread per rank. mpi is refused: the MPI transport
                        is PAR-06, deferred
  -method  NAME         hilbert | linear | roundrobin | all (default all;
                        solve mode: one of the three, default hilbert)
  -sweeps  N            Jacobi sweeps in the halo gate (default 8)
  -alpha   A            under-relaxation factor (default 0.5)
  -quiet                print only the gate lines and the verdict
  -solve   simple       run the decomposed-cavity gate of SPEC-LIT 139.4
                        (PAR-10c) instead of the case gates: no <case> is
                        read, the mesh is the generator's cavity
  -n      N             cells per side of the -solve cavity (default 64)

Rank r opens its own Gpu on device L[r mod |L|], L = OFGPU_DEVICES or every
device (SPEC-LIT 138.7). Under mpiexec (PMI_RANK or PMI_SIZE set) the driver
refuses: run one process with -comm threads.
";

/// The solver tolerance: a solver control, not a gate tolerance - every
/// comparison is exact. `ofgpu-decompose`'s value in f64,
/// `tests/distsolve_spmd.rs`'s under `--features single`.
#[cfg(feature = "single")]
const TOL: Scalar = 1e-5;
#[cfg(not(feature = "single"))]
const TOL: Scalar = 1e-10;

/// The environment variables a PMI launcher (`mpiexec`) sets.
const PMI_VARS: [&str; 2] = ["PMI_RANK", "PMI_SIZE"];

struct Args {
    case: PathBuf,
    nps: Vec<usize>,
    comm: String,
    methods: Vec<String>,
    sweeps: usize,
    alpha: Scalar,
    quiet: bool,
}

fn parse(argv: &[String]) -> Result<Args> {
    let mut a = Args {
        case: PathBuf::from(&argv[0]),
        nps: vec![2, 3, 4],
        comm: "threads".to_string(),
        methods: vec![
            "hilbert".to_string(),
            "linear".to_string(),
            "roundrobin".to_string(),
        ],
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
            "-np" => {
                let v = need(i)?;
                a.nps = v
                    .split(',')
                    .map(|s| s.trim().parse::<usize>())
                    .collect::<std::result::Result<_, _>>()
                    .map_err(|e| Error::Config(format!("-np '{v}': {e}")))?;
                if a.nps.iter().any(|&n| n == 0) {
                    return Err(Error::Config(format!(
                        "-np '{v}': a rank count must be at least 1"
                    )));
                }
                i += 2;
            }
            "-comm" => {
                a.comm = need(i)?;
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

/// The options of `-solve simple`, SPEC-LIT 139.4's decomposed-cavity gate.
/// No `<case>` is read; the mesh is `distflow::cavity_mesh(n)`.
struct SolveArgs {
    n: usize,
    nps: Vec<usize>,
    methods: Vec<String>,
    quiet: bool,
}

fn parse_solve(argv: &[String]) -> Result<SolveArgs> {
    let mut a = SolveArgs {
        n: CAVITY_N,
        nps: vec![1, 2, 4],
        methods: vec!["hilbert".to_string()],
        quiet: false,
    };
    let mut i = 0;
    while i < argv.len() {
        let need = |i: usize| -> Result<String> {
            argv.get(i + 1)
                .cloned()
                .ok_or_else(|| Error::Config(format!("{} needs a value", argv[i])))
        };
        match argv[i].as_str() {
            "-n" => {
                a.n = need(i)?
                    .parse()
                    .map_err(|e| Error::Config(format!("-n: {e}")))?;
                if a.n == 0 {
                    return Err(Error::Config("-n: cells per side must be >= 1".to_string()));
                }
                i += 2;
            }
            "-np" => {
                let v = need(i)?;
                a.nps = v
                    .split(',')
                    .map(|s| s.trim().parse::<usize>())
                    .collect::<std::result::Result<_, _>>()
                    .map_err(|e| Error::Config(format!("-np '{v}': {e}")))?;
                if a.nps.iter().any(|&n| n == 0) {
                    return Err(Error::Config(format!(
                        "-np '{v}': a rank count must be at least 1"
                    )));
                }
                i += 2;
            }
            "-method" => {
                let v = need(i)?;
                a.methods = if v == "all" {
                    vec!["hilbert".to_string()]
                } else {
                    vec![v]
                };
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

/// The transport, refused by name unless it is `threads`. Runs before the
/// case is read or any device is opened.
///
/// A PMI launcher is refused whatever `-comm` says: under `mpiexec` every
/// process would otherwise run a whole threads world of its own.
fn refuse_unbuilt_transport(comm: &str) -> Result<()> {
    for var in PMI_VARS {
        if std::env::var_os(var).is_some() {
            return Err(Error::Config(format!(
                "{var} is set, so mpiexec (or another PMI launcher) started \
                 this process: the MPI transport is deferred (PAR-06) and is \
                 not built. Run one process with -comm threads"
            )));
        }
    }
    match comm {
        "threads" => Ok(()),
        "mpi" => Err(Error::Config(
            "-comm mpi: the MPI transport is deferred (PAR-06) and is not \
             built. Use -comm threads"
                .to_string(),
        )),
        other => Err(Error::Config(format!(
            "-comm '{other}': unknown transport; known: threads (mpi is \
             deferred, PAR-06)"
        ))),
    }
}

fn method_named(name: &str, m: &HostMesh, np: usize) -> Result<PartitionMethod> {
    match name {
        "hilbert" => Ok(PartitionMethod::Hilbert),
        "linear" => Ok(PartitionMethod::Linear),
        // Cell c to part c % P: it cuts nearly every face in the mesh.
        "roundrobin" => Ok(PartitionMethod::Explicit(
            (0..m.n_cells).map(|c| (c % np) as Label).collect(),
        )),
        other => Err(Error::Config(format!(
            "unknown partition method '{other}'; known: hilbert, linear, \
             roundrobin, all"
        ))),
    }
}

/// Part `r` becomes part `(r + shift) mod P`: the same cells in the same
/// groups, owned by a different rank.
fn rotate(map: &[Label], np: usize, shift: usize) -> Vec<Label> {
    map.iter()
        .map(|&r| ((r as usize + shift) % np) as Label)
        .collect()
}

/// FNV-1a, 64 bit, over the rank count (u64, little endian) and then every
/// entry of the map (its little-endian bytes). Deterministic across runs,
/// processes and machines, which `std`'s hasher does not promise.
fn map_hash(np: usize, map: &[Label]) -> u64 {
    const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0000_0100_0000_01b3;
    let mut h = OFFSET;
    let bytes = (np as u64)
        .to_le_bytes()
        .into_iter()
        .chain(map.iter().flat_map(|l| l.to_le_bytes()));
    for b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(PRIME);
    }
    h
}

// ==========================================================================
//  The systems and fields - ofgpu-decompose's, unchanged
// ==========================================================================

/// A deterministic field of the GLOBAL cell id.
fn field(c: usize) -> Scalar {
    0.25 + 0.0625 * ((c * 37) % 23) as Scalar
}

/// A Dirichlet value per boundary cell, keyed on the GLOBAL cell id.
fn wall(c: usize) -> Scalar {
    0.5 + 0.125 * ((c * 11) % 7) as Scalar
}

/// §71.8's matrix: a Laplacian plus a `ddt`, with pinned cells chosen by
/// GLOBAL id.
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

/// §73.8's matrix: a symmetric positive definite Poisson operator plus a
/// `ddt`, folded on the host.
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

/// §72's adversarial field: a full-width mantissa over a hundred binades
/// with mixed signs, so a rounding lands in essentially every addition.
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

/// The halo payloads at a GLOBAL cell id.
fn vec_at(g: usize) -> Vec3 {
    Vec3::new(field(g), -field(g), 2.0 * field(g))
}

fn tensor_at(g: usize) -> Tensor {
    let s = field(g);
    Tensor {
        xx: s, xy: -2.0 * s, xz: 3.0 * s,
        yx: -4.0 * s, yy: 5.0 * s, yz: -6.0 * s,
        zx: 7.0 * s, zy: -8.0 * s, zz: 9.0 * s,
    }
}

fn label_at(g: usize) -> Label {
    3 * g as Label - 1
}

/// A tensor's nine components, row-major (xx..zz).
fn comps(t: &Tensor) -> [Scalar; 9] {
    [t.xx, t.xy, t.xz, t.yx, t.yy, t.yz, t.zx, t.zy, t.zz]
}

fn krylov_controls(solver: LinearSolverKind) -> SolverControls {
    SolverControls {
        solver,
        precon: Preconditioner::Diagonal,
        tolerance: TOL,
        rel_tol: 0.0,
        max_iter: 600,
        min_iter: 0,
        check_interval: 1,
        fixed_iters: false,
        report_residuals: true,
        gamg: ofgpu::io::case::GamgControls::default(),
    }
}

const KRYLOV: [(LinearSolverKind, &str); 2] = [
    (LinearSolverKind::PCG, "PCG"),
    (LinearSolverKind::PBiCGStab, "PBiCGStab"),
];

// ==========================================================================
//  §71.8's sweep pipeline on one part (ofgpu-decompose's Rig)
// ==========================================================================

/// One part's device state. `psi` lives outside, in the vector the exchange
/// takes.
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

/// `relax`, `set_values`, the fold, and `1/diag`.
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

/// `psi += (b - A psi)/diag`.
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
//  The references, on the main thread
// ==========================================================================

struct Refs {
    psi_in: Vec<Scalar>,
    /// §71.8: the undecomposed sweep pipeline's answer.
    swept: Vec<Scalar>,
    vol: Vec<Scalar>,
    hard: Vec<Scalar>,
    /// §72.8: exact sum and dot of (swept, V), then of (hard, V).
    reduced: [Scalar; 4],
    /// §73.8: the whole mesh's answer and iteration count, per KRYLOV entry.
    krylov: Vec<(Vec<Scalar>, usize)>,
}

fn references(gpu: &Gpu, m: &HostMesh, a_lap: &HostLduMatrix, a_spd: &HostLduMatrix, args: &Args) -> Result<Refs> {
    let lk = LduKernels::new(gpu)?;
    let sk = SolverKernels::new(gpu)?;
    let psi_in: Vec<Scalar> = (0..m.n_cells).map(field).collect();

    let mut serial = Rig::new(gpu, m, a_lap)?;
    let mut spsi = gpu.upload(&psi_in)?;
    prepare(gpu, &lk, &sk, &mut serial, &spsi, args.alpha)?;
    for _ in 0..args.sweeps {
        sweep(gpu, &sk, &mut serial, &mut spsi)?;
    }
    gpu.sync()?;
    let swept = gpu.download(&spsi)?;

    let vol = m.v.clone();
    let hard: Vec<Scalar> = (0..m.n_cells).map(nasty).collect();
    let mut red = ExactReduction::new(gpu, &[m.n_cells])?;
    let mut reduced = [0.0 as Scalar; 4];
    for (k, x) in [&swept, &hard].into_iter().enumerate() {
        let dx = [gpu.upload(x)?];
        let dv = [gpu.upload(&vol)?];
        red.sum(gpu, &sk, &dx)?;
        gpu.sync()?;
        reduced[2 * k] = red.value(gpu)?;
        red.dot(gpu, &sk, &dx, &dv)?;
        gpu.sync()?;
        reduced[2 * k + 1] = red.value(gpu)?;
    }

    let one = Decomposition::build(m, 1, &PartitionMethod::Linear)?;
    let mut ex = HaloExchange::new(gpu, &one)?;
    let sys = DistSystem::split(gpu, &mut ex, m, &one, a_spd)?;
    let mut krylov = Vec::new();
    for (kind, label) in KRYLOV {
        let ctrl = krylov_controls(kind);
        let mut w = DistWorkspace::new(gpu, &sys, DistReduce::Exact)?;
        let mut psi = vec![gpu.upload(&one.split_field(0, &psi_in)?)?];
        let perf = dist_solve(gpu, &sk, &mut psi, &sys, &mut ex, &mut w, &ctrl)?;
        gpu.sync()?;
        if !perf.converged {
            return Err(Error::Config(format!(
                "{label} did not converge on the whole mesh in {} iterations \
                 (tolerance {TOL:e}); the Krylov gate has no reference",
                perf.n_iterations
            )));
        }
        let got = one.gather_field(&[gpu.download(&psi[0])?])?;
        if !args.quiet {
            println!(
                "{label:<10} whole mesh: {} iterations to {TOL:e}, final residual {:.3e}",
                perf.n_iterations, perf.final_residual
            );
        }
        krylov.push((got, perf.n_iterations));
    }
    Ok(Refs { psi_in, swept, vol, hard, reduced, krylov })
}

// ==========================================================================
//  One rank
// ==========================================================================

/// What one rank found at one rotation. Counts are of cells that DIFFER.
#[derive(Default)]
struct RankRot {
    /// The hashes every rank sent, in rank order, as this rank received them.
    hashes: Vec<u64>,
    owned: usize,
    ghosts: usize,
    /// Local cells (of four payloads) whose exchanged value is not the
    /// owner's.
    halo_bad: usize,
    /// Owned cells whose swept value is not the undecomposed run's.
    sweep_bad: usize,
    sweep_worst: Scalar,
    /// Which of the four exact reductions moved.
    red_bad: usize,
    /// Per KRYLOV entry: differing owned cells, the worst |d|, iterations.
    krylov: Vec<(usize, Scalar, usize)>,
}

/// Four payloads through the exchange, ghosts poisoned beforehand: every
/// local cell must hold the value at its global id.
fn payload_gate(gpu: &Gpu, comm: &dyn Comm, halo: &mut SpmdHalo, part: &PartMesh) -> Result<usize> {
    let n = part.mesh.n_cells;
    let gid = |i: usize| part.global_cell[i] as usize;
    let nl = part.n_local();
    let mut bad = 0usize;

    let s: Vec<Scalar> = (0..nl).map(|i| if i < n { field(gid(i)) } else { 1.0e30 }).collect();
    let mut d = gpu.upload(&s)?;
    halo.scalar(gpu, comm, &mut d)?;
    let got = gpu.download(&d)?;
    bad += (0..nl).filter(|&i| got[i].to_bits() != field(gid(i)).to_bits()).count();

    let v: Vec<Vec3> = (0..nl)
        .map(|i| if i < n { vec_at(gid(i)) } else { Vec3::new(1.0e30, 1.0e30, 1.0e30) })
        .collect();
    let mut d = gpu.upload(&v)?;
    halo.vector(gpu, comm, &mut d)?;
    let got = gpu.download(&d)?;
    bad += (0..nl)
        .filter(|&i| {
            let w = vec_at(gid(i));
            [got[i].x, got[i].y, got[i].z].map(|c| c.to_bits()) != [w.x, w.y, w.z].map(|c| c.to_bits())
        })
        .count();

    let t: Vec<Tensor> = (0..nl)
        .map(|i| if i < n { tensor_at(gid(i)) } else { Tensor::ZERO })
        .collect();
    let mut d = gpu.upload(&t)?;
    halo.tensor(gpu, comm, &mut d)?;
    let got = gpu.download(&d)?;
    bad += (0..nl)
        .filter(|&i| comps(&got[i]).map(Scalar::to_bits) != comps(&tensor_at(gid(i))).map(Scalar::to_bits))
        .count();

    let l: Vec<Label> = (0..nl).map(|i| if i < n { label_at(gid(i)) } else { -7 }).collect();
    let mut d = gpu.upload(&l)?;
    halo.label(gpu, comm, &mut d)?;
    let got = gpu.download(&d)?;
    bad += (0..nl).filter(|&i| got[i] != label_at(gid(i))).count();
    Ok(bad)
}

/// Everything rank `comm.rank()` does in one world: every rotation of the
/// partitioner's map, every gate.
#[allow(clippy::too_many_arguments)]
fn rank_body(
    comm: &dyn Comm,
    placement: &Placement,
    m: &HostMesh,
    np: usize,
    method: &PartitionMethod,
    a_lap: &HostLduMatrix,
    a_spd: &HostLduMatrix,
    refs: &Refs,
    args: &Args,
) -> Result<Vec<RankRot>> {
    let rank = comm.rank();
    let gpu = Gpu::for_rank(placement, rank)?;
    let lk = LduKernels::new(&gpu)?;
    let sk = SolverKernels::new(&gpu)?;
    // The rank computes the map itself, as an MPI rank would.
    let base = partition(m, np, method)?;
    let mut out = Vec::with_capacity(np);
    for shift in 0..np {
        let map = rotate(&base, np, shift);
        let mut all = vec![0u8; 8 * np];
        comm.allgather(&map_hash(np, &map).to_le_bytes(), &mut all)?;
        let hashes = all
            .chunks_exact(8)
            .map(|c| u64::from_le_bytes(c.try_into().expect("8 bytes")))
            .collect();
        let d = Decomposition::from_map(m, np, map)?;
        let part = &d.parts[rank];
        let n = part.mesh.n_cells;
        let gid = |i: usize| part.global_cell[i] as usize;
        let mut halo = SpmdHalo::new(&gpu, part, np)?;
        let mut rr = RankRot {
            hashes,
            owned: n,
            ghosts: part.n_halo,
            ..Default::default()
        };

        // ---- HALO: the payloads, then §71.8's pipeline --------------------
        rr.halo_bad = payload_gate(&gpu, comm, &mut halo, part)?;

        let split = d.split_matrix(m, rank, a_lap)?;
        let mut rig = Rig::new(&gpu, &part.mesh, &split)?;
        let mut isf = gpu.upload(&split.is_fixed)?;
        let mut fv = gpu.upload(&split.fixed_value)?;
        halo.label(&gpu, comm, &mut isf)?;
        halo.scalar(&gpu, comm, &mut fv)?;
        rig.a.is_fixed = isf;
        rig.a.fixed_value = fv;
        let mut psi = gpu.upload(&d.split_field(rank, &refs.psi_in)?)?;
        halo.scalar(&gpu, comm, &mut psi)?;
        prepare(&gpu, &lk, &sk, &mut rig, &psi, args.alpha)?;
        for _ in 0..args.sweeps {
            halo.scalar(&gpu, comm, &mut psi)?;
            sweep(&gpu, &sk, &mut rig, &mut psi)?;
        }
        gpu.sync()?;
        let got = gpu.download(&psi)?;
        for (i, g) in got.iter().enumerate().take(n) {
            let w = refs.swept[gid(i)];
            if g.to_bits() != w.to_bits() {
                rr.sweep_bad += 1;
                rr.sweep_worst = rr.sweep_worst.max((g - w).abs());
            }
        }

        // ---- REDUCTION: the halo poisoned, so a summed ghost cannot agree --
        let mut red = SpmdReduction::new(&gpu, comm, n)?;
        let poisoned = |x: &[Scalar], p: Scalar| -> Result<DevBuf<Scalar>> {
            let mut f = d.split_field(rank, x)?;
            for h in f.iter_mut().skip(n) {
                *h = p;
            }
            gpu.upload(&f)
        };
        let dv = poisoned(&refs.vol, -1.0e30)?;
        for (k, x) in [&refs.swept, &refs.hard].into_iter().enumerate() {
            let dx = poisoned(x, 1.0e30)?;
            red.sum(&gpu, &sk, comm, &dx)?;
            gpu.sync()?;
            if red.value(&gpu)?.to_bits() != refs.reduced[2 * k].to_bits() {
                rr.red_bad += 1;
            }
            red.dot(&gpu, &sk, comm, &dx, &dv)?;
            gpu.sync()?;
            if red.value(&gpu)?.to_bits() != refs.reduced[2 * k + 1].to_bits() {
                rr.red_bad += 1;
            }
        }

        // ---- KRYLOV ---------------------------------------------------------
        let sys = DistSystem::split_one(&gpu, &mut halo, comm, m, &d, a_spd)?;
        for ((kind, _), (want, _)) in KRYLOV.into_iter().zip(&refs.krylov) {
            let ctrl = krylov_controls(kind);
            let mut w = SpmdWorkspace::new(&gpu, comm, &sys, DistReduce::Exact)?;
            w.set_flag_vote(true);
            let mut psi = gpu.upload(&d.split_field(rank, &refs.psi_in)?)?;
            let perf = spmd_solve(&gpu, &sk, comm, &mut psi, &sys, &mut halo, &mut w, &ctrl)?;
            gpu.sync()?;
            let got = gpu.download(&psi)?;
            let (mut bad, mut worst) = (0usize, 0.0 as Scalar);
            for (i, g) in got.iter().enumerate().take(n) {
                let w = want[gid(i)];
                if g.to_bits() != w.to_bits() {
                    bad += 1;
                    worst = worst.max((g - w).abs());
                }
            }
            rr.krylov.push((bad, worst, perf.n_iterations));
        }
        out.push(rr);
    }
    Ok(out)
}

// ==========================================================================
//  The driver
// ==========================================================================

/// Gate tallies over the whole run.
#[derive(Default)]
struct Tally {
    rotations: usize,
    failed: usize,
}

fn verdict(ok: bool) -> &'static str {
    if ok {
        "PASS"
    } else {
        "FAIL"
    }
}

/// SPEC-LIT 139.4 (PAR-10c): `-solve simple` - the lid-driven cavity at
/// Re = 100 (Ghia, Ghia & Shin 1982, DOI 10.1016/0021-9991(82)90058-4), run
/// serial on the whole mesh and then decomposed, one `ThreadComm` thread per
/// rank, every rank on its own `Gpu`. Each rank count prints its L_inf
/// differences against the serial answer and PASS/FAIL against the gate
/// constants of `distflow`; the exit status is 0 only when every run passes.
fn run_solve(args: &SolveArgs) -> Result<()> {
    let gpu = Gpu::new(0)?;
    let hm = cavity_mesh(args.n)?;
    if !args.quiet {
        println!("solve  : simple - the lid-driven cavity at Re = 100 (Ghia, Ghia & Shin 1982)");
        println!("mesh   : {} x {} x 1, {} cells", args.n, args.n, hm.n_cells);
        println!("device : {}", gpu.name()?);
        println!("np     : {:?}", args.nps);
        println!();
    }

    let serial = cavity_serial(&gpu, &hm)?;
    if !serial.converged {
        return Err(Error::Config(format!(
            "the serial cavity did not converge in {} iterations (U \
             residualControl {CAVITY_U_TOL:e}); the gate has no \
             reference",
            serial.iterations,
        )));
    }
    let errs_s = cavity_ghia_errors(&hm, &serial.u);
    let worst_s = errs_s.iter().fold(0.0 as Scalar, |a, &x| a.max(x));
    let u_scale = serial.u.iter().fold(0.0 as Scalar, |a, v| {
        a.max(v.x.abs()).max(v.y.abs()).max(v.z.abs())
    });
    let p_scale = serial.p.iter().fold(0.0 as Scalar, |a, &v| a.max(v.abs()));
    if !args.quiet {
        println!(
            "serial : {} iterations, worst U residual {:.3e}, continuity \
             {:.3e}, worst centreline |e| {worst_s:.4}",
            serial.iterations, serial.u_residual, serial.continuity
        );
        println!();
    }

    let mut all_ok = true;
    let t_all = Instant::now();
    for &np in &args.nps {
        for name in &args.methods {
            // The name is validated on every run; P = 1 is the Linear cut,
            // which IS the one-part cut - so that is the name printed too.
            let named = method_named(name, &hm, np)?;
            let method = if np == 1 { PartitionMethod::Linear } else { named };
            let name = if np == 1 { "linear" } else { name.as_str() };
            let t0 = Instant::now();
            let dec = Decomposition::build(&hm, np, &method)?;
            let got = ThreadComm::run(np, |comm| -> Result<CavityRun> {
                let gpu = Gpu::new(0)?;
                cavity_part(&gpu, comm, &dec)
            })?;
            let mut up = vec![Vec3::ZERO; hm.n_cells];
            let mut pp = vec![0.0 as Scalar; hm.n_cells];
            let mut iters = Vec::new();
            let mut conv_ok = true;
            // A rank that failed fails the run by its own message: its owned
            // cells would otherwise stay zero in the scatter below.
            let got = got.into_iter().collect::<Result<Vec<CavityRun>>>()?;
            for r in got.iter() {
                for (i, &g) in r.global_cell.iter().enumerate() {
                    up[g] = r.u[i];
                    pp[g] = r.p[i];
                }
                iters.push(r.iterations);
                conv_ok &= r.converged;
            }
            let u_rel = up
                .iter()
                .zip(&serial.u)
                .fold(0.0 as Scalar, |a, (x, w)| {
                    a.max((x.x - w.x).abs())
                        .max((x.y - w.y).abs())
                        .max((x.z - w.z).abs())
                })
                / u_scale;
            let p_rel = pp
                .iter()
                .zip(&serial.p)
                .fold(0.0 as Scalar, |a, (&x, &w)| a.max((x - w).abs()))
                / p_scale;
            let errs = cavity_ghia_errors(&hm, &up);
            let worst_e = errs.iter().fold(0.0 as Scalar, |a, &x| a.max(x));
            // The centreline rule, at every Ghia station: the decomposed
            // error within the serial run's own error plus the tolerance.
            let centre_ok = errs
                .iter()
                .zip(&errs_s)
                .all(|(&ep, &es)| ep <= es + CAVITY_TOL_CENTRE);
            let it_s = serial.iterations as f64;
            let it_ok = iters
                .iter()
                .all(|&it| (it as f64 - it_s).abs() <= CAVITY_ITER_BAND * it_s);
            let ok = u_rel <= CAVITY_TOL_FIELD
                && p_rel <= CAVITY_TOL_FIELD
                && centre_ok
                && it_ok
                && conv_ok;
            all_ok &= ok;
            println!(
                "P = {np:<2} {name:<10} {} - iters {:?} (serial {}), U rel \
                 {u_rel:.3e} <= {CAVITY_TOL_FIELD:e}, p rel {p_rel:.3e}, \
                 worst centreline |e| {worst_e:.4} vs serial {worst_s:.4}, \
                 {:.1} s",
                verdict(ok),
                iters,
                serial.iterations,
                t0.elapsed().as_secs_f64()
            );
        }
    }

    let secs = t_all.elapsed().as_secs_f64();
    if all_ok {
        println!(
            "ofgpu-par -solve simple: PASS - every rank count reproduced the \
             serial cavity within the gate constants, {secs:.1} s"
        );
        Ok(())
    } else {
        println!("ofgpu-par -solve simple: FAIL - {secs:.1} s");
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    if argv.is_empty() || argv[0] == "-h" || argv[0] == "--help" {
        println!("{USAGE}");
        std::process::exit(if argv.is_empty() { 2 } else { 0 });
    }
    // SPEC-LIT 139.4's solve mode: `ofgpu-par -solve simple [options]`. The
    // value is refused by name here, where the mode is chosen, so a typo
    // never reads a case.
    if argv[0] == "-solve" {
        match argv.get(1).map(String::as_str) {
            Some("simple") => {
                let args = parse_solve(&argv[2..])?;
                return run_solve(&args);
            }
            other => {
                return Err(Error::Config(format!(
                    "-solve '{}': known: simple",
                    other.unwrap_or("")
                )));
            }
        }
    }
    let args = parse(&argv)?;
    refuse_unbuilt_transport(&args.comm)?;
    let (m, _cc, lowered) = common::load_case(&args.case)?;
    common::refuse_motion_block(lowered.as_ref(), "ofgpu-par")?;
    for &np in &args.nps {
        if np > m.n_cells {
            return Err(Error::Config(format!(
                "-np {np}: the mesh has only {} cells",
                m.n_cells
            )));
        }
    }
    for name in &args.methods {
        method_named(name, &m, 1)?;
    }

    let placement = Placement::from_env(Gpu::device_count()?)?;
    let gpu = Gpu::for_rank(&placement, 0)?;
    if !args.quiet {
        println!("case   : {}", args.case.display());
        println!("device : {} (placement {:?})", gpu.name()?, placement.devices());
        println!(
            "mesh   : {} cells, {} internal faces, {} boundary faces",
            m.n_cells, m.n_internal_faces, m.n_boundary_faces
        );
        println!("comm   : threads   ranks {:?}   sweeps {}", args.nps, args.sweeps);
        println!();
    }

    let t_all = Instant::now();
    let a_lap = laplacian(&m);
    let a_spd = poisson(&m);
    let refs = references(&gpu, &m, &a_lap, &a_spd, &args)?;
    if !args.quiet {
        println!();
    }

    let mut tally = Tally::default();
    let mut worlds = 0usize;
    for &np in &args.nps {
        for name in &args.methods {
            let method = method_named(name, &m, np)?;
            let base = partition(&m, np, &method)?;
            let t0 = Instant::now();
            let ranks = ThreadComm::run(np, |comm| {
                rank_body(comm, &placement, &m, np, &method, &a_lap, &a_spd, &refs, &args)
            })?;
            let ranks: Vec<Vec<RankRot>> = ranks
                .into_iter()
                .enumerate()
                .map(|(r, x)| x.map_err(|e| Error::Config(format!("P = {np} {name} rank {r}: {e}"))))
                .collect::<Result<_>>()?;
            worlds += 1;
            let at = |shift: usize| format!("P = {np}  {name:<10} rot {shift}");
            for shift in 0..np {
                let rot: Vec<&RankRot> = ranks.iter().map(|r| &r[shift]).collect();
                let want = map_hash(np, &rotate(&base, np, shift));
                let map_ok = rot.iter().all(|r| r.hashes.len() == np && r.hashes.iter().all(|&h| h == want));
                let owned: usize = rot.iter().map(|r| r.owned).sum();
                let ghosts: usize = rot.iter().map(|r| r.ghosts).sum();
                let halo_bad: usize = rot.iter().map(|r| r.halo_bad).sum();
                let sweep_bad: usize = rot.iter().map(|r| r.sweep_bad).sum();
                let sweep_worst = rot.iter().map(|r| r.sweep_worst).fold(0.0 as Scalar, Scalar::max);
                let red_bad: usize = rot.iter().map(|r| r.red_bad).sum();
                let halo_ok = halo_bad == 0 && sweep_bad == 0;
                println!(
                    "{}  MAP       {} - {np} rank hashes {want:016x}",
                    at(shift),
                    verdict(map_ok)
                );
                println!(
                    "{}  HALO      {} - 4 payloads on {owned} owned + {ghosts} ghost \
                     cells, {halo_bad} differ; {} sweeps: {sweep_bad} of {owned} \
                     cells differ, worst |d| {sweep_worst:e}",
                    at(shift),
                    verdict(halo_ok),
                    args.sweeps
                );
                println!(
                    "{}  REDUCTION {} - exact sum and dot of 2 fields: {red_bad} of {} \
                     ranks x 4 values moved",
                    at(shift),
                    verdict(red_bad == 0),
                    np
                );
                let mut kry_ok = true;
                for (k, (_, label)) in KRYLOV.iter().enumerate() {
                    let want_iters = refs.krylov[k].1;
                    let bad: usize = rot.iter().map(|r| r.krylov[k].0).sum();
                    let worst = rot.iter().map(|r| r.krylov[k].1).fold(0.0 as Scalar, Scalar::max);
                    let iters: Vec<usize> = rot.iter().map(|r| r.krylov[k].2).collect();
                    let ok = bad == 0 && iters.iter().all(|&i| i == want_iters);
                    kry_ok &= ok;
                    println!(
                        "{}  {label:<9} {} - {bad} of {owned} cells differ, worst \
                         |d| {worst:e}; iterations {:?} (whole mesh {want_iters})",
                        at(shift),
                        verdict(ok),
                        iters
                    );
                }
                tally.rotations += 1;
                if !(map_ok && halo_ok && red_bad == 0 && kry_ok) {
                    tally.failed += 1;
                }
            }
            if !args.quiet {
                println!(
                    "P = {np}  {name:<10} {np} rotation(s) in {:.1} s",
                    t0.elapsed().as_secs_f64()
                );
                println!();
            }
        }
    }

    let secs = t_all.elapsed().as_secs_f64();
    if tally.failed == 0 {
        println!(
            "ofgpu-par: PASS - {worlds} worlds, {} rotations, every gate bit for \
             bit (worst == 0) with the whole mesh's iteration counts, {secs:.1} s",
            tally.rotations
        );
        Ok(())
    } else {
        println!(
            "ofgpu-par: FAIL - {} of {} rotations failed a gate, {secs:.1} s",
            tally.failed, tally.rotations
        );
        std::process::exit(1);
    }
}

fn main() {
    if let Err(e) = run() {
        eprintln!("ofgpu-par: {e}");
        std::process::exit(1);
    }
}
