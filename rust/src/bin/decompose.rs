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
//! bit**. The matrix is assembled on the host and distributed by
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
use ofgpu::halo::HaloExchange;
use ofgpu::ldu::{GpuLduMatrix, HostLduMatrix};
use ofgpu::ldu_ops::{self, LduKernels};
use ofgpu::mesh::GpuMesh;
use ofgpu::solver::{self, SolverKernels};
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

    let mut runs = 0usize;
    let mut failed = 0usize;

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
            if !args.quiet {
                println!();
            }
        }
    }

    println!("{}/{} decompositions bitwise identical to the whole mesh", runs - failed, runs);
    if failed > 0 {
        std::process::exit(1);
    }
    Ok(())
}
