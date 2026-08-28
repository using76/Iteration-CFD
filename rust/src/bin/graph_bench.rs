// meteor-cfd - Copyright (c) 2026 주식회사 메테오시뮬레이션 (Meteo Simulation Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! ofgpu-graph-bench — what does CUDA Graph actually buy?
//!
//! One outer iteration of a turbulence model is ~50 kernel launches. Each
//! launch costs the CPU a few microseconds of driver work, and on a small mesh
//! that overhead is a real fraction of the iteration. A CUDA graph records the
//! whole launch sequence once and replays it with a single call, so the driver
//! cost collapses to one submission.
//!
//! The catch is that a graph cannot capture a decision the host makes. The
//! adaptive linear solver reads a convergence flag back every few sweeps, so it
//! is not capturable; `fixed_iters` exists precisely to remove that read-back.
//! This benchmark therefore measures three things that are NOT the same run:
//!
//!   1. adaptive        - the normal mode, solver stops when converged
//!   2. fixed, no graph - identical work to (3), launched one kernel at a time
//!   3. fixed + graph   - identical work to (2), replayed from a graph
//!
//! (2) vs (3) is the honest measure of what the graph saves. (1) is there so
//! the cost of going fixed-iteration is visible too, because that trade is
//! part of the decision.
//!
//! Provenance: ORIGINAL - a benchmark harness for CUDA-graph capture against
//! per-launch execution. It measures this crate's own kernels; there is no
//! external source for it (`PROVENANCE.md`, `src/bin/*`). No GPL-licensed
//! source was consulted.

use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use ofgpu::field::{GpuSurfaceScalarField, GpuVectorField};
use ofgpu::field_ops::{correct_boundary_conditions_vector, FieldKernels};
use ofgpu::field_setup::{
    compute_phi_from_u, max_div_phi, setup_scalar_field, setup_vector_field,
    wall_coeffs_from_case, NutRoughness, WallFaces,
};
use ofgpu::io::case::{find_start_time, model_coeff, read_case_controls};
use ofgpu::io::fields::{read_scalar_field, read_vector_field};
use ofgpu::io::polymesh::{build_host_mesh, read_poly_mesh};
use ofgpu::models::k_epsilon::{KEpsilon, KEpsilonCoeffs};
use ofgpu::turbulence::FlowState;
use ofgpu::{GpuMesh, Gpu, Result};

#[path = "common/mod.rs"]
mod common;
use common::{atoi, device_banner, g, next_arg};

struct Opts {
    case_dir: PathBuf,
    iters: usize,
    sweeps: i32,
}

fn parse() -> Result<Opts> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        return Err(ofgpu::Error::Config(
            concat!(
                "usage: ofgpu-graph-bench <caseDir> [-iters N] [-sweeps N]\n",
                "   -iters  N   outer iterations to time (default 200)\n",
                "   -sweeps N   linear-solver sweeps per equation in the fixed\n",
                "               modes (default 3; pick the count the adaptive\n",
                "               solver actually uses, or the comparison is unfair)"
            )
            .to_string(),
        ));
    }

    let mut o = Opts {
        case_dir: PathBuf::from(&args[1]),
        iters: 200,
        sweeps: 3,
    };

    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "-iters" => o.iters = atoi(&next_arg(&args, &mut i)?) as usize,
            "-sweeps" => o.sweeps = atoi(&next_arg(&args, &mut i)?) as i32,
            other => {
                return Err(ofgpu::Error::Config(format!("unknown option {other}")))
            }
        }
        i += 1;
    }
    Ok(o)
}

fn run(o: &Opts) -> Result<()> {
    let gpu = Gpu::new(0)?;
    println!("{}", device_banner(&gpu, "graph benchmark")?);

    // ---- mesh and case ----------------------------------------------------
    let hm = build_host_mesh(&read_poly_mesh(&o.case_dir)?)?;
    let mesh = GpuMesh::upload(&gpu, &hm)?;
    gpu.sync()?;

    println!(
        "\n{} cells, {} internal faces, {} boundary faces",
        hm.n_cells, hm.n_internal_faces, hm.n_boundary_faces
    );

    let cc = read_case_controls(&o.case_dir)?;
    let d = KEpsilonCoeffs::default();
    let coeffs = KEpsilonCoeffs {
        cmu: model_coeff(&cc, "Cmu", d.cmu),
        c1: model_coeff(&cc, "C1", d.c1),
        c2: model_coeff(&cc, "C2", d.c2),
        c3: model_coeff(&cc, "C3", d.c3),
        sigmak: model_coeff(&cc, "sigmak", d.sigmak),
        sigma_eps: model_coeff(&cc, "sigmaEps", d.sigma_eps),
    };

    // ---- fields -----------------------------------------------------------
    let t_dir = o.case_dir.join(find_start_time(&o.case_dir)?);
    let raw_u = read_vector_field(&t_dir.join("U"), hm.n_cells)?;
    let raw_k = read_scalar_field(&t_dir.join("k"), hm.n_cells)?;
    let raw_e = read_scalar_field(&t_dir.join("epsilon"), hm.n_cells)?;

    let fk = FieldKernels::new(&gpu)?;
    let mut u = GpuVectorField::zeros(&gpu, &mesh, "U")?;
    setup_vector_field(&gpu, &mut u, &raw_u, &hm)?;
    correct_boundary_conditions_vector(&gpu, &fk, &mut u, &mesh)?;

    let mut phi = GpuSurfaceScalarField::zeros(&gpu, &mesh, "phi")?;
    // The benchmark always reconstructs phi rather than reading 0/phi: the
    // point is to time the same work every run, not to be faithful to a
    // particular flux file.
    compute_phi_from_u(&gpu, &mut phi, &u, &hm)?;
    println!(
        "max |sum_f phi| per cell = {}",
        g(f64::from(max_div_phi(&gpu, &phi, &hm)?))
    );

    // SPEC-LIT 15.5: nut's own patch types decide nu_t's wall treatment.
    let nut_path = t_dir.join("nut");
    let raw_nut = if nut_path.exists() {
        Some(read_scalar_field(&nut_path, hm.n_cells)?)
    } else {
        None
    };
    let wf = WallFaces::from_case(&raw_e, raw_nut.as_ref(), &hm)?;
    let roughness = NutRoughness::from_case(raw_nut.as_ref(), &hm)?;
    let flow = FlowState::new(&u, &phi, cc.nu);

    // A fresh model per mode, so no mode inherits another's converged state.
    let build = |gpu: &Gpu, fixed: bool| -> Result<KEpsilon<'_>> {
        let mut turb = cc.turb;
        if fixed {
            turb.k_solver.fixed_iters = true;
            turb.k_solver.max_iter = o.sweeps;
            turb.k_solver.report_residuals = false;
            turb.epsilon_solver.fixed_iters = true;
            turb.epsilon_solver.max_iter = o.sweeps;
            turb.epsilon_solver.report_residuals = false;
        }
        let mut m = KEpsilon::new(
            gpu,
            &hm,
            &mesh,
            coeffs,
            turb,
            wall_coeffs_from_case(&cc.wall),
            &wf,
            &roughness,
        )?;
        setup_scalar_field(gpu, m.k_mut(), &raw_k, &hm)?;
        setup_scalar_field(gpu, m.epsilon_mut(), &raw_e, &hm)?;
        m.initialise(gpu, &flow)?;
        Ok(m)
    };

    let n = o.iters;
    let cells = hm.n_cells as f64;
    let row = |name: &str, secs: f64| {
        let ms = secs / n as f64 * 1e3;
        println!(
            "  {:<22} {:>8.3} ms/iter   {:>7.1} Mcell-iter/s",
            name,
            ms,
            cells * n as f64 / secs / 1e6
        );
    };

    println!("\ntiming {n} outer iterations, {} solver sweeps in the fixed modes\n", o.sweeps);

    // ---- 1. adaptive ------------------------------------------------------
    {
        let mut model = build(&gpu, false)?;
        for _ in 0..5 {
            model.correct(&gpu, &flow)?;
        }
        gpu.sync()?;
        let t = Instant::now();
        for _ in 0..n {
            model.correct(&gpu, &flow)?;
        }
        gpu.sync()?;
        row("adaptive", t.elapsed().as_secs_f64());
    }

    // ---- 2. fixed sweeps, launched one kernel at a time -------------------
    // Keep the final k so mode 3 can be checked against it. Both modes run
    // 5 warm-up plus n timed iterations from the same initial fields, and the
    // capture in mode 3 executes nothing, so the two histories are identical
    // and the answers must be too.
    let (per_launch, k_ref) = {
        let mut model = build(&gpu, true)?;
        for _ in 0..5 {
            model.correct(&gpu, &flow)?;
        }
        gpu.sync()?;
        let t = Instant::now();
        for _ in 0..n {
            model.correct(&gpu, &flow)?;
        }
        gpu.sync()?;
        let s = t.elapsed().as_secs_f64();
        row("fixed, per-launch", s);
        (s, gpu.download(&model.k().f)?)
    };

    // ---- capture smoke test -----------------------------------------------
    // Before blaming the model, establish that capture works at all on this
    // stream. If a single elementwise launch cannot be captured, the problem
    // is in the plumbing, not in `correct()`.
    {
        let a = gpu.zeros::<ofgpu::Scalar>(1024)?;
        let mut b = gpu.zeros::<ofgpu::Scalar>(1024)?;
        let probe = gpu.capture(|_| {
            ofgpu::field_ops::copy_field(&gpu, &fk, &mut b, &a, 1024)?;
            Ok(())
        });
        match probe {
            Ok(Some(_)) => println!("\n  [capture smoke test] one kernel: ok"),
            Ok(None) => println!("\n  [capture smoke test] one kernel: empty graph"),
            Err(e) => println!("\n  [capture smoke test] one kernel FAILED: {e}"),
        }
    }

    // ---- 3. the same work, replayed from a graph --------------------------
    {
        let mut model = build(&gpu, true)?;

        // Warm up first: the very first call of each kernel pays for module
        // loading, and capturing that would bake nothing useful into the graph.
        for _ in 0..5 {
            model.correct(&gpu, &flow)?;
        }
        gpu.sync()?;

        let t_cap = Instant::now();
        let graph = gpu.capture(|_| {
            model.correct(&gpu, &flow)?;
            Ok(())
        })?;
        let cap_secs = t_cap.elapsed().as_secs_f64();

        let Some(mut graph) = graph else {
            println!("  capture produced an empty graph - nothing to replay");
            return Ok(());
        };
        graph.upload()?;
        gpu.sync()?;

        // The capture itself executed nothing, so replay n times for the same
        // total work as the other two modes.
        let t = Instant::now();
        for _ in 0..n {
            graph.launch()?;
        }
        gpu.sync()?;
        let s = t.elapsed().as_secs_f64();
        row("fixed, CUDA graph", s);

        println!(
            "\n  graph capture + instantiate: {} s (once)",
            g(cap_secs)
        );
        println!(
            "  graph replay is {:.2}x the per-launch path  ({:+.1} %)",
            per_launch / s,
            (per_launch / s - 1.0) * 100.0
        );

        // A fast wrong answer is worth nothing. Both modes ran 5 warm-up plus
        // n timed iterations from the same initial fields, and the capture
        // itself executes nothing, so the two histories are identical and the
        // results have to be bit-for-bit equal.
        let k_graph = gpu.download(&model.k().f)?;
        let differing = k_ref.iter().zip(&k_graph).filter(|(a, b)| a != b).count();
        let worst = k_ref
            .iter()
            .zip(&k_graph)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0 as ofgpu::Scalar, ofgpu::Scalar::max);

        println!(
            "  k vs per-launch: {differing} of {} cells differ, max |diff| {}",
            k_ref.len(),
            g(f64::from(worst))
        );

        if differing != 0 {
            return Err(ofgpu::Error::Config(
                "CUDA graph replay did not reproduce the per-launch result".into(),
            ));
        }
    }

    Ok(())
}

fn main() -> ExitCode {
    let o = match parse() {
        Ok(o) => o,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::FAILURE;
        }
    };
    match run(&o) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("\nerror: {e}");
            ExitCode::FAILURE
        }
    }
}
