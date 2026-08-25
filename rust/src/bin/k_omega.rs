// meteor-cfd - Copyright (c) 2026 주식회사 메테오시뮬레이션 (Meteo Simulation Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

//! `ofgpu-k-omega` - run the GPU-native k-omega model on a case directory.
//!
//! ```text
//! ofgpu-k-omega <caseDir> [options]
//!
//!   -iters N        outer iterations (default: from controlDict endTime)
//!   -fixedIters N   run the linear solver for exactly N sweeps and never
//!                   read a residual, so the time loop performs ZERO host
//!                   transfers of any kind
//!   -write NAME     time directory to write into (default: from controlDict)
//!   -noWrite        do not write fields; useful when timing
//!   -check N        test the convergence measure every N iterations
//!   -permissive     downgrade an unsupported setting from an error to a
//!                   warning that says what was substituted (SPEC-LIT 13.4)
//! ```
//!
//! There is no `-blended`. ofgpu always blends the two branches of the law of
//! the wall - SPEC-LIT 6.4 marks the blending *DESIGN* and this is the choice
//! we made; see the note beside `WallFunctionCoeffs` in `io::case`. The flag
//! used to exist, was printed in the banner, and was read by nothing: the
//! kernels blended regardless, so `blended no` - the default - silently got
//! the blended form and the banner printed a claim that was false.
//!
//! The case decides which model runs, not this binary:
//! `constant/momentumTransport`'s `RAS { model ...; }` is dispatched through
//! `ofgpu::models::select_turbulence_model`.
//!
//! The velocity field is held frozen: this program solves the two turbulence
//! transport equations on a given `U` and `phi`. After the mesh and the
//! initial fields are uploaded, no field data crosses the PCIe bus again
//! until the results are written.
//!
//! Provenance: the driver was carried across from this project's own earlier
//! C++ driver, deliberately kept line for line
//! alongside `k_epsilon.rs`: the two drivers differ only in which dissipation
//! variable they read, name and write, and holding everything else still is
//! what makes that visible.

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use ofgpu::field::{GpuSurfaceScalarField, GpuVectorField};
use ofgpu::field_ops::{correct_boundary_conditions_vector, FieldKernels};
use ofgpu::field_setup::{
    compute_phi_from_u, harvest_scalar_field, harvest_surface_scalar_field, max_div_phi,
    setup_scalar_field,
    setup_scalar_field_with, setup_vector_field, update_inlet_outlet, wall_coeffs_from_case,
    BcInputs, NutRoughness, WallFaces,
};
use ofgpu::io::case::{find_start_time, model_coeff, read_case_controls};
use ofgpu::io::fields::{read_scalar_field, read_vector_field, RawScalarField};
use ofgpu::io::polymesh::{build_host_mesh, read_poly_mesh};
use ofgpu::models::{select_turbulence_model, KOmega, KOmegaCoeffs, RasModel};
use ofgpu::turbulence::FlowState;
use ofgpu::{Error, GpuMesh, Label, Result, Scalar};

#[path = "common/mod.rs"]
mod common;

use common::{atoi, build_writers, device_banner, g, next_arg, parse_output_formats, sci, OutputFormat};

/// Everything the command line can change.
struct Options {
    case_dir: PathBuf,
    n_iters: Label,
    fixed_iters: Label,
    check_every: Label,
    do_write: bool,
    write_time: String,
    /// SPEC-LIT 13.4's one escape hatch: unsupported settings become warnings
    /// that say what was substituted.
    permissive: bool,
    /// `-output foam|vtu|nvdb|vdb|usda`, comma list - see `common::build_writers`.
    output: Vec<OutputFormat>,
}

fn usage() {
    eprintln!(
        "usage: ofgpu-k-omega <caseDir> [-iters N] [-fixedIters N] \
         [-write NAME] [-noWrite] [-check N] [-permissive] [-output LIST]"
    );
}

fn parse(args: &[String]) -> Result<Options> {
    if args.len() < 2 {
        usage();
        return Err(Error::Config("no case directory given".to_string()));
    }

    let mut o = Options {
        case_dir: PathBuf::from(&args[1]),
        n_iters: -1,
        fixed_iters: -1,
        check_every: 25,
        do_write: true,
        write_time: String::new(),
        permissive: false,
        output: vec![OutputFormat::Foam],
    };

    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "-iters" => o.n_iters = atoi(&next_arg(args, &mut i)?) as Label,
            "-fixedIters" => o.fixed_iters = atoi(&next_arg(args, &mut i)?) as Label,
            "-check" => o.check_every = atoi(&next_arg(args, &mut i)?) as Label,
            "-write" => o.write_time = next_arg(args, &mut i)?,
            "-noWrite" => o.do_write = false,
            "-permissive" => o.permissive = true,
            "-output" => o.output = parse_output_formats(&next_arg(args, &mut i)?)?,
            other => {
                usage();
                return Err(Error::Config(format!("unknown option {other}")));
            }
        }
        i += 1;
    }

    ofgpu::io::contract::set_permissive(o.permissive);

    Ok(o)
}

/// Load `phi` from the time directory when it is there, and reconstruct it
/// from `U` when it is not.
///
/// A `phi` written by a real solver satisfies the discrete continuity
/// equation; nothing reconstructed from a cell-centred `U` can be relied on
/// to, so the file always wins.
fn load_phi(
    gpu: &ofgpu::Gpu,
    phi: &mut GpuSurfaceScalarField,
    u: &GpuVectorField,
    hm: &ofgpu::HostMesh,
    t_dir: &Path,
) -> Result<()> {
    let path = t_dir.join("phi");

    if !path.exists() {
        compute_phi_from_u(gpu, phi, u, hm)?;
        println!("phi reconstructed as interpolate(U) & Sf");
        return Ok(());
    }

    let raw = read_scalar_field(&path, hm.n_internal_faces)?;
    gpu.write(&mut phi.f, &raw.internal)?;

    let mut bphi = vec![0.0 as Scalar; hm.n_boundary_faces];
    for p in &hm.patches {
        let Some(spec) = raw.spec(&p.name)? else {
            continue;
        };
        for i in 0..p.size {
            let v = &spec.value;
            bphi[p.start + i] = if v.is_empty() {
                0.0
            } else if v.len() == 1 {
                v[0]
            } else {
                v.get(i).copied().unwrap_or(0.0)
            };
        }
    }
    gpu.write(&mut phi.bf, &bphi)?;

    println!("phi read from {}", path.display());
    Ok(())
}

fn run(o: &Options) -> Result<()> {
    // ---- device -----------------------------------------------------------
    let gpu = ofgpu::Gpu::new(0)?;
    println!("{}", device_banner(&gpu, "k-omega")?);

    // ---- mesh -------------------------------------------------------------
    let t0 = Instant::now();

    let raw_mesh = read_poly_mesh(&o.case_dir)?;
    let hm = build_host_mesh(&raw_mesh)?;
    hm.print_report();

    let mesh = GpuMesh::upload(&gpu, &hm)?;
    gpu.sync()?;

    println!("mesh uploaded in {} s", g(t0.elapsed().as_secs_f64()));

    // ---- controls ---------------------------------------------------------
    let mut cc = read_case_controls(&o.case_dir)?;

    if o.n_iters > 0 {
        cc.turb.n_outer_iterations = o.n_iters;
    }
    cc.turb.convergence_check_every = o.check_every;

    if o.fixed_iters > 0 {
        // -fixedIters is the genuinely transfer-free mode, so the residual
        // read-back goes too: with both off, nothing at all crosses the bus
        // between upload and write.
        cc.turb.k_solver.fixed_iters = true;
        cc.turb.k_solver.max_iter = o.fixed_iters;
        cc.turb.k_solver.report_residuals = false;
        cc.turb.epsilon_solver.fixed_iters = true;
        cc.turb.epsilon_solver.max_iter = o.fixed_iters;
        cc.turb.epsilon_solver.report_residuals = false;
    }

    ofgpu::io::case::print_effective_settings(&cc);

    let selection = select_turbulence_model(&cc)?;
    match selection.model {
        RasModel::KOmega | RasModel::Laminar => {}
        other => {
            return Err(Error::Config(format!(
                "this case asks for the {} model; ofgpu-k-omega builds only kOmega",
                other.name()
            )));
        }
    }

    if !selection.active {
        println!(
            "turbulence is off in this case: nu_t is frozen at zero and the \
             model will not be corrected"
        );
    }

    let d = KOmegaCoeffs::default();
    let coeffs = KOmegaCoeffs {
        beta_star: model_coeff(&cc, "betaStar", d.beta_star),
        beta: model_coeff(&cc, "beta", d.beta),
        gamma: model_coeff(&cc, "gamma", d.gamma),
        alpha_k: model_coeff(&cc, "alphaK", d.alpha_k),
        alpha_omega: model_coeff(&cc, "alphaOmega", d.alpha_omega),
    };

    // The wall-function line says `blended` because the wall functions are
    // blended - always, unconditionally, by construction. It is not reporting
    // a switch, because there is no switch to report.
    println!(
        "nu = {} | betaStar {} beta {} gamma {} alphaK {} alphaOmega {} | \
         omegaWallFunction blended, Cmu {} kappa {} E {}",
        g(f64::from(cc.nu)),
        g(f64::from(coeffs.beta_star)),
        g(f64::from(coeffs.beta)),
        g(f64::from(coeffs.gamma)),
        g(f64::from(coeffs.alpha_k)),
        g(f64::from(coeffs.alpha_omega)),
        g(f64::from(cc.wall.cmu)),
        g(f64::from(cc.wall.kappa)),
        g(f64::from(cc.wall.e))
    );

    // ---- fields -----------------------------------------------------------
    let t = find_start_time(&o.case_dir)?;
    let t_dir = o.case_dir.join(&t);

    let raw_u = read_vector_field(&t_dir.join("U"), hm.n_cells)?;
    let raw_k = read_scalar_field(&t_dir.join("k"), hm.n_cells)?;
    let raw_w = read_scalar_field(&t_dir.join("omega"), hm.n_cells)?;

    let fk = FieldKernels::new(&gpu)?;

    let mut u = GpuVectorField::zeros(&gpu, &mesh, "U")?;
    setup_vector_field(&gpu, &mut u, &raw_u, &hm)?;
    correct_boundary_conditions_vector(&gpu, &fk, &mut u, &mesh)?;

    let mut phi = GpuSurfaceScalarField::zeros(&gpu, &mesh, "phi")?;
    load_phi(&gpu, &mut phi, &u, &hm, &t_dir)?;

    println!(
        "max |sum_f phi| per cell = {}   (0 means the flux is discretely conservative)",
        g(f64::from(max_div_phi(&gpu, &phi, &hm)?))
    );

    // SPEC-LIT 15.5: nut's wall treatment comes from nut's own patch types
    // and omega's wall-cell constraint from omega's, never one from the other.
    let nut_path = t_dir.join("nut");
    let mut raw_nut = if nut_path.exists() {
        Some(read_scalar_field(&nut_path, hm.n_cells)?)
    } else {
        None
    };
    let mut raw_k = raw_k;
    let mut raw_w = raw_w;
    // SPEC-LIT 29.1: the four per-field wall types must form one row.
    ofgpu::field_setup::validate_wall_rows(
        &hm.patches,
        raw_nut.as_mut(),
        Some(&mut raw_k),
        None,
        Some(&mut raw_w),
    )?;
    let wf_faces = WallFaces::from_case(&raw_w, raw_nut.as_ref(), &hm)?;
    let roughness = NutRoughness::from_case(raw_nut.as_ref(), &hm)?;

    let flow = FlowState::new(&u, &phi, cc.nu);

    let mut model = KOmega::new(
        &gpu,
        &hm,
        &mesh,
        coeffs,
        cc.turb,
        wall_coeffs_from_case(&cc.wall),
        &wf_faces,
        &roughness,
    )?;

    // k needs U (3/2 (I |U|)^2) and omega needs k (k^{1/2}/(C_mu^{1/4} L)),
    // hence the order and the read-back between the two.
    let u_b = gpu.download(&u.bf)?;
    setup_scalar_field_with(
        &gpu,
        model.k_mut(),
        &raw_k,
        &hm,
        BcInputs {
            u_b: Some(&u_b),
            cmu: Some(cc.wall.cmu),
            ..Default::default()
        },
    )?;

    let k_b = gpu.download(&model.k().bf)?;
    setup_scalar_field_with(
        &gpu,
        model.omega_mut(),
        &raw_w,
        &hm,
        BcInputs {
            u_b: Some(&u_b),
            k_b: Some(&k_b),
            cmu: Some(cc.wall.cmu),
            ..Default::default()
        },
    )?;

    if let Some(raw_nut) = &raw_nut {
        setup_scalar_field(&gpu, model.nut_mut(), raw_nut, &hm)?;
    }

    update_inlet_outlet(&gpu, model.k_mut(), &phi, &hm)?;
    update_inlet_outlet(&gpu, model.omega_mut(), &phi, &hm)?;

    if selection.active {
        model.initialise(&gpu, &flow)?;
    } else {
        model.freeze_nut(&gpu)?;
    }

    // ---- time loop - device only from here to the write -------------------
    println!(
        "\niterating {} times, relax k {} omega {}{}",
        cc.turb.n_outer_iterations,
        g(f64::from(cc.turb.k_relax)),
        g(f64::from(cc.turb.eps_relax)),
        if cc.turb.k_solver.fixed_iters {
            " | fixed-iteration solver: zero host transfers"
        } else {
            ""
        }
    );

    gpu.sync()?;
    let t_loop = Instant::now();

    let mut done: Label = 0;

    for it in 1..=cc.turb.n_outer_iterations {
        if !selection.active {
            // `turbulence off` / `simulationType laminar`: the closure does
            // not run, and nu_t stays at the zero `freeze_nut` put there.
            done = 1;
            break;
        }
        let (omega_perf, k_perf) = model.correct(&gpu, &flow)?;
        done = it;

        if it % cc.turb.convergence_check_every == 0 || it == 1 {
            let change = model.convergence_measure(&gpu)?;

            println!(
                "{it:>7}  omega res {} ({})  k res {} ({})  max dk/k {}",
                sci(f64::from(omega_perf.initial_residual), 3),
                omega_perf.n_iterations,
                sci(f64::from(k_perf.initial_residual), 3),
                k_perf.n_iterations,
                sci(f64::from(change), 3)
            );

            // SPEC-LIT 13.4 / the case's own SIMPLE/residualControl, on the
            // INITIAL residuals - the residual of the system as it stood
            // before this iteration's linear solve, which is what measures the
            // OUTER iteration. When the case gives no residualControl the run
            // falls back to the max-relative-change measure it always used,
            // and says which of the two stopped it.
            if !cc.residual_control.is_empty() {
                if it > 1
                    && cc.residual_control.all_satisfied(&[
                        ("omega", omega_perf.initial_residual),
                        ("k", k_perf.initial_residual),
                    ])
                {
                    println!("converged: every residualControl entry met");
                    break;
                }
            } else if it > 1 && change < cc.turb.convergence_tol {
                println!("converged: max relative change below {}", g(f64::from(cc.turb.convergence_tol)));
                break;
            }
        }
    }

    gpu.sync()?;
    let wall = t_loop.elapsed().as_secs_f64();

    // `done` is at least 1 whenever the loop ran; guard anyway, because
    // `-iters 0` is a legal thing for a user to type.
    let n = done.max(1) as f64;

    // Scientific with three digits, not `%g`: `std::scientific` and
    // `setprecision(3)` are sticky on an `ostream`, and the residual line
    // inside the loop already set both. Printing these with the default
    // format would put a diff on the summary line of every run.
    println!(
        "\n{done} iterations in {} s  ->  {} ms/iteration  ->  {} Mcell-iterations/s",
        sci(wall, 3),
        sci((wall / n) * 1e3, 3),
        sci((hm.n_cells as f64 * n / wall) / 1e6, 3)
    );

    // ---- write ------------------------------------------------------------
    if o.do_write {
        let wt = if o.write_time.is_empty() {
            cc.write_time.clone()
        } else {
            o.write_time.clone()
        };

        // Carry the ORIGINAL boundary types across so the written fields can
        // be used as the start time of another run.
        let mut out_k = seed_types(&raw_k);
        let mut out_w = seed_types(&raw_w);
        // The case's own nut patch types, so the written field still says
        // `nutLowReWallFunction` where the input did - and so the result
        // directory can be used as the start time of another run without
        // silently becoming a `calculated` wall.
        let mut out_nut = match &raw_nut {
            Some(raw) => seed_types(raw),
            None => RawScalarField::default(),
        };

        harvest_scalar_field(&gpu, &mut out_k, model.k(), &hm)?;
        harvest_scalar_field(&gpu, &mut out_w, model.omega(), &hm)?;
        harvest_scalar_field(&gpu, &mut out_nut, model.nut(), &hm)?;

        // The flux this run was driven by. Without it a restart from this
        // directory re-derives one, and a re-derived flux is not the
        // conservative one the pressure equation produced - SPEC-LIT 5.1.
        let mut out_phi = RawScalarField::default();
        harvest_surface_scalar_field(&gpu, &mut out_phi, &phi, &hm)?;

        out_k.dimensions = "[0 2 -2 0 0 0 0]".to_string();
        out_w.dimensions = "[0 0 -1 0 0 0 0]".to_string();
        out_nut.dimensions = "[0 2 -1 0 0 0 0]".to_string();

        // One seam call per requested format, replacing what used to be four
        // scattered `fields::write_*` sites - see `ofgpu::io::writer`.
        let foam_fields = [
            ofgpu::io::FoamField::scalar("k", &out_k),
            ofgpu::io::FoamField::scalar("omega", &out_w),
            ofgpu::io::FoamField::scalar("nut", &out_nut),
            ofgpu::io::FoamField::surface("phi", &out_phi),
        ];
        let vis_fields = [
            ofgpu::io::OutputField::scalar("k", &out_k.internal),
            ofgpu::io::OutputField::scalar("omega", &out_w.internal),
            ofgpu::io::OutputField::scalar("nut", &out_nut.internal),
        ];
        let cart = ofgpu::pressure::cartesian::detect(&hm)
            .ok()
            .map(|c| ofgpu::io::cartesian_info(&hm, &c));
        let ctx = ofgpu::io::WriteCtx {
            time: 0.0,
            step: done as usize,
            name: &wt,
            mesh: &hm,
            cart: cart.as_ref(),
            fields: &vis_fields,
            foam: &foam_fields,
        };
        let mut writers = build_writers(&o.case_dir, "kOmega", &o.output)?;
        for w in &mut writers {
            w.write_step(&ctx)?;
        }

        println!("written to {}", o.case_dir.join(&wt).display());
    }

    Ok(())
}

/// A destination field carrying only the source's boundary *type* strings.
fn seed_types(src: &RawScalarField) -> RawScalarField {
    src.types_only()
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();

    let o = match parse(&args) {
        Ok(o) => o,
        Err(e) => {
            eprintln!("\nerror: {e}");
            return ExitCode::from(1);
        }
    };

    match run(&o) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("\nerror: {e}");
            ExitCode::from(1)
        }
    }
}
