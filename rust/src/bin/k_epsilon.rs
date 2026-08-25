// meteor-cfd - Copyright (c) 2026 주식회사 메테오시뮬레이션 (Meteo Simulation Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

//! `ofgpu-k-epsilon` - run the GPU-native k-epsilon model on a case.
//!
//! `<case>` is either an OpenFOAM case DIRECTORY (`constant/polyMesh` + `0/`,
//! as always) or a single `.jsonc`/`.json` case FILE - told apart by
//! extension (`common::is_json_case`). The JSONC path builds its mesh
//! straight into memory (`blockgen::build_mesh`) and its fields off the
//! case's own `initial`/`patches`, with no disk polyMesh or `0/` directory at
//! any point - `docs/05-io-redesign.md` phase 1 (B3). Output for a JSONC case
//! goes to `<stem>_jsonc/` next to the file (`common::json_case_output_dir`),
//! never into a same-named OpenFOAM directory that might already exist.
//!
//! ```text
//! ofgpu-k-epsilon <caseDir|case.jsonc> [options]
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
//! The velocity field is held frozen: this program solves the two turbulence
//! transport equations on a given `U` and `phi`, which is exactly the part of
//! `simpleFoam` that the model owns. The momentum and pressure equations are
//! not ported.
//!
//! After the mesh and the initial fields are uploaded, no field data crosses
//! the PCIe bus again until the results are written.
//!
//! Provenance: the driver - argument parsing, case loading, the reporting
//! loop - was carried across from this project's own earlier C++ driver. The
//! model it drives was rewritten from SPEC-LIT.md; see src/models/k_epsilon.rs.

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
use ofgpu::io::case::{find_start_time, model_coeff};
use ofgpu::io::fields::{read_scalar_field, read_vector_field, RawScalarField};
use ofgpu::models::{select_turbulence_model, KEpsilon, KEpsilonCoeffs, RasModel};
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
        "usage: ofgpu-k-epsilon <caseDir> [-iters N] [-fixedIters N] \
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

    // Set before any dictionary is opened, so every rejection this run makes
    // sees the same policy.
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
    println!("{}", device_banner(&gpu, "k-epsilon")?);

    // ---- mesh + controls ----------------------------------------------------
    // `common::load_case` is the docs/05-io-redesign.md phase 1 seam: a
    // `.jsonc`/`.json` case FILE builds its mesh straight into memory
    // (`blockgen::build_mesh`, no disk polyMesh at all) and an OpenFOAM case
    // DIRECTORY still reads one, same as before - either way this driver
    // gets the same `(HostMesh, CaseControls)` pair from here on. `json` is
    // `Some` only on the JSONC path, and carries the raw fields (`U`, `k`,
    // `epsilon`, `nut`, ...) a `0/` directory would otherwise be read from.
    let t0 = Instant::now();

    let (hm, mut cc, json) = common::load_case(&o.case_dir)?;
    hm.print_report();

    let mesh = GpuMesh::upload(&gpu, &hm)?;
    gpu.sync()?;

    println!("mesh uploaded in {} s", g(t0.elapsed().as_secs_f64()));

    // ---- controls ---------------------------------------------------------

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

    // The case decides which model runs. Before this dispatch existed, `RAS {
    // model kOmegaSST; }` ran standard k-epsilon here and said nothing.
    let selection = select_turbulence_model(&cc)?;
    match selection.model {
        RasModel::KEpsilon | RasModel::Laminar => {}
        other => {
            return Err(Error::Config(format!(
                "this case asks for the {} model; run it with ofgpu-{} \
                 (ofgpu-k-epsilon builds only kEpsilon)",
                other.name(),
                other.name().to_lowercase().replace("kepsilon", "k-epsilon").replace("komega", "k-omega")
            )));
        }
    }

    if !selection.active {
        println!(
            "turbulence is off in this case ({}): nu_t is frozen at zero and \
             the model will not be corrected",
            if selection.model == RasModel::Laminar {
                "simulationType laminar"
            } else {
                "RAS { turbulence off; }"
            }
        );
    }

    let d = KEpsilonCoeffs::default();
    let coeffs = KEpsilonCoeffs {
        cmu: model_coeff(&cc, "Cmu", d.cmu),
        c1: model_coeff(&cc, "C1", d.c1),
        c2: model_coeff(&cc, "C2", d.c2),
        c3: model_coeff(&cc, "C3", d.c3),
        sigmak: model_coeff(&cc, "sigmak", d.sigmak),
        sigma_eps: model_coeff(&cc, "sigmaEps", d.sigma_eps),
    };

    println!(
        "nu = {} | Cmu {} C1 {} C2 {} sigmak {} sigmaEps {}",
        g(f64::from(cc.nu)),
        g(f64::from(coeffs.cmu)),
        g(f64::from(coeffs.c1)),
        g(f64::from(coeffs.c2)),
        g(f64::from(coeffs.sigmak)),
        g(f64::from(coeffs.sigma_eps))
    );

    // ---- fields -------------------------------------------------------------
    // JSONC has no `0/` directory: every field comes off `json`'s own
    // `LoweredScalarField`/`LoweredVectorField`
    // (`LoweredScalarField::to_raw`/`LoweredVectorField::to_raw`, sized now
    // that `hm.n_cells` is known) rather than off disk. `t_dir` stays `None`
    // on that path - there is no time directory to look a `phi`/`nut` file up
    // in, which is also why phi is always RECONSTRUCTED rather than read for
    // a JSONC case (see below).
    let (raw_u, raw_k, raw_e, raw_nut, t_dir) = if let Some(lc) = &json {
        let u = lc.u_field.to_raw(hm.n_cells);
        let k = lc
            .k_field
            .as_ref()
            .ok_or_else(|| Error::Config(
                "this case does not solve k (initial.k is absent); ofgpu-k-epsilon needs it"
                    .to_string(),
            ))?
            .to_raw(hm.n_cells);
        let e = lc
            .epsilon_field
            .as_ref()
            .ok_or_else(|| Error::Config(
                "this case does not solve epsilon (initial.epsilon is absent); ofgpu-k-epsilon \
                 needs it"
                    .to_string(),
            ))?
            .to_raw(hm.n_cells);
        let nut = lc.nut_field.as_ref().map(|f| f.to_raw(hm.n_cells));
        (u, k, e, nut, None)
    } else {
        let t = find_start_time(&o.case_dir)?;
        let t_dir = o.case_dir.join(&t);

        let u = read_vector_field(&t_dir.join("U"), hm.n_cells)?;
        let k = read_scalar_field(&t_dir.join("k"), hm.n_cells)?;
        let e = read_scalar_field(&t_dir.join("epsilon"), hm.n_cells)?;

        // SPEC-LIT 15.5: nut's wall treatment comes from nut's own patch
        // types and epsilon's wall-cell constraint from epsilon's, never one
        // from the other.
        let nut_path = t_dir.join("nut");
        let mut k = k;
        let mut e = e;
        let mut nut = if nut_path.exists() {
            Some(read_scalar_field(&nut_path, hm.n_cells)?)
        } else {
            None
        };
        // SPEC-LIT 29.1: the four per-field wall types must form one row.
        ofgpu::field_setup::validate_wall_rows(
            &hm.patches,
            nut.as_mut(),
            Some(&mut k),
            Some(&mut e),
            None,
        )?;
        (u, k, e, nut, Some(t_dir))
    };

    let fk = FieldKernels::new(&gpu)?;

    let mut u = GpuVectorField::zeros(&gpu, &mesh, "U")?;
    setup_vector_field(&gpu, &mut u, &raw_u, &hm)?;
    correct_boundary_conditions_vector(&gpu, &fk, &mut u, &mesh)?;

    let mut phi = GpuSurfaceScalarField::zeros(&gpu, &mesh, "phi")?;
    match &t_dir {
        Some(t_dir) => load_phi(&gpu, &mut phi, &u, &hm, t_dir)?,
        // No time directory to look a `phi` file up in - a JSONC case never
        // carries one, so this is not the fallback `load_phi` takes when a
        // FILE happens to be missing, it is the only option there is.
        None => {
            compute_phi_from_u(&gpu, &mut phi, &u, &hm)?;
            println!("phi reconstructed as interpolate(U) & Sf");
        }
    }

    println!(
        "max |sum_f phi| per cell = {}   (0 means the flux is discretely conservative)",
        g(f64::from(max_div_phi(&gpu, &phi, &hm)?))
    );

    let wf_faces = WallFaces::from_case(&raw_e, raw_nut.as_ref(), &hm)?;
    let roughness = NutRoughness::from_case(raw_nut.as_ref(), &hm)?;

    let flow = FlowState::new(&u, &phi, cc.nu);

    let mut model = KEpsilon::new(
        &gpu,
        &hm,
        &mesh,
        coeffs,
        cc.turb,
        wall_coeffs_from_case(&cc.wall),
        &wf_faces,
        &roughness,
    )?;

    // `turbulentIntensityKineticEnergyInlet` is 3/2 (I |U|)^2 and
    // `turbulentMixingLengthDissipationRateInlet` is C_mu^{3/4} k^{3/2}/L, so
    // k needs U and epsilon needs k. Hence the order, and hence the boundary
    // values being read back between the two.
    let u_b = gpu.download(&u.bf)?;
    setup_scalar_field_with(
        &gpu,
        model.k_mut(),
        &raw_k,
        &hm,
        BcInputs {
            u_b: Some(&u_b),
            cmu: Some(coeffs.cmu),
            ..Default::default()
        },
    )?;

    let k_b = gpu.download(&model.k().bf)?;
    setup_scalar_field_with(
        &gpu,
        model.epsilon_mut(),
        &raw_e,
        &hm,
        BcInputs {
            u_b: Some(&u_b),
            k_b: Some(&k_b),
            cmu: Some(coeffs.cmu),
            ..Default::default()
        },
    )?;

    if let Some(raw_nut) = &raw_nut {
        setup_scalar_field(&gpu, model.nut_mut(), raw_nut, &hm)?;
    }

    update_inlet_outlet(&gpu, model.k_mut(), &phi, &hm)?;
    update_inlet_outlet(&gpu, model.epsilon_mut(), &phi, &hm)?;

    if selection.active {
        model.initialise(&gpu, &flow)?;
    } else {
        model.freeze_nut(&gpu)?;
    }

    // ---- time loop - device only from here to the write -------------------
    println!(
        "\niterating {} times, relax k {} epsilon {}{}",
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
            // `turbulence off` and `simulationType laminar` both mean the
            // closure does not run. Not "runs and is ignored" - the fields
            // must come out of the run exactly as they went in, and nu_t must
            // stay at the zero `freeze_nut` put there.
            done = 1;
            break;
        }
        let (eps_perf, k_perf) = model.correct(&gpu, &flow)?;
        done = it;

        if it % cc.turb.convergence_check_every == 0 || it == 1 {
            let change = model.convergence_measure(&gpu)?;

            println!(
                "{it:>7}  epsilon res {} ({})  k res {} ({})  max dk/k {}",
                sci(f64::from(eps_perf.initial_residual), 3),
                eps_perf.n_iterations,
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
                        ("epsilon", eps_perf.initial_residual),
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

        // Carry the ORIGINAL boundary types across. `harvest_scalar_field`
        // only fills in a type where none is set, so seeding them here keeps
        // the written fields round-trippable: the result directory can be
        // used as the start time of another run.
        let mut out_k = seed_types(&raw_k);
        let mut out_e = seed_types(&raw_e);
        // The case's own nut patch types, so the written field still says
        // `nutLowReWallFunction` where the input did - and so the result
        // directory can be used as the start time of another run without
        // silently becoming a `calculated` wall.
        let mut out_nut = match &raw_nut {
            Some(raw) => seed_types(raw),
            None => RawScalarField::default(),
        };

        harvest_scalar_field(&gpu, &mut out_k, model.k(), &hm)?;
        harvest_scalar_field(&gpu, &mut out_e, model.epsilon(), &hm)?;
        harvest_scalar_field(&gpu, &mut out_nut, model.nut(), &hm)?;

        // The flux this run was driven by. Without it a restart from this
        // directory re-derives one, and a re-derived flux is not the
        // conservative one the pressure equation produced - SPEC-LIT 5.1.
        let mut out_phi = RawScalarField::default();
        harvest_surface_scalar_field(&gpu, &mut out_phi, &phi, &hm)?;

        out_k.dimensions = "[0 2 -2 0 0 0 0]".to_string();
        out_e.dimensions = "[0 2 -3 0 0 0 0]".to_string();
        out_nut.dimensions = "[0 2 -1 0 0 0 0]".to_string();

        // One seam call per requested format, replacing what used to be four
        // scattered `fields::write_*` sites - see `ofgpu::io::writer`.
        let foam_fields = [
            ofgpu::io::FoamField::scalar("k", &out_k),
            ofgpu::io::FoamField::scalar("epsilon", &out_e),
            ofgpu::io::FoamField::scalar("nut", &out_nut),
            ofgpu::io::FoamField::surface("phi", &out_phi),
        ];
        let vis_fields = [
            ofgpu::io::OutputField::scalar("k", &out_k.internal),
            ofgpu::io::OutputField::scalar("epsilon", &out_e.internal),
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
        // The OUTPUT root: the case directory itself for an OpenFOAM case,
        // `common::json_case_output_dir` for a JSONC one - `o.case_dir` is a
        // FILE in that case, not a directory `FoamWriter`/`VtuWriter`/... can
        // write time directories under.
        let out_root = common::output_root(&o.case_dir);
        let mut writers = build_writers(&out_root, "kEpsilon", &o.output)?;
        for w in &mut writers {
            w.write_step(&ctx)?;
        }

        println!("written to {}", out_root.join(&wt).display());
    }

    Ok(())
}

/// A destination field carrying only the source's boundary *type* strings.
///
/// `types_only` carries the pattern keys across too, so a case written with
/// `".*"` keeps its types on the way out; `harvest_scalar_field` then expands
/// them into one explicit entry per patch.
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
