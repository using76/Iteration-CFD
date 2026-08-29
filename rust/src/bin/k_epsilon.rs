// meteor-cfd - Copyright (c) 2026 주식회사 메테오시뮬레이션 (Meteo Simulation Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

//! `ofgpu-k-epsilon` - run a GPU-native k-epsilon model on a case.
//!
//! THREE models, not one: standard k-epsilon (SPEC-LIT §6.1), realizable
//! (§40) and RNG (§41). All three transport the same two fields under the
//! same two names, read the same `0/k` and `0/epsilon`, and write the same
//! three outputs, so which one runs is a line in
//! `constant/momentumTransport` and nothing else about the run changes.
//! `LaunderSharmaKE` is deliberately NOT here: it needs `wallTreatment
//! lowRe` and a wall-resolving mesh, which is a different case, not a
//! different coefficient set.
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
//!
//! Provenance: ORIGINAL driver code over LITERATURE numerics. The k-epsilon
//! model itself is cited in `src/turbulence.rs`/`src/models/*` and SPEC-LIT.md
//! S6.1; this file is argument parsing, case loading, the iteration order and
//! the reporting loop, which are this project's own (`PROVENANCE.md`,
//! `src/bin/*`). No GPL-licensed source was consulted.

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use ofgpu::field::{GpuScalarField, GpuSurfaceScalarField, GpuVectorField};
use ofgpu::field_ops::{correct_boundary_conditions_vector, FieldKernels};
use ofgpu::field_setup::{
    compute_phi_from_u, harvest_scalar_field, harvest_surface_scalar_field, max_div_phi,
    setup_scalar_field,
    setup_scalar_field_with, setup_vector_field, update_inlet_outlet, wall_coeffs_from_case,
    BcInputs, NutRoughness, WallFaces,
};
use ofgpu::io::case::{find_start_time, model_coeff};
use ofgpu::io::fields::{read_scalar_field, read_vector_field, RawScalarField};
use ofgpu::models::{
    realizable_ke_coeffs, rng_ke_coeffs, select_turbulence_model, KEpsilon, KEpsilonCoeffs,
    RasModel, RealizableKe, RealizableKeCoeffs, RngKe, RngKeCoeffs,
};
use ofgpu::solver::SolverPerformance;
use ofgpu::turbulence::FlowState;
use ofgpu::{Error, GpuMesh, Label, Result, Scalar};

#[path = "common/mod.rs"]
mod common;

use common::{atoi, device_banner, g, next_arg, parse_output_formats, sci, OutputFormat};

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
    /// Whether the command line NAMED `-output` - SPEC-LIT §44.6. Not the
    /// same question as what it is set to: it defaults to `foam`, so every
    /// run has a value. A case with an `output` block and a command line
    /// with `-output` name the same thing twice, and that is an error.
    output_flags: Vec<&'static str>,
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
        output_flags: Vec::new(),
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
            "-output" => {
                o.output = parse_output_formats(&next_arg(args, &mut i)?)?;
                o.output_flags.push("-output");
            }
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

    // SPEC-LIT §13.4. Three settings a case can name that this driver cannot
    // honour, refused by name rather than read and dropped:
    //
    //   * `physics.gravity` / `constant/g` - §17's `G_b` needs a temperature
    //     field, and this driver reads none.
    //   * `run.adjustTimeStep`, `run.maxCo`.
    //
    // The `output` block used to be on that list and is not any more:
    // SPEC-LIT §44 implemented it, and this driver honours everything in it
    // that a STEADY single-write run can honour - the formats, the field
    // selection, the voxel precision. What it cannot honour it refuses by
    // name, immediately below.
    //
    // And two it CAN name and this driver simply has no equation for:
    // `physics.fluid.Pr`/`Prt` are required fields of the JSONC format, so
    // refusing them would refuse every JSONC case; they get §13.4.2's other
    // half - one printed line - instead.
    common::refuse_buoyancy_without_temperature(&o.case_dir, &cc, json.as_ref(), "ofgpu-k-epsilon")?;
    common::refuse_non_orth_correctors_without_another_equation(&cc, "ofgpu-k-epsilon")?;
    common::refuse_rheology_without_momentum(&cc, "ofgpu-k-epsilon")?;
    common::refuse_unimplemented_blocks(json.as_ref())?;

    // SPEC-LIT §44, resolved before the mesh is uploaded so a case that asks
    // for something this driver cannot do fails before any kernel launches.
    let mut output_plan = common::output_plan(json.as_ref())?;
    if let Some(plan) = &mut output_plan {
        // Under `-permissive` this answers `false`, and the warning it just
        // printed ("substituting the command line") is then this driver's
        // job to make TRUE rather than a guess about it.
        if !common::refuse_output_named_twice(plan, &o.output_flags)? {
            output_plan = None;
        }
    }
    if let Some(plan) = &mut output_plan {
        // §44.4: this driver advances an iteration counter, not a clock.
        plan.refuse_interval_when_steady(
            "ofgpu-k-epsilon",
            "it writes its converged state once, into the time directory -write names",
        )?;
        // §44.1: and it writes no checkpoint of any kind - there is no
        // -restartWrite here either.
        plan.refuse_restart(
            "ofgpu-k-epsilon",
            "ofgpu-fire, ofgpu-buoyant and ofgpu-vof do write .mcr checkpoints, and ofgpu-fire honours output.restart",
        )?;
        plan.refuse_visualisation_on_a_non_cartesian_mesh(
            ofgpu::pressure::cartesian::detect(&hm).is_ok(),
        )?;
        // §44.2's early half. This driver's cell fields are these three and
        // only these three; `phi` is a SURFACE field and goes only to the
        // OpenFOAM writer, which is why it is not on the menu.
        plan.check_fields(&["k", "epsilon", "nut"])?;
        if plan.is_empty() {
            output_plan = None;
        }
    }

    if let Some(l) = &json {
        println!(
            "    physics.fluid         Pr and Prt are not read by ofgpu-k-epsilon: it \
solves k and epsilon on a frozen U and transports no scalar for them to diffuse \
(ofgpu-plume, ofgpu-buoyant, ofgpu-fire do)"
        );
        println!(
            "    run                   endTime {} -> {} outer iteration(s), deltaT {}",
            g(f64::from(l.run.end_time)),
            cc.turb.n_outer_iterations,
            g(f64::from(l.run.delta_t))
        );
    }

    // The case decides which model runs. Before this dispatch existed, `RAS {
    // model kOmegaSST; }` ran standard k-epsilon here and said nothing.
    let selection = select_turbulence_model(&cc)?;
    match selection.model {
        RasModel::KEpsilon
        | RasModel::RealizableKE
        | RasModel::RNGkEpsilon
        | RasModel::Laminar => {}
        other => {
            return Err(Error::Config(format!(
                "this case asks for the {} model; run it with {} \
                 (ofgpu-k-epsilon builds kEpsilon, realizableKE and RNGkEpsilon)",
                other.name(),
                common::driver_for(other),
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

    // One coefficient set per model, and ONLY the selected one is read: each
    // reader refuses, by name, the entries its own model does not consult
    // (SPEC-LIT §40.6, §41.4), so reading all three would refuse a perfectly
    // good `kEpsilon` case for carrying a `C1`.
    let variant = match selection.model {
        RasModel::RealizableKE => {
            let c = realizable_ke_coeffs(&cc)?;
            println!(
                "nu = {} | realizableKE (SPEC-LIT S40): A0 {} C2 {} sigmak {} sigmaEps {}",
                g(f64::from(cc.nu)),
                g(f64::from(c.a0)),
                g(f64::from(c.c2)),
                g(f64::from(c.sigmak)),
                g(f64::from(c.sigma_eps))
            );
            println!(
                "    C_mu is a FIELD here (S40.4): log-layer value {}, kappa implied by \
these coefficients {} (S40.8). `Cmu {}` reaches the WALL FUNCTIONS and the \
epsilon bound - not nu_t",
                g(f64::from(c.log_layer_cmu())),
                g(f64::from(c.implied_kappa())),
                g(f64::from(cc.wall.cmu)),
            );
            Variant::Realizable(c)
        }
        RasModel::RNGkEpsilon => {
            let c = rng_ke_coeffs(&cc)?;
            println!(
                "nu = {} | RNGkEpsilon (SPEC-LIT S41): Cmu {} C1 {} C2 {} alphak {} \
alphaEps {} eta0 {} beta {}",
                g(f64::from(cc.nu)),
                g(f64::from(c.cmu)),
                g(f64::from(c.c1)),
                g(f64::from(c.c2)),
                g(f64::from(c.alpha_k)),
                g(f64::from(c.alpha_eps)),
                g(f64::from(c.eta0)),
                g(f64::from(c.beta))
            );
            println!(
                "    diffusivity is alpha (nu + nu_t), NOT nu + nu_t/sigma (S41.2); \
kappa implied by these coefficients {} (S41.3)",
                g(f64::from(c.implied_kappa()))
            );
            Variant::Rng(c)
        }
        _ => {
            let d = KEpsilonCoeffs::default();
            let c = KEpsilonCoeffs {
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
                g(f64::from(c.cmu)),
                g(f64::from(c.c1)),
                g(f64::from(c.c2)),
                g(f64::from(c.sigmak)),
                g(f64::from(c.sigma_eps))
            );
            Variant::Standard(c)
        }
    };

    // The `C_mu` the INLET boundary conditions are built from -
    // `turbulentMixingLengthDissipationRateInlet` is `C_mu^{3/4} k^{3/2}/L`,
    // which follows from `nu_t = C_mu k^2/eps` and therefore has to be the
    // model's own constant. Realizable's `C_mu` is not a constant at all, so
    // it takes the log-layer value the wall functions use (SPEC-LIT §40.5).
    let bc_cmu = variant.inlet_cmu(cc.wall.cmu);

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

    let wall = wall_coeffs_from_case(&cc.wall);
    let mut model = match variant {
        Variant::Standard(c) => KeModel::Standard(KEpsilon::new(
            &gpu, &hm, &mesh, c, cc.turb, wall, &wf_faces, &roughness,
        )?),
        Variant::Realizable(c) => KeModel::Realizable(RealizableKe::new(
            &gpu, &hm, &mesh, c, cc.turb, wall, &wf_faces, &roughness,
        )?),
        Variant::Rng(c) => KeModel::Rng(RngKe::new(
            &gpu, &hm, &mesh, c, cc.turb, wall, &wf_faces, &roughness,
        )?),
    };

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
            cmu: Some(bc_cmu),
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
            cmu: Some(bc_cmu),
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
        // SPEC-LIT §44: the case's `output` block, or the command line's
        // `-output`, never both (§44.6 refused that above). `force` is set
        // because this driver writes exactly once, at the end - which is
        // also why every `interval` in the block was refused above.
        let mut pipeline = match &output_plan {
            Some(plan) => {
                ofgpu::io::OutputPipeline::from_plan(plan, &out_root, model.tag(), "restart")?
            }
            None => ofgpu::io::OutputPipeline::from_command_line(
                &out_root,
                model.tag(),
                &o.output,
                0.0,
            )?,
        };
        println!("{}", pipeline.describe());
        pipeline.write(&ctx, 0.0, true)?;

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

/// Which coefficient set the case named, before the mesh exists.
///
/// Separate from [`KeModel`] because the coefficients are read (and their
/// §13.4 refusals raised) at the top of `run`, while the models themselves
/// cannot be built until the mesh and the wall faces are.
enum Variant {
    Standard(KEpsilonCoeffs),
    Realizable(RealizableKeCoeffs),
    Rng(RngKeCoeffs),
}

impl Variant {
    /// The `C_mu` the `turbulentMixingLengthDissipationRateInlet` and
    /// `turbulentIntensityKineticEnergyInlet` triples are evaluated with.
    ///
    /// `fallback` is the wall functions' own (`RAS { Cmu; }`, default 0.09),
    /// which is what realizable takes: SPEC-LIT §40.4 shows its own `C_mu` in
    /// an equilibrium shear layer IS 0.09, and an inlet has no strain history
    /// for the variable form to read anyway.
    fn inlet_cmu(&self, fallback: Scalar) -> Scalar {
        match self {
            Self::Standard(c) => c.cmu,
            Self::Realizable(_) => fallback,
            Self::Rng(c) => c.cmu,
        }
    }
}

/// The three models this driver builds, behind one match.
///
/// An enum rather than a trait object: `src/models/mod.rs` argues at length
/// against a turbulence trait, and the argument applies here - this driver
/// KNOWS the three concrete types it can build, so the dispatch is a `match`
/// the compiler can see through rather than a virtual call. (The coupled
/// drivers are the other case, and they have `CoupledTurbulence` for exactly
/// the reason `coupled.rs`'s own doc gives.)
enum KeModel<'m> {
    Standard(KEpsilon<'m>),
    Realizable(RealizableKe<'m>),
    Rng(RngKe<'m>),
}

impl<'m> KeModel<'m> {
    fn k(&self) -> &GpuScalarField {
        match self {
            Self::Standard(m) => m.k(),
            Self::Realizable(m) => m.k(),
            Self::Rng(m) => m.k(),
        }
    }
    fn k_mut(&mut self) -> &mut GpuScalarField {
        match self {
            Self::Standard(m) => m.k_mut(),
            Self::Realizable(m) => m.k_mut(),
            Self::Rng(m) => m.k_mut(),
        }
    }
    fn epsilon(&self) -> &GpuScalarField {
        match self {
            Self::Standard(m) => m.epsilon(),
            Self::Realizable(m) => m.epsilon(),
            Self::Rng(m) => m.epsilon(),
        }
    }
    fn epsilon_mut(&mut self) -> &mut GpuScalarField {
        match self {
            Self::Standard(m) => m.epsilon_mut(),
            Self::Realizable(m) => m.epsilon_mut(),
            Self::Rng(m) => m.epsilon_mut(),
        }
    }
    fn nut(&self) -> &GpuScalarField {
        match self {
            Self::Standard(m) => m.nut(),
            Self::Realizable(m) => m.nut(),
            Self::Rng(m) => m.nut(),
        }
    }
    fn nut_mut(&mut self) -> &mut GpuScalarField {
        match self {
            Self::Standard(m) => m.nut_mut(),
            Self::Realizable(m) => m.nut_mut(),
            Self::Rng(m) => m.nut_mut(),
        }
    }
    fn initialise(&mut self, gpu: &ofgpu::Gpu, flow: &FlowState) -> Result<()> {
        match self {
            Self::Standard(m) => m.initialise(gpu, flow),
            Self::Realizable(m) => m.initialise(gpu, flow),
            Self::Rng(m) => m.initialise(gpu, flow),
        }
    }
    fn freeze_nut(&mut self, gpu: &ofgpu::Gpu) -> Result<()> {
        match self {
            Self::Standard(m) => m.freeze_nut(gpu),
            Self::Realizable(m) => m.freeze_nut(gpu),
            Self::Rng(m) => m.freeze_nut(gpu),
        }
    }
    fn correct(
        &mut self,
        gpu: &ofgpu::Gpu,
        flow: &FlowState,
    ) -> Result<(SolverPerformance, SolverPerformance)> {
        match self {
            Self::Standard(m) => m.correct(gpu, flow),
            Self::Realizable(m) => m.correct(gpu, flow),
            Self::Rng(m) => m.correct(gpu, flow),
        }
    }
    fn convergence_measure(&mut self, gpu: &ofgpu::Gpu) -> Result<Scalar> {
        match self {
            Self::Standard(m) => m.convergence_measure(gpu),
            Self::Realizable(m) => m.convergence_measure(gpu),
            Self::Rng(m) => m.convergence_measure(gpu),
        }
    }
    /// The name the writer stamps on the output directory.
    fn tag(&self) -> &'static str {
        match self {
            Self::Standard(_) => "kEpsilon",
            Self::Realizable(_) => "realizableKE",
            Self::Rng(_) => "RNGkEpsilon",
        }
    }
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

// ==========================================================================
//  Tests - SPEC-LIT 13.4.1's standing requirement
// ==========================================================================
//
// Named `k_epsilon_tests` rather than `tests` because `common/mod.rs` is included by
// `#[path]` and already contributes a `tests` module to this crate.

#[cfg(test)]
mod k_epsilon_tests {
    use super::*;
    use common::knobs::{apply, assert_none_inert, scratch_dir, written_state, Knob, NO_PRE};
    use ofgpu::blockgen::{write_case, CaseKind};

    fn argv(v: &[&str]) -> Vec<String> {
        std::iter::once("ofgpu-k-epsilon".to_string())
            .chain(v.iter().map(|s| (*s).to_string()))
            .collect()
    }

    fn channel(tag: &str) -> PathBuf {
        let dir = scratch_dir(tag);
        let case = dir.join("case");
        write_case(&case, CaseKind::Channel, 16, 10, 1).expect("generate the channel case");
        // `write_case` writes `RAS { model kEpsilon; }` for every case kind;
        // this driver builds kEpsilon and refuses anything else by name.
        apply(
            &case,
            &Knob {
                label: "RAS/model",
                file: "constant/momentumTransport",
                from: "    model           kEpsilon;",
                to: "    model           kEpsilon;",
                pre: NO_PRE,
            },
            false,
        );
        case
    }

    /// Build a fresh channel case, apply `k` if `side` is set, run this
    /// driver's own `parse` + `run`, and return everything it wrote.
    fn run_knob(k: &Knob, side: bool, tag: &str) -> Vec<(String, String)> {
        let case = channel(tag);
        apply(&case, k, side);

        let args = argv(&[
            case.to_string_lossy().as_ref(),
            "-iters",
            "20",
            "-check",
            "2",
        ]);
        let o = parse(&args).expect("the knob command line must parse");
        run(&o).expect("the knob case must run");

        // `controlDict`'s endTime is 1, so the run writes `<case>/1`.
        let out = written_state(&case.join("1"));
        assert!(!out.is_empty(), "the run wrote nothing to compare");
        out
    }

    /// **The standing test SPEC-LIT 13.4.1 requires of every setting this
    /// driver claims to honour.**
    ///
    /// `laplacianSchemes`/`snGradSchemes` is absent for 13.4.1's one
    /// admissible reason - §2.4's correction vanishes identically on the
    /// orthogonal box `blockgen` builds.
    #[test]
    fn every_wired_setting_changes_what_the_run_writes() {
        if ofgpu::Gpu::new(0).is_err() {
            return;
        }

        let cases: Vec<Knob> = vec![
            Knob {
                label: "divSchemes/div(phi,k)",
                file: "system/fvSchemes",
                from: "div(phi,k)       bounded Gauss upwind;",
                to: "div(phi,k)       Gauss linear;",
                pre: NO_PRE,
            },
            Knob {
                label: "divSchemes/div(phi,epsilon)",
                file: "system/fvSchemes",
                from: "div(phi,epsilon) bounded Gauss upwind;",
                to: "div(phi,epsilon) Gauss linear;",
                pre: NO_PRE,
            },
            // `gradSchemes` is read by a convection scheme that carries a
            // limiter or a deferred correction and by nothing else, and the
            // generated case's `div(phi,k) bounded Gauss upwind` carries
            // neither - so the gradient-reading entry goes in `pre`, on BOTH
            // sides, and the pair differs in `gradSchemes` alone.
            Knob {
                label: "gradSchemes/default",
                file: "system/fvSchemes",
                from: "gradSchemes\n{\n    default         Gauss linear;\n}",
                to: "gradSchemes\n{\n    default         cellLimited Gauss linear 1;\n}",
                pre: (
                    "system/fvSchemes",
                    "div(phi,k)       bounded Gauss upwind;",
                    "div(phi,k)       Gauss limitedLinear 1;",
                ),
            },
            Knob {
                label: "relaxationFactors/equations/k",
                file: "system/fvSolution",
                from: "        k               0.7;",
                to: "        k               0.3;",
                pre: NO_PRE,
            },
            Knob {
                label: "relaxationFactors/equations/epsilon",
                file: "system/fvSolution",
                from: "        epsilon         0.7;",
                to: "        epsilon         0.3;",
                pre: NO_PRE,
            },
            Knob {
                label: "solvers/k/tolerance",
                file: "system/fvSolution",
                from: "    k\n    {\n        solver          PBiCGStab;\n        preconditioner  diagonal;\n        tolerance       1e-08;\n        relTol          0.01;",
                to: "    k\n    {\n        solver          PBiCGStab;\n        preconditioner  diagonal;\n        tolerance       1e-02;\n        relTol          0.5;",
                pre: NO_PRE,
            },
            // SPEC-LIT §40 and §41. `realizableKE` and `RNGkEpsilon` were
            // RECOGNISED-AND-REFUSED names until this batch; now they select
            // models, and these knobs are what says they select DIFFERENT
            // ones. Two per model, as §38.7 established: the selector itself,
            // and one coefficient OF the selected model - a reader can wire
            // the selector and still drop the numbers.
            Knob {
                label: "RAS/model realizableKE (SPEC-LIT 40)",
                file: "constant/momentumTransport",
                from: "    model           kEpsilon;",
                to: "    model           realizableKE;",
                pre: NO_PRE,
            },
            Knob {
                label: "RAS/model RNGkEpsilon (SPEC-LIT 41)",
                file: "constant/momentumTransport",
                from: "    model           kEpsilon;",
                to: "    model           RNGkEpsilon;",
                pre: NO_PRE,
            },
            // The SAME model on both sides, differing in `A0` alone - the
            // constant SPEC-LIT §40.3 derives, and the one a case would set
            // to reach the NASA TM's printed 4.0.
            Knob {
                label: "RAS/A0 (realizableKE, SPEC-LIT 40.3)",
                file: "constant/momentumTransport",
                from: "    turbulence      on;",
                to: "    turbulence      on;\n    A0              4.0;",
                pre: (
                    "constant/momentumTransport",
                    "    model           kEpsilon;",
                    "    model           realizableKE;",
                ),
            },
            Knob {
                label: "RAS/C2 (realizableKE)",
                file: "constant/momentumTransport",
                from: "    turbulence      on;",
                to: "    turbulence      on;\n    C2              1.7;",
                pre: (
                    "constant/momentumTransport",
                    "    model           kEpsilon;",
                    "    model           realizableKE;",
                ),
            },
            // §41.2's inverse Prandtl number - the one that multiplies
            // `nu + nu_t` rather than `nu_t`, and needed the new affine
            // face-diffusivity kernel to be expressible at all.
            Knob {
                label: "RAS/alphaEps (RNGkEpsilon, SPEC-LIT 41.2)",
                file: "constant/momentumTransport",
                from: "    turbulence      on;",
                to: "    turbulence      on;\n    alphaEps        1.1;",
                pre: (
                    "constant/momentumTransport",
                    "    model           kEpsilon;",
                    "    model           RNGkEpsilon;",
                ),
            },
            Knob {
                label: "RAS/eta0 (RNGkEpsilon)",
                file: "constant/momentumTransport",
                from: "    turbulence      on;",
                to: "    turbulence      on;\n    eta0            3.0;",
                pre: (
                    "constant/momentumTransport",
                    "    model           kEpsilon;",
                    "    model           RNGkEpsilon;",
                ),
            },
            Knob {
                label: "RAS/beta (RNGkEpsilon)",
                file: "constant/momentumTransport",
                from: "    turbulence      on;",
                to: "    turbulence      on;\n    beta            0.05;",
                pre: (
                    "constant/momentumTransport",
                    "    model           kEpsilon;",
                    "    model           RNGkEpsilon;",
                ),
            },
            // SPEC-LIT 15.6: one constant, reaching both the model and the
            // wall functions.
            Knob {
                label: "RAS/Cmu",
                file: "constant/momentumTransport",
                from: "    turbulence      on;",
                to: "    turbulence      on;\n    Cmu             0.12;",
                pre: NO_PRE,
            },
            Knob {
                label: "RAS/kappa (wall functions)",
                file: "constant/momentumTransport",
                from: "    turbulence      on;",
                to: "    turbulence      on;\n    kappa           0.38;",
                pre: NO_PRE,
            },
            Knob {
                label: "RAS/turbulence off",
                file: "constant/momentumTransport",
                from: "    turbulence      on;",
                to: "    turbulence      off;",
                pre: NO_PRE,
            },
            Knob {
                label: "constant/physicalProperties nu",
                file: "constant/physicalProperties",
                from: "nu              [0 2 -1 0 0 0 0] 1e-05;",
                to: "nu              [0 2 -1 0 0 0 0] 1e-04;",
                pre: NO_PRE,
            },
            // A LOOSE tolerance on purpose: what is being demonstrated is
            // that the entry reaches the loop and stops it early, not that
            // twenty iterations of a channel converge. The `pre` drops the
            // check interval to 2, because at the default 25 the residual
            // test only ever runs on iteration 1, where `it > 1` blocks it -
            // the entry would then be untestable rather than unread.
            Knob {
                label: "SIMPLE/residualControl",
                file: "system/fvSolution",
                from: "    nNonOrthogonalCorrectors 0;",
                to: "    nNonOrthogonalCorrectors 0;\n    residualControl { k 0.5; epsilon 0.5; }",
                pre: NO_PRE,
            },
        ];

        let mut inert: Vec<&str> = Vec::new();
        for k in &cases {
            let a = run_knob(k, false, "a");
            let b = run_knob(k, true, "b");
            if a == b {
                inert.push(k.label);
            }
        }
        assert_none_inert(&inert);
    }


    // ======================================================================
    //  SPEC-LIT S44 - the `output` block, on the OTHER driver that reads it
    // ======================================================================
    //
    // `ofgpu-k-epsilon`'s own pair test above runs on an OpenFOAM case
    // DIRECTORY, which has no `output` block at all. This one runs on a
    // `.jsonc` case, because that is the only format that carries one - and
    // it is here, in the second driver, for exactly the reason S13.4.2 gives
    // for one shared refusal: a block honoured by `ofgpu-fire` and silently
    // ignored by the other driver that reads the same format is the defect
    // this whole subsection exists to prevent.

    /// A complete `ofgpu-k-epsilon` JSONC case: a small Cartesian duct with
    /// a frozen inlet-driven `U`, `k`, `epsilon` and `nut`. `output` is
    /// spliced in verbatim (with its leading comma) or empty.
    fn ke_json_case(output: &str) -> String {
        format!(
            r#"{{
  "name": "outputBlockTest",
  "mesh": {{
    "kind": "cartesian",
    "bounds": {{ "min": [0.0, 0.0, 0.0], "max": [0.4, 0.05, 0.05] }},
    "cells":  [10, 5, 3],
    "boundaries": {{
      "xmin": "inlet", "xmax": "outlet",
      "ymin": "wallLo", "ymax": "wallHi",
      "zmin": "sideLo", "zmax": "sideHi"
    }}
  }},
  "physics": {{
    "gravity": [0, 0, 0],
    "fluid": {{ "nu": 1.5e-5, "Pr": 0.71, "Prt": 0.85, "TRef": 293.15 }},
    "buoyancy": "densityRatio"
  }},
  "turbulence": {{
    "kind": "RAS",
    "model": "kEpsilon",
    "wallFunctions": {{ "kappa": 0.41, "E": 9.8 }},
    "wallTreatment": "standard"
  }},
  "patches": [
    {{
      "match": "inlet", "kind": "inlet",
      "U": {{ "type": "fixedValue", "value": [3.0, 0, 0] }},
      "p": {{ "type": "zeroGradient" }},
      "T": {{ "type": "fixedValue", "value": 293.15 }},
      "k": {{ "type": "fixedValue", "value": 0.03 }},
      "epsilon": {{ "type": "fixedValue", "value": 0.3 }},
      "nut": {{ "type": "calculated", "value": 0 }}
    }},
    {{
      "match": "outlet", "kind": "open",
      "U": {{ "type": "inletOutlet", "inletValue": [0, 0, 0] }},
      "p": {{ "type": "fixedValue", "value": 0.0 }},
      "T": {{ "type": "inletOutlet", "inletValue": 293.15 }},
      "k": {{ "type": "zeroGradient" }},
      "epsilon": {{ "type": "zeroGradient" }},
      "nut": {{ "type": "calculated", "value": 0 }}
    }},
    {{
      "match": ".*", "kind": "wall",
      "U": {{ "type": "fixedValue", "value": [0, 0, 0] }},
      "p": {{ "type": "zeroGradient" }},
      "T": {{ "type": "zeroGradient" }}
    }}
  ],
  "initial": {{
    "U": [3.0, 0, 0], "T": 293.15, "p": 0.0,
    "k": 0.03, "epsilon": 0.3, "nut": 0.0
  }},
  "numerics": {{
    "algorithm": {{ "kind": "SIMPLE" }},
    "ddt": "steadyState",
    "div": {{
      "default": "Gauss upwind",
      "div(phi,k)": "bounded Gauss upwind",
      "div(phi,epsilon)": "bounded Gauss upwind"
    }},
    "grad": "Gauss linear",
    "laplacian": {{ "snGrad": "corrected", "nonOrthogonalCorrectors": 0 }},
    "relaxation": {{ "k": 0.7, "epsilon": 0.7 }},
    "solvers": [
      {{ "match": ".*", "solver": "PBiCGStab", "preconditioner": "diagonal", "tolerance": 1e-9, "relTol": 0.01, "maxIter": 200 }}
    ]
  }},
  "run": {{ "endTime": 0.0, "deltaT": 0.001 }}{output}
}}"#
        )
    }

    /// Everything a JSONC run wrote, as `(relative path, BYTES)` - binary,
    /// for the reason `ofgpu-fire`'s own `written_bytes` gives: `.vdb` and
    /// `.nvdb` are not text, and a text walker skips them in silence.
    fn json_written_bytes(root: &Path) -> Vec<(String, Vec<u8>)> {
        fn walk(dir: &Path, prefix: &str, out: &mut Vec<(String, Vec<u8>)>) {
            let Ok(rd) = std::fs::read_dir(dir) else { return };
            for e in rd.flatten() {
                let p = e.path();
                let name = e.file_name().to_string_lossy().to_string();
                let rel = if prefix.is_empty() { name.clone() } else { format!("{prefix}/{name}") };
                if p.is_dir() {
                    walk(&p, &rel, out);
                } else if let Ok(b) = std::fs::read(&p) {
                    out.push((rel, b));
                }
            }
        }
        let mut out = Vec::new();
        walk(root, "", &mut out);
        out.sort();
        out
    }

    fn run_json_case(output: &str, tag: &str, extra: &[&str]) -> Result<Vec<(String, Vec<u8>)>> {
        let dir = scratch_dir(tag);
        let path = dir.join("case.jsonc");
        std::fs::write(&path, ke_json_case(output)).expect("write case");
        let mut args: Vec<String> =
            vec!["ofgpu-k-epsilon".to_string(), path.to_string_lossy().to_string()];
        args.extend(["-iters".to_string(), "6".to_string(), "-check".to_string(), "100".to_string()]);
        args.extend(extra.iter().map(|s| (*s).to_string()));
        let o = parse(&args)?;
        run(&o)?;
        Ok(json_written_bytes(&common::json_case_output_dir(&path)))
    }

    /// SPEC-LIT S44.7's pair table, for this driver: the four entries a
    /// STEADY single-write run can honour. `interval` is not among them
    /// because a steady run refuses it - which is the next test.
    #[test]
    fn every_output_setting_changes_what_the_run_writes() {
        if ofgpu::Gpu::new(0).is_err() {
            return;
        }
        let pairs: [(&str, &str, &str); 5] = [
            (
                "output (the block itself)",
                "",
                r#", "output": { "visualisation": { "format": "vdb" } }"#,
            ),
            (
                "output.visualisation.format",
                r#", "output": { "visualisation": { "format": "vdb" } }"#,
                r#", "output": { "visualisation": { "format": "nvdb" } }"#,
            ),
            (
                "output.visualisation.fields",
                r#", "output": { "visualisation": { "format": "vdb", "fields": ["k"] } }"#,
                r#", "output": { "visualisation": { "format": "vdb", "fields": ["k", "nut"] } }"#,
            ),
            (
                "output.visualisation.precision",
                r#", "output": { "visualisation": { "format": "vdb" } }"#,
                r#", "output": { "visualisation": { "format": "vdb", "precision": "fp16" } }"#,
            ),
            (
                "output.exact.format",
                r#", "output": { "exact": { "format": "vtu" } }"#,
                r#", "output": { "exact": { "format": "openfoam" } }"#,
            ),
        ];

        let mut inert: Vec<&str> = Vec::new();
        for (label, a, b) in pairs {
            let ra = run_json_case(a, "outa", &[]).expect("side a must run");
            let rb = run_json_case(b, "outb", &[]).expect("side b must run");
            if ra == rb {
                inert.push(label);
            }
        }
        assert_none_inert(&inert);
    }

    /// SPEC-LIT S44.1/S44.4/S44.6/S44.2 - the four things this driver
    /// refuses by name, each naming what to do instead.
    #[test]
    fn the_output_block_entries_this_driver_cannot_honour_are_refused_by_name() {
        if ofgpu::Gpu::new(0).is_err() {
            return;
        }

        // S44.4 - a steady run has no clock.
        let e = run_json_case(
            r#", "output": { "visualisation": { "format": "vdb", "interval": 2.0 } }"#,
            "interval",
            &[],
        )
        .expect_err("a steady driver must refuse a physical-time interval");
        let m = format!("{e}");
        assert!(m.contains("output.visualisation.interval"), "{m}");
        assert!(m.contains("steady"), "{m}");

        // S44.1 - and it writes no checkpoint at all.
        let e = run_json_case(
            r#", "output": { "restart": { "keep": 2 } }"#,
            "restart",
            &[],
        )
        .expect_err("a driver with no checkpoint must refuse output.restart");
        let m = format!("{e}");
        assert!(m.contains("output.restart"), "{m}");
        assert!(m.contains("ofgpu-fire"), "the error must name a driver that does: {m}");

        // S44.6 - the case and the command line both naming the output.
        let e = run_json_case(
            r#", "output": { "visualisation": { "format": "vdb" } }"#,
            "twice",
            &["-output", "vtu"],
        )
        .expect_err("naming the output twice must be refused");
        let m = format!("{e}");
        assert!(m.contains("output (case file)") && m.contains("-output"), "{m}");

        // S44.2 - a field this driver does not have. It solves k and
        // epsilon on a frozen U, so `U` itself is not one of its outputs -
        // which makes it the sharpest name to try.
        let e = run_json_case(
            r#", "output": { "visualisation": { "format": "vdb", "fields": ["k", "U"] } }"#,
            "nofield",
            &[],
        )
        .expect_err("a field this driver does not write must be refused");
        let m = format!("{e}");
        assert!(m.contains("\"U\""), "{m}");
        for have in ["k", "epsilon", "nut"] {
            assert!(m.contains(have), "the error must list {have}: {m}");
        }
    }

    /// SPEC-LIT 13.4 and 17. `constant/g` is read into `cc.buoyancy` by
    /// `read_case_controls` and was consulted by nothing here, while
    /// `KEpsilon::set_buoyancy` has existed all along: a case naming
    /// gravity ran with `G_b` identically zero and was not told. This driver
    /// reads no temperature, so the term cannot be built - the refusal names
    /// the drivers that can.
    #[test]
    fn a_case_that_names_gravity_is_refused_by_name() {
        let case = channel("grav");
        std::fs::write(
            case.join("constant").join("g"),
            "FoamFile\n{\n    version 2.0;\n    format ascii;\n    class uniformDimensionedVectorField;\n    location \"constant\";\n    object g;\n}\ndimensions [0 1 -2 0 0 0 0];\nvalue (0 0 -9.81);\n",
        )
        .expect("write constant/g");

        let cc = ofgpu::io::case::read_case_controls(&case).expect("controls");
        assert!(cc.buoyancy.is_active(), "the knob must actually put gravity in the case");

        let e = common::refuse_buoyancy_without_temperature(&case, &cc, None, "ofgpu-k-epsilon")
            .expect_err("a case naming gravity must be refused");
        let msg = format!("{e}");
        assert!(msg.contains("gravity"), "the error must name the setting: {msg}");
        assert!(msg.contains("ofgpu-plume"), "the error must name an alternative: {msg}");
        assert!(msg.contains("9.81"), "the error must quote what the case said: {msg}");
    }

    /// SPEC-LIT 13.4, found BY the pair test above rather than by the audit
    /// that prompted it.
    ///
    /// `nNonOrthogonalCorrectors 2` used to be read into
    /// `TurbulenceControls::n_non_orth_correctors`, printed by
    /// `print_effective_settings`, and looped over by nothing: no turbulence
    /// model in this crate carries the `for _pass in 0..=..` that
    /// `energy.rs`, `momentum.rs`, `scalar_transport.rs` and `simple.rs` each
    /// do. Two runs of this driver differing only in it wrote bit-identical
    /// fields - the definition of inert - and this driver has no other
    /// equation for it to reach, so it is refused by name.
    #[test]
    fn a_non_orthogonal_corrector_count_that_reaches_no_equation_is_a_named_error() {
        let case = channel("nonorth");
        apply(
            &case,
            &Knob {
                label: "SIMPLE/nNonOrthogonalCorrectors",
                file: "system/fvSolution",
                from: "    nNonOrthogonalCorrectors 0;",
                to: "    nNonOrthogonalCorrectors 2;",
                pre: NO_PRE,
            },
            true,
        );

        let cc = ofgpu::io::case::read_case_controls(&case).expect("controls");
        assert_eq!(cc.turb.n_non_orth_correctors, 2, "the knob must reach the controls");

        let e = common::refuse_non_orth_correctors_without_another_equation(&cc, "DRIVER")
            .expect_err("a corrector count that reaches no equation must be refused");
        let msg = format!("{e}");
        assert!(msg.contains("nNonOrthogonalCorrectors"), "must name the setting: {msg}");
        assert!(msg.contains("ofgpu-plume"), "must name where it IS honoured: {msg}");

        // Zero - what every case in this tree writes - is silent.
        let mut zero = cc;
        zero.turb.n_non_orth_correctors = 0;
        assert!(common::refuse_non_orth_correctors_without_another_equation(&zero, "D").is_ok());
    }
}
