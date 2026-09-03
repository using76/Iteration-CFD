// meteor-cfd - Copyright (c) 2026 주식회사 메테오시뮬레이션 (Meteo Simulation Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

//! `ofgpu-sa` - run Spalart-Allmaras, or one of its detached-eddy hybrids,
//! on a case directory. SPEC-LIT §56, §57, §58.
//!
//! ```text
//! ofgpu-sa <caseDir> [options]
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
//!   -output LIST    foam|vtu|nvdb|vdb|usda, comma separated
//! ```
//!
//! **Why this driver exists at all.** `models::registry::build_coupled` can
//! build Spalart-Allmaras, but the two drivers that call it -
//! `ofgpu-buoyant` and `ofgpu-lowmach` - both solve a temperature under gravity,
//! and SPEC-LIT §56.8 refuses SA there by name: §17's buoyancy production
//! `G_b` enters a `k` equation and this model has none. Without an isothermal
//! driver, SA would be a model the registry could select and no binary could
//! run - a capability that stops at the case reader. This is that driver, and
//! it is `ofgpu-k-omega` with one transport equation instead of two.
//!
//! The velocity field is held frozen: this program solves the `nu~` equation
//! on a given `U` and `phi`. After the mesh and the initial fields are
//! uploaded, no field data crosses the PCIe bus again until the results are
//! written.
//!
//! **The hybrids run here too**, and the four guards of SPEC-LIT §57.10 are
//! what make that honest: a steady run, a 2-D mesh, an upwind-biased
//! `div(phi,U)` and `cubeRootVol` as the filter width are each refused by
//! name before anything is built. A DES that runs but has never had anything
//! to resolve is worse than a refusal.
//!
//! Provenance: ORIGINAL driver code over LITERATURE numerics. The model
//! itself is cited in `src/models/spalart_allmaras.rs`,
//! `src/models/des.rs` and SPEC-LIT.md §56/§57; this file is argument
//! parsing, case loading, the iteration order and the reporting loop, which
//! are this project's own (`PROVENANCE.md`, `src/bin/*`), kept deliberately
//! line for line alongside `k_omega.rs`.
//! No GPL-licensed source was consulted.

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use ofgpu::field::{GpuSurfaceScalarField, GpuVectorField};
use ofgpu::field_ops::{correct_boundary_conditions_vector, FieldKernels};
use ofgpu::field_setup::{
    compute_phi_from_u, harvest_scalar_field, harvest_surface_scalar_field, max_div_phi,
    setup_scalar_field, setup_vector_field, update_inlet_outlet, wall_coeffs_from_case,
    NutRoughness, WallFaces,
};
use ofgpu::io::case::{find_start_time, read_case_controls};
use ofgpu::io::fields::{read_scalar_field, read_vector_field, RawScalarField};
use ofgpu::io::polymesh::{build_host_mesh, read_poly_mesh};
use ofgpu::models::des::DesLengthScale;
use ofgpu::models::{
    refuse_two_dimensional_hybrid, sa_coeffs, select_turbulence_model, RasModel,
    SpalartAllmaras,
};
use ofgpu::turbulence::FlowState;
use ofgpu::{Error, GpuMesh, Label, Result, Scalar};

#[path = "common/mod.rs"]
mod common;

use common::{
    atoi, build_writers, device_banner, g, next_arg, parse_output_formats, sci, OutputFormat,
};

/// Everything the command line can change.
struct Options {
    case_dir: PathBuf,
    n_iters: Label,
    fixed_iters: Label,
    check_every: Label,
    do_write: bool,
    write_time: String,
    permissive: bool,
    output: Vec<OutputFormat>,
}

fn usage() {
    eprintln!(
        "usage: ofgpu-sa <caseDir> [-iters N] [-fixedIters N] [-write NAME] \
         [-noWrite] [-check N] [-permissive] [-output LIST]"
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
/// from `U` when it is not - `k_omega.rs`'s own, unchanged, because a flux
/// written by a real solver satisfies discrete continuity and nothing
/// reconstructed from a cell-centred `U` can be relied on to.
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

#[allow(clippy::too_many_lines)]
fn run(o: &Options) -> Result<()> {
    // ---- device -----------------------------------------------------------
    let gpu = ofgpu::Gpu::new(0)?;
    println!("{}", device_banner(&gpu, "Spalart-Allmaras")?);

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
        cc.turb.k_solver.fixed_iters = true;
        cc.turb.k_solver.max_iter = o.fixed_iters;
        cc.turb.k_solver.report_residuals = false;
        cc.turb.epsilon_solver.fixed_iters = true;
        cc.turb.epsilon_solver.max_iter = o.fixed_iters;
        cc.turb.epsilon_solver.report_residuals = false;
    }

    ofgpu::io::case::print_effective_settings(&cc);

    // This driver reads no temperature, so §17's `G_b` cannot be built here
    // at all - which is exactly why SA is reachable from it and not from the
    // coupled drivers (SPEC-LIT §56.8).
    common::refuse_buoyancy_without_temperature(&o.case_dir, &cc, None, "ofgpu-sa")?;
    common::refuse_non_orth_correctors_without_another_equation(&cc, "ofgpu-sa")?;
    common::refuse_rheology_without_momentum(&cc, "ofgpu-sa")?;

    let selection = select_turbulence_model(&cc)?;
    match selection.model {
        RasModel::SpalartAllmaras | RasModel::HybridSa | RasModel::Laminar => {}
        other => {
            return Err(Error::Config(format!(
                "this case asks for the {} model; run it with {} \
                 (ofgpu-sa builds SpalartAllmaras and its SA-background hybrids)",
                other.name(),
                common::driver_for(other),
            )));
        }
    }

    // SPEC-LIT §57.10's guard 3, the one that needs a mesh.
    if let Some(h) = &selection.des {
        refuse_two_dimensional_hybrid(&hm, h)?;
    }

    if !selection.active {
        println!(
            "turbulence is off in this case: nu_t is frozen at zero and the \
             model will not be corrected"
        );
    }

    // The hybrid's own constants live in its block; a plain RANS run reads
    // `RAS { ... }`.
    let coeffs = match &selection.des {
        Some(h) => h.sa,
        None => sa_coeffs(&cc)?,
    };

    println!(
        "nu = {} | model {} | variant {} | Cb1 {} Cb2 {} Cv1 {} Cn1 {} \
         sigmaNut {} kappa {} | c_w1 = Cb1/kappa^2 + (1+Cb2)/sigmaNut = {} \
         (DERIVED, SPEC-LIT (56.6))",
        g(f64::from(cc.nu)),
        selection.describe(),
        coeffs.variant.name(),
        g(f64::from(coeffs.cb1)),
        g(f64::from(coeffs.cb2)),
        g(f64::from(coeffs.cv1)),
        g(f64::from(coeffs.cn1)),
        g(f64::from(coeffs.sigma)),
        g(f64::from(coeffs.kappa)),
        g(f64::from(coeffs.cw1())),
    );
    if selection.des.is_some() {
        println!(
            "NOTE: the low-Reynolds correction Psi of Shur et al. (2008) is \
             NOT implemented - neither open-access restatement read carries it \
             (SPEC-LIT 57.5)"
        );
    }

    // ---- fields -----------------------------------------------------------
    let t = find_start_time(&o.case_dir)?;
    let t_dir = o.case_dir.join(&t);

    let raw_u = read_vector_field(&t_dir.join("U"), hm.n_cells)?;
    let raw_nt = read_scalar_field(&t_dir.join("nuTilda"), hm.n_cells)?;

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

    // SPEC-LIT §15.5: `nut`'s wall treatment comes from `nut`'s own patch
    // types. `nuTilda` pins no wall CELL at all - `nu~ = 0` is an exact
    // Dirichlet condition (§56.7) - so `WallFaces::constrained_cells` is
    // built from it and is empty, which is the correct answer rather than an
    // omission.
    let nut_path = t_dir.join("nut");
    let raw_nut = if nut_path.exists() {
        Some(read_scalar_field(&nut_path, hm.n_cells)?)
    } else {
        None
    };
    let wf_faces = WallFaces::from_case(&raw_nt, raw_nut.as_ref(), &hm)?;
    let roughness = NutRoughness::from_case(raw_nut.as_ref(), &hm)?;

    let flow = FlowState::new(&u, &phi, cc.nu);

    // SPEC-LIT §6.6: the wall distance, once, at setup. Both the model and
    // (57.19)'s `h_wn` stand on it.
    let wd = ofgpu::walldistance::wall_distance(
        &gpu,
        &hm,
        &mesh,
        &cc.p_solver,
        cc.turb.n_non_orth_correctors,
    )?;
    println!(
        "wall distance: {} wall faces, {} iterations, residual {}, max y = {}",
        wd.n_wall_faces,
        wd.iterations,
        sci(f64::from(wd.final_residual), 3),
        g(f64::from(wd.max(&gpu)?))
    );

    let mut model = SpalartAllmaras::new(
        &gpu,
        &hm,
        &mesh,
        coeffs,
        cc.turb,
        wall_coeffs_from_case(&cc.wall),
        &wf_faces,
        &roughness,
        &wd.y.f,
    )?;

    if let Some(h) = &selection.des {
        model.set_des(Some(DesLengthScale::new(
            &gpu,
            &mesh,
            &wd.y.f,
            &wd.grad_y,
            h.branch,
            h.delta,
            h.background,
            h.coeffs,
        )?));
    }

    setup_scalar_field(&gpu, model.nu_tilda_mut(), &raw_nt, &hm)?;
    if let Some(raw_nut) = &raw_nut {
        setup_scalar_field(&gpu, model.nut_mut(), raw_nut, &hm)?;
    }
    update_inlet_outlet(&gpu, model.nu_tilda_mut(), &phi, &hm)?;

    if selection.active {
        model.initialise(&gpu, &flow)?;
    } else {
        model.freeze_nut(&gpu)?;
    }

    // ---- time loop - device only from here to the write -------------------
    println!(
        "\niterating {} times, relax nuTilda {}{}",
        cc.turb.n_outer_iterations,
        g(f64::from(cc.turb.eps_relax)),
        if cc.turb.epsilon_solver.fixed_iters {
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
            done = 1;
            break;
        }
        // SA has ONE equation, so `correct` returns the same performance
        // record twice rather than inventing a second solve - see its own
        // doc. Only one column is printed here, which is what a one-equation
        // model has to report.
        let (perf, _) = model.correct(&gpu, &flow)?;
        done = it;

        if it % cc.turb.convergence_check_every == 0 || it == 1 {
            let change = model.convergence_measure(&gpu)?;
            println!(
                "{it:>7}  nuTilda res {} ({})  max dnuTilda/nuTilda {}",
                sci(f64::from(perf.initial_residual), 3),
                perf.n_iterations,
                sci(f64::from(change), 3)
            );

            if !cc.residual_control.is_empty() {
                if it > 1
                    && cc
                        .residual_control
                        .all_satisfied(&[("nuTilda", perf.initial_residual)])
                {
                    println!("converged: every residualControl entry met");
                    break;
                }
            } else if it > 1 && change < cc.turb.convergence_tol {
                println!(
                    "converged: max relative change below {}",
                    g(f64::from(cc.turb.convergence_tol))
                );
                break;
            }
        }
    }

    gpu.sync()?;
    let wall = t_loop.elapsed().as_secs_f64();
    let n = done.max(1) as f64;

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

        let mut out_nt = raw_nt.types_only();
        let mut out_nut = match &raw_nut {
            Some(raw) => raw.types_only(),
            None => RawScalarField::default(),
        };

        harvest_scalar_field(&gpu, &mut out_nt, model.nu_tilda(), &hm)?;
        harvest_scalar_field(&gpu, &mut out_nut, model.nut(), &hm)?;

        let mut out_phi = RawScalarField::default();
        harvest_surface_scalar_field(&gpu, &mut out_phi, &phi, &hm)?;

        out_nt.dimensions = "[0 2 -1 0 0 0 0]".to_string();
        out_nut.dimensions = "[0 2 -1 0 0 0 0]".to_string();

        let foam_fields = [
            ofgpu::io::FoamField::scalar("nuTilda", &out_nt),
            ofgpu::io::FoamField::scalar("nut", &out_nut),
            ofgpu::io::FoamField::surface("phi", &out_phi),
        ];
        let vis_fields = [
            ofgpu::io::OutputField::scalar("nuTilda", &out_nt.internal),
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
        let mut writers = build_writers(
            &o.case_dir,
            "SpalartAllmaras",
            &o.output,
            ofgpu::io::nvdb::Precision::F32,
        )?;
        for w in &mut writers {
            w.write_step(&ctx)?;
        }

        println!("written to {}", o.case_dir.join(&wt).display());
    }

    Ok(())
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
// Named `sa_tests` rather than `tests` because `common/mod.rs` is included by
// `#[path]` and already contributes a `tests` module to this crate.

#[cfg(test)]
mod sa_tests {
    use super::*;
    use common::knobs::{apply, assert_none_inert, scratch_dir, written_state, Knob, NO_PRE};
    use ofgpu::blockgen::{write_case, CaseKind};

    fn argv(v: &[&str]) -> Vec<String> {
        std::iter::once("ofgpu-sa".to_string())
            .chain(v.iter().map(|s| (*s).to_string()))
            .collect()
    }

    fn channel(tag: &str) -> PathBuf {
        let dir = scratch_dir(tag);
        let case = dir.join("case");
        write_case(&case, CaseKind::Channel, 16, 10, 1).expect("generate the channel case");
        // `write_case` writes `RAS { model kEpsilon; }` for every case kind;
        // this driver builds SpalartAllmaras and refuses anything else by
        // name. Everything else the model needs - `0/nuTilda` with §56.7's
        // own boundary table, `div(phi,nuTilda)`, `solvers/nuTilda` and its
        // relaxation factor - the generator now writes, which is what makes
        // `ofgpu-generate-mesh` and `ofgpu-sa` compose (SPEC-LIT §58.1).
        apply(
            &case,
            &Knob {
                label: "RAS/model",
                file: "constant/momentumTransport",
                from: "    model           kEpsilon;",
                to: "    model           SpalartAllmaras;",
                pre: NO_PRE,
            },
            true,
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

        let out = written_state(&case.join("1"));
        assert!(!out.is_empty(), "the run wrote nothing to compare");
        out
    }

    /// **The standing test SPEC-LIT §13.4.1 requires of every setting this
    /// driver claims to honour** - and the one that would have caught
    /// §58.1's seventh instance.
    ///
    /// `div(phi,nuTilda)`, `solvers/nuTilda` and
    /// `relaxationFactors/equations/nuTilda` all reach the `nu~` equation
    /// through `io::case::dissipation_from_model`, which answered `None` for
    /// `SpalartAllmaras` before §56 and let the reader fall back to
    /// "whichever entry the case happened to write" - `epsilon`. Every one of
    /// the three knobs below would then have been inert.
    #[test]
    fn every_wired_setting_changes_what_the_run_writes() {
        if ofgpu::Gpu::new(0).is_err() {
            return;
        }

        let cases: Vec<Knob> = vec![
            Knob {
                label: "divSchemes/div(phi,nuTilda)",
                file: "system/fvSchemes",
                from: "div(phi,nuTilda) bounded Gauss upwind;",
                to: "div(phi,nuTilda) Gauss linear;",
                pre: NO_PRE,
            },
            Knob {
                label: "relaxationFactors/equations/nuTilda",
                file: "system/fvSolution",
                from: "        nuTilda         0.7;",
                to: "        nuTilda         0.2;",
                pre: NO_PRE,
            },
            Knob {
                label: "solvers/nuTilda/maxIter",
                file: "system/fvSolution",
                from: "    nuTilda
    {
        solver          PBiCGStab;
        preconditioner  diagonal;
        tolerance       1e-08;
        relTol          0.01;
        maxIter         200;",
                to: "    nuTilda
    {
        solver          PBiCGStab;
        preconditioner  diagonal;
        tolerance       1e-08;
        relTol          0.01;
        maxIter         1;",
                pre: NO_PRE,
            },
            Knob {
                label: "RAS/variant",
                file: "constant/momentumTransport",
                from: "    model           SpalartAllmaras;",
                to: "    model           SpalartAllmaras;\n    variant         ft2;",
                pre: NO_PRE,
            },
            Knob {
                label: "RAS/Cb1",
                file: "constant/momentumTransport",
                from: "    model           SpalartAllmaras;",
                to: "    model           SpalartAllmaras;\n    Cb1             0.14;",
                pre: NO_PRE,
            },
            Knob {
                label: "RAS/sigmaNut",
                file: "constant/momentumTransport",
                from: "    model           SpalartAllmaras;",
                to: "    model           SpalartAllmaras;\n    sigmaNut        0.9;",
                pre: NO_PRE,
            },
            Knob {
                label: "RAS/Cv1",
                file: "constant/momentumTransport",
                from: "    model           SpalartAllmaras;",
                to: "    model           SpalartAllmaras;\n    Cv1             9.0;",
                pre: NO_PRE,
            },
            Knob {
                label: "RAS/turbulence",
                file: "constant/momentumTransport",
                from: "    turbulence      on;",
                to: "    turbulence      off;",
                pre: NO_PRE,
            },
        ];

        let mut inert = Vec::new();
        for (i, k) in cases.iter().enumerate() {
            let a = run_knob(k, false, &format!("sa_off_{i}"));
            let b = run_knob(k, true, &format!("sa_on_{i}"));
            if a == b {
                inert.push(k.label);
            } else {
                let which: Vec<&str> = a
                    .iter()
                    .zip(b.iter())
                    .filter(|(x, y)| x != y)
                    .map(|(x, _)| x.0.as_str())
                    .collect();
                println!("{}: moved {which:?}", k.label);
            }
        }
        assert_none_inert(&inert);
    }

    /// A model this driver does not build is refused by name, and the
    /// refusal points at a binary that exists.
    #[test]
    fn a_model_this_driver_does_not_build_is_refused_by_name() {
        if ofgpu::Gpu::new(0).is_err() {
            return;
        }
        let dir = scratch_dir("sa_wrong_model");
        let case = dir.join("case");
        write_case(&case, CaseKind::Channel, 8, 6, 1).expect("generate");
        // The generator leaves `RAS { model kEpsilon; }`.
        let args = argv(&[case.to_string_lossy().as_ref(), "-iters", "1", "-noWrite"]);
        let o = parse(&args).expect("parse");
        let e = run(&o).expect_err("kEpsilon must be refused by ofgpu-sa");
        let t = e.to_string();
        assert!(t.contains("kEpsilon"), "{t}");
        assert!(t.contains("ofgpu-k-epsilon"), "the refusal names no driver: {t}");
    }

    /// **SPEC-LIT §57.10's guards, at the driver.** The `channel` case is
    /// 2-D and steady, so a hybrid on it must be refused - twice over - and
    /// the message must say which.
    #[test]
    fn a_hybrid_on_the_two_dimensional_steady_channel_is_refused() {
        if ofgpu::Gpu::new(0).is_err() {
            return;
        }
        let dir = scratch_dir("sa_hybrid_guard");
        let case = dir.join("case");
        write_case(&case, CaseKind::Channel, 8, 6, 1).expect("generate");
        std::fs::write(
            case.join("constant").join("momentumTransport"),
            "FoamFile { version 2.0; format ascii; class dictionary; \
             object momentumTransport; }\nsimulationType LES;\n\
             LES { model SpalartAllmarasDDES; }\n",
        )
        .expect("write momentumTransport");

        let args = argv(&[case.to_string_lossy().as_ref(), "-iters", "1", "-noWrite"]);
        let o = parse(&args).expect("parse");
        let e = run(&o).expect_err("a steady 2-D hybrid must be refused");
        let t = e.to_string();
        // The steady guard fires first, before the mesh is consulted.
        assert!(
            t.contains("ddtSchemes") || t.contains("2-D"),
            "the refusal names neither guard: {t}"
        );
    }
}
