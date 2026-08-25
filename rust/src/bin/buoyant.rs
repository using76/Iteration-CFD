// meteor-cfd - Copyright (c) 2026 주식회사 메테오시뮬레이션 (Meteo Simulation Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

//! `ofgpu-buoyant` - the plume, driven by its own buoyancy.
//!
//! ```text
//! ofgpu-buoyant <caseDir> [options]
//!
//!   -iters N          outer iterations (default: from controlDict endTime)
//!   -fixedIters N     run every linear solver for exactly N sweeps and never
//!                     read a residual, so the time loop performs ZERO host
//!                     transfers of any kind
//!   -check N          test the convergence measure every N iterations
//!   -write NAME       time directory to write into (default: from controlDict)
//!   -noWrite          do not write fields; useful when timing
//!   -graph            capture ONE unit of work as a CUDA graph and replay it;
//!                     requires -fixedIters and -backend pbicgstab
//!   -noPotential      do not solve for the starting flux; start from
//!                     interpolate(U) & Sf instead
//!   -inletPatch NAME  patch the flow enters through (default: inlet)
//!   -outletPatch NAME patch it leaves through (default: outlet)
//!
//!   -nCorrectors N    PIMPLE outer correctors per unit of work (default 2)
//!   -backend NAME     auto | pbicgstab | fft | amgx (default auto, which runs
//!                     the selector and prints its table)
//!   -probe            print the mass-weighted mean T of the upper and the
//!                     lower third of the domain at every write - the
//!                     stratification a working buoyancy MUST produce
//!
//!   transient mode - additions. Without -endTime none of them is read and the
//!   driver runs SIMPLE to steady state instead:
//!
//!   -endTime T        physical end time in seconds
//!   -deltaT dt        time step in seconds (default: controlDict deltaT)
//!   -writeInterval W  write every W seconds of PHYSICAL time
//!   -outerIters N     passes of [SIMPLE x nCorrectors, k-epsilon, T] per step
//!                     (default 1 - see below)
//! ```
//!
//! # What this is, next to `ofgpu-plume`
//!
//! `ofgpu-plume` holds the velocity field FROZEN. It transports a temperature
//! on a flux somebody else computed, and however hot that temperature gets it
//! can never push the flow. The result is a picture of a plume with none of a
//! plume's physics in it: no updraught, no entrainment, no ceiling jet, and no
//! stratification - the hot gas goes wherever the frozen flux was already
//! going.
//!
//! This driver solves for the velocity. One unit of work is
//!
//! ```text
//! nCorrectors x SIMPLE      momentum predictor -> pressure -> flux correction
//! k-epsilon    correct      on the velocity SIMPLE just produced
//! T            correct      on the flux SIMPLE just made conservative
//! ```
//!
//! and the coupling that makes it a plume closes through the buoyancy: `T`
//! feeds `b = g*(TRef/T - 1)` in the momentum equation, the momentum equation
//! feeds `phi`, and `phi` carries `T`. See `BUOYANT.md` section 2 for why the
//! body force is the ideal-gas ratio and not a Boussinesq expansion - over
//! this case's 293 K to 1173 K, `dT/T` is 3.0 and Boussinesq is wrong by a
//! factor of three.
//!
//! Everything else - the flags, the write intervals, the timing block - is
//! `ofgpu-plume`'s, deliberately, so the two runs are directly comparable.
//!
//! # `-nCorrectors` and `-outerIters` are different knobs
//!
//! `-nCorrectors` is PIMPLE's OUTER corrector count (SPEC-LIT 14): how many
//! times the momentum-pressure system is re-linearised inside one time step,
//! with relaxation switched off on the last one so the step ends on the
//! unrelaxed equations. The pressure correctors *inside* each of those - PISO's
//! `nCorrectors`, where `H` is re-evaluated between solves - come from
//! `fvSolution`.
//!
//! It does **not** advance the velocity by `N` sub-steps. It used to:
//! `Simple::correct` refreshed `U^{n-1}` on entry, so `-nCorrectors 2` ran two
//! Euler sub-steps of `U` per step while `T` and `k`/`epsilon` ran one, and the
//! fields drifted apart in time with nothing in the output saying so. The
//! old-time level is now stored once per TIME STEP, by
//! `Simple::begin_time_step`, and `Simple::correct_outer` refreshes it only in
//! a steady run - where `1/dt` is zero and it is never read.
//!
//! `-outerIters` repeats the WHOLE unit - turbulence and temperature included.
//! It defaults to 1 here where `ofgpu-plume` defaults to 2, and the reason is
//! `store_old_time`: `KEpsilon::correct` and `ScalarTransport::correct` each
//! refresh their own old-time level on entry, so a second pass differences
//! against the first pass rather than against the start of the step and
//! advances `k`, `epsilon` and `T` by a second `dt`. With `-outerIters 1` a
//! step is exactly one Euler implicit step of every transported field, which
//! is what a transient plume needs.
//!
//! # Why the backend is chosen by measurement
//!
//! The pressure equation is the only one here where the linear solver matters
//! at all; `BUOYANT.md` section 5 says why, and
//! [`ofgpu::pressure::choose_pressure_backend`] is what acts on it. `-backend
//! auto` assembles the real pressure matrix, probes it for structural facts,
//! runs every applicable backend on it, checks each answer against a tight
//! PBiCGStab reference, and keeps the fastest that agreed. The whole table is
//! printed. `-backend NAME` skips the measurement and takes the named one -
//! and still refuses it if the probe says it cannot represent this system,
//! because applicability is a hard constraint and not a preference.

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use ofgpu::field::{GpuSurfaceScalarField, GpuVectorField};
use ofgpu::field_ops::{correct_boundary_conditions_vector, FieldKernels};
use ofgpu::field_setup::{
    compute_phi_from_u, harvest_scalar_field, harvest_surface_scalar_field,
    harvest_vector_field, max_div_phi,
    setup_scalar_field, setup_vector_field, update_inlet_outlet, wall_coeffs_from_case,
    NutRoughness, WallFaces,
};
use ofgpu::io::case::{
    find_start_time, format_time_name, model_coeff, read_case_controls, CaseControls,
    SolverControls, TurbulenceControls,
};
use ofgpu::io::dict::FoamDict;
use ofgpu::io::fields::{
    read_scalar_field, read_vector_field, RawScalarField, RawVectorField,
};
use ofgpu::io::polymesh::{build_host_mesh, read_poly_mesh};
use ofgpu::models::{KEpsilon, KEpsilonCoeffs};
use ofgpu::turbulence::{BuoyancyProduction, C3Mode};
use ofgpu::momentum::{BuoyancyCoeffs, MomentumControls};
use ofgpu::potential_flow::{
    mean_inflow_speed, solve_potential_flow, PotentialFlowResult, PotentialFlowSpec,
};
use ofgpu::pressure::{
    choose_pressure_backend, AmgxBackend, FftBackend, PbicgstabBackend, PressureBackend,
    SystemProbe,
};
use ofgpu::scalar_transport::{weighted_stats, ScalarTransport, ScalarTransportCoeffs};
use ofgpu::simple::{Simple, SimpleControls};
use ofgpu::{Error, Gpu, GpuMesh, Graph, HostMesh, Label, Result, Scalar, Vec3};

#[path = "common/mod.rs"]
mod common;

use common::{
    atoi, build_writers, device_banner, find_restart_field, from_restart_scalars,
    from_restart_vectors, g, mean, next_arg, parse_output_formats, restart_scalar, restart_shell,
    restart_surface, restart_vector, sci, OutputFormat,
};
use ofgpu::restart::{self, RestartData};

/// Units of work run the ordinary way before a graph is captured.
///
/// The first call of each kernel pays for module loading and the first solve
/// pays for the preconditioner setup; capturing either would bake a one-off
/// into a recording that is replayed thousands of times.
const GRAPH_WARMUP: i64 = 5;

/// SIMPLE iterations per unit of work when the command line does not say.
const DEFAULT_N_CORRECTORS: Label = 2;

/// Passes of the whole unit per time step when the command line does not say.
/// One, so a step is one Euler implicit step - see the module header.
const DEFAULT_OUTER_ITERS: Label = 1;

// ==========================================================================
//  Command line
// ==========================================================================

/// Which pressure backend the user asked for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BackendPick {
    /// Probe, measure, and print the table. The default, because the crossover
    /// between a direct and an iterative Poisson solve depends on the mesh and
    /// the card and any hardcoded answer would be wrong somewhere.
    Auto,
    Pbicgstab,
    Fft,
    Amgx,
}

impl BackendPick {
    fn from_name(s: &str) -> Result<Self> {
        match s.to_ascii_lowercase().as_str() {
            "auto" => Ok(Self::Auto),
            "pbicgstab" | "pbicg" => Ok(Self::Pbicgstab),
            "fft" | "cufft" => Ok(Self::Fft),
            "amgx" => Ok(Self::Amgx),
            other => Err(Error::Config(format!(
                "-backend takes auto|pbicgstab|fft|amgx, got \"{other}\""
            ))),
        }
    }
}

/// Everything the command line can change.
struct Options {
    case_dir: PathBuf,
    n_iters: Label,
    fixed_iters: Label,
    check_every: Label,
    do_write: bool,
    write_time: String,
    graph: bool,

    /// Physical end time. Non-positive means the flag was absent, and that is
    /// what selects the steady path.
    end_time: f64,
    /// Non-positive means "not given"; controlDict's `deltaT` then stands.
    delta_t: f64,
    /// Non-positive means "not given"; one write at the end time then stands.
    write_interval: f64,
    outer_iters: Label,
    n_correctors: Label,

    backend: BackendPick,
    probe_layers: bool,

    /// Start from `interpolate(U) & Sf` rather than solving for a conservative
    /// flux. The first pressure correction fixes either one; this only changes
    /// how much work it has to do.
    no_potential: bool,
    inlet_patch: String,
    outlet_patch: String,
    /// `-output foam|vtu|nvdb|vdb|usda`, comma list.
    output: Vec<OutputFormat>,
    /// `-restartWrite N` - write a `.mcr` checkpoint every N STEPS.
    restart_write: Option<u64>,
    /// `-restartFrom FILE` - load state from a checkpoint, skipping the
    /// potential-flow / interpolated-flux starting-phi fallback.
    restart_from: Option<PathBuf>,
}

fn usage() {
    eprintln!(
        "usage: ofgpu-buoyant <caseDir> [-iters N] [-fixedIters N] [-check N] \
         [-write NAME] [-noWrite] [-graph]\n       \
         [-endTime T] [-deltaT dt] [-writeInterval W] [-outerIters N] \
         [-nCorrectors N]\n       \
         [-backend auto|pbicgstab|fft|amgx] [-probe]\n       \
         [-noPotential] [-inletPatch NAME] [-outletPatch NAME]\n       \
         [-output LIST] [-restartWrite N] [-restartFrom FILE]
                [-permissive]"
    );
}

/// A physical time from the command line.
///
/// Deliberately stricter than [`atoi`], which every other flag uses: `-deltaT
/// 0.001` read by an integer parser is zero, and a zero timestep does not fail.
/// It makes `1/dt` infinite and fills every matrix with NaNs several hundred
/// launches away from the mistake.
fn parse_time(flag: &str, s: &str) -> Result<f64> {
    match s.trim().parse::<f64>() {
        Ok(v) if v.is_finite() => Ok(v),
        _ => Err(Error::Config(format!(
            "{flag} needs a finite number of seconds, got \"{s}\""
        ))),
    }
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
        graph: false,
        end_time: -1.0,
        delta_t: -1.0,
        write_interval: -1.0,
        outer_iters: DEFAULT_OUTER_ITERS,
        n_correctors: DEFAULT_N_CORRECTORS,
        backend: BackendPick::Auto,
        probe_layers: false,
        no_potential: false,
        inlet_patch: "inlet".to_string(),
        outlet_patch: "outlet".to_string(),
        output: vec![OutputFormat::Foam],
        restart_write: None,
        restart_from: None,
    };

    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "-iters" => o.n_iters = atoi(&next_arg(args, &mut i)?) as Label,
            "-fixedIters" => o.fixed_iters = atoi(&next_arg(args, &mut i)?) as Label,
            "-check" => o.check_every = atoi(&next_arg(args, &mut i)?) as Label,
            "-write" => o.write_time = next_arg(args, &mut i)?,
            "-noWrite" => o.do_write = false,
            // SPEC-LIT 13.4's escape hatch. Set before any dictionary is
            // opened, so every rejection this run makes sees the same policy.
            "-permissive" => ofgpu::io::contract::set_permissive(true),
            "-graph" => o.graph = true,
            "-endTime" => o.end_time = parse_time("-endTime", &next_arg(args, &mut i)?)?,
            "-deltaT" => o.delta_t = parse_time("-deltaT", &next_arg(args, &mut i)?)?,
            "-writeInterval" => {
                o.write_interval = parse_time("-writeInterval", &next_arg(args, &mut i)?)?;
            }
            "-outerIters" => o.outer_iters = atoi(&next_arg(args, &mut i)?) as Label,
            "-nCorrectors" => o.n_correctors = atoi(&next_arg(args, &mut i)?) as Label,
            "-backend" => o.backend = BackendPick::from_name(&next_arg(args, &mut i)?)?,
            "-probe" => o.probe_layers = true,
            "-noPotential" => o.no_potential = true,
            "-inletPatch" => o.inlet_patch = next_arg(args, &mut i)?,
            "-outletPatch" => o.outlet_patch = next_arg(args, &mut i)?,
            "-output" => o.output = parse_output_formats(&next_arg(args, &mut i)?)?,
            "-restartWrite" => {
                let n = atoi(&next_arg(args, &mut i)?);
                if n <= 0 {
                    return Err(Error::Config("-restartWrite needs a positive step count".into()));
                }
                o.restart_write = Some(n as u64);
            }
            "-restartFrom" => o.restart_from = Some(PathBuf::from(next_arg(args, &mut i)?)),
            other => {
                usage();
                return Err(Error::Config(format!("unknown option {other}")));
            }
        }
        i += 1;
    }

    // Caught here rather than left to the driver: a capture containing a
    // read-back fails somewhere inside cudarc with a message about stream
    // capture state, which tells the user nothing about what they typed.
    if o.graph && o.fixed_iters <= 0 {
        usage();
        return Err(Error::Config(
            "-graph needs -fixedIters N: a CUDA graph cannot capture the adaptive \
             solver's convergence read-back, and -fixedIters is what removes it"
                .to_string(),
        ));
    }

    // The cuFFT backend reads the assembled matrix back to the HOST on every
    // solve in order to re-derive the operator it is inverting - see
    // `pressure/fft.rs` - and AMGX may do the same. Neither is capturable, and
    // `auto` cannot promise which one it will pick.
    if o.graph && o.backend != BackendPick::Pbicgstab {
        usage();
        return Err(Error::Config(
            "-graph needs -backend pbicgstab: the other backends read the matrix \
             back to the host inside solve(), which a stream capture forbids"
                .to_string(),
        ));
    }

    for (flag, v) in [("-outerIters", o.outer_iters), ("-nCorrectors", o.n_correctors)] {
        if v < 1 {
            usage();
            return Err(Error::Config(format!("{flag} must be at least 1, got {v}")));
        }
    }

    // A non-positive end time is the sentinel for "steady", so a user who
    // typed one meaning something else is told rather than silently handed a
    // steady run.
    let given = |flag: &str| args.iter().any(|a| a == flag);

    for (flag, v) in [
        ("-endTime", o.end_time),
        ("-deltaT", o.delta_t),
        ("-writeInterval", o.write_interval),
    ] {
        if v <= 0.0 && given(flag) {
            usage();
            return Err(Error::Config(format!("{flag} must be positive, got {v}")));
        }
    }

    if o.end_time <= 0.0 && (given("-deltaT") || given("-writeInterval")) {
        eprintln!(
            "[ofgpu-buoyant] -deltaT and -writeInterval do nothing without -endTime; \
             this run is steady"
        );
    }

    Ok(o)
}

// ==========================================================================
//  Case controls the SIMPLE loop needs and `read_case_controls` does not read
// ==========================================================================

/// `fvSolution`'s `solvers/U`, `solvers/p`, `relaxationFactors` and
/// `SIMPLE/nNonOrthogonalCorrectors`, folded into the momentum and pressure
/// controls.
///
/// [`read_case_controls`] stops at the turbulence equations, because until now
/// nothing in this crate solved for `U` or `p`. Rather than widen that
/// function - every other driver would then carry settings it never uses -
/// the two extra sub-dictionaries are read here, the same way `ofgpu-plume`
/// reads `solvers/T`.
///
/// The defaults are `BUOYANT.md` section 3's: `U` 0.7, `p` 0.3.
fn read_simple_controls(case_dir: &Path, cc: &CaseControls) -> Result<SimpleControls> {
    // `div(phi,U)`, by its own name. This line used to read `cc.turb.div_scheme`
    // - the TURBULENCE equation's entry - so a case saying
    // `div(phi,U) Gauss linearUpwind; div(phi,k) bounded Gauss upwind;`
    // discretised its momentum equation first-order and said nothing.
    let u_conv = cc.schemes.div("div(phi,U)")?;

    let mut sc = SimpleControls {
        momentum: MomentumControls {
            nu: cc.nu,
            div_scheme: u_conv.scheme,
            bounded_convection: u_conv.bounded,
            grad_scheme: cc.schemes.grad("grad(U)")?,
            sn_grad: ofgpu::io::case::resolve_sn_grad(&cc.schemes, "default")?,
            n_non_orth_correctors: cc.turb.n_non_orth_correctors,
            // The scheme, not just the boolean: SPEC-LIT 13.4.
            ddt: cc.turb.ddt,
            lts: cc.lts,
            steady: cc.turb.steady,
            delta_t: cc.turb.delta_t,
            ..MomentumControls::default()
        },
        n_non_orth_correctors: cc.turb.n_non_orth_correctors,
        // SPEC-LIT 14. `read_case_controls` has already read whichever of
        // SIMPLE/PISO/PIMPLE the case wrote; every one of these entries used
        // to be parsed and dropped.
        n_correctors: cc.algorithm.n_correctors,
        n_outer_correctors: cc.algorithm.n_outer_correctors,
        momentum_predictor: cc.algorithm.momentum_predictor,
        ..SimpleControls::default()
    };

    // SPEC-LIT 5.3. `consistent yes;` selects SIMPLEC, whose `rAtU` keeps the
    // neighbour corrections plain SIMPLE drops. The code has had SIMPLEC in it
    // all along and nothing ever set this flag, so a case asking for it got
    // SIMPLE with whatever alpha_p it had written for SIMPLEC - which for the
    // usual `alpha_p 1` is a divergent run.
    sc.momentum.simplec = cc.algorithm.consistent;

    let p = case_dir.join("system").join("fvSolution");
    if !p.exists() {
        return Ok(sc);
    }
    let d = FoamDict::read(&p)?;

    read_solver(&mut sc.momentum.u_solver, &d, "U")?;
    read_solver(&mut sc.p_solver, &d, "p")?;

    sc.momentum.u_relax = d.scalar("relaxationFactors/equations/U", sc.momentum.u_relax);
    // p is relaxed as a FIELD, not as an equation (Patankar 1980 6.7) - the relaxation is
    // applied to the solution, not folded into the matrix - so that is where
    // the entry lives. The equations spelling is accepted too, because cases
    // in the wild carry it.
    sc.p_relax = d.scalar(
        "relaxationFactors/fields/p",
        d.scalar("relaxationFactors/equations/p", sc.p_relax),
    );

    // A corrector COUNT for both the pressure and the momentum equation. It
    // used to be a count for the pressure equation AND the on/off switch for
    // the momentum equation's non-orthogonal correction, so writing 0 - normal
    // on an orthogonal mesh - disabled the correction rather than asking for
    // one pass of it. Whether it is applied is `snGradSchemes`.
    //
    // Read through `AlgorithmControls` so `PIMPLE { nNonOrthogonalCorrectors
    // 2; }` counts as much as the `SIMPLE` spelling of it.
    let algo = ofgpu::io::case::AlgorithmControls::read(&d);
    sc.n_non_orth_correctors = algo.n_non_orth_correctors;
    sc.momentum.n_non_orth_correctors = algo.n_non_orth_correctors;
    sc.n_correctors = algo.n_correctors;
    sc.n_outer_correctors = algo.n_outer_correctors;
    sc.momentum_predictor = algo.momentum_predictor;
    sc.momentum.simplec = algo.consistent;

    // Not validated here: `SimpleControls::validate` is private and
    // `Simple::new` runs it before anything is allocated, so a relaxation
    // factor outside (0, 1] is still refused - just a few lines later, with a
    // message that names the same entry.
    Ok(sc)
}

/// One `solvers/<var>` sub-dictionary into a [`SolverControls`].
///
/// Delegates to the crate's own reader rather than repeating it: this file
/// used to carry a copy that read `preconditioner` and ignored `solver`, which
/// is exactly the silent discard SPEC-LIT 13.4 is about.
fn read_solver(sc: &mut SolverControls, d: &FoamDict, var: &str) -> Result<()> {
    ofgpu::io::case::read_solver_controls(sc, d, var)
}

/// The two Prandtl numbers, from the case if it names them.
///
/// `Pr` is a fluid property and lives beside `nu` in `physicalProperties`;
/// `Prt` is a model constant, so it is looked up the way every other model
/// constant is - in the RAS dictionary - and only then falls back to the
/// `physicalProperties` spelling that the buoyant tutorials use.
fn read_prandtl(case_dir: &Path, cc: &CaseControls) -> Result<ScalarTransportCoeffs> {
    let d = ScalarTransportCoeffs::default();
    let mut c = ScalarTransportCoeffs {
        pr: d.pr,
        prt: model_coeff(cc, "Prt", d.prt),
    };

    for nm in ["physicalProperties", "transportProperties"] {
        let p = case_dir.join("constant").join(nm);
        if p.exists() {
            let f = FoamDict::read(&p)?;
            c.pr = f.scalar("Pr", c.pr);
            c.prt = f.scalar("Prt", c.prt);
            break;
        }
    }

    Ok(c)
}

/// `fvSolution`'s `solvers/T` and `relaxationFactors/equations/T`, folded into
/// the copy of the controls the temperature equation will use.
///
/// [`ScalarTransport`] reads its linear solver out of `k_solver` and its
/// relaxation out of `k_relax`, because `TurbulenceControls` has no slot for a
/// passive scalar. Overwriting exactly those two on a *copy* is how a case
/// gives `T` its own settings without disturbing the model's.
fn read_t_controls(case_dir: &Path, base: &TurbulenceControls) -> Result<TurbulenceControls> {
    let mut t = *base;

    let p = case_dir.join("system").join("fvSolution");
    if !p.exists() {
        return Ok(t);
    }
    let d = FoamDict::read(&p)?;

    read_solver(&mut t.k_solver, &d, "T")?;
    t.k_relax = d.scalar("relaxationFactors/equations/T", t.k_relax);

    Ok(t)
}

/// `fvSolution`'s `solvers/Phi`, for the one-off potential-flow seed.
///
/// Its own entry rather than a borrowed one: the Laplace matrix is symmetric
/// and much stiffer than any transport equation, so the turbulence solver's
/// `relTol 0.01` would stop it long before the flux was conservative, which is
/// the only thing this solve is for.
fn read_phi_controls(case_dir: &Path) -> Result<SolverControls> {
    let mut sc = SolverControls {
        tolerance: 1e-12,
        rel_tol: 0.0,
        max_iter: 5000,
        ..Default::default()
    };

    let p = case_dir.join("system").join("fvSolution");
    if !p.exists() {
        return Ok(sc);
    }
    read_solver(&mut sc, &FoamDict::read(&p)?, "Phi")?;
    Ok(sc)
}

/// Every linear solver switched into the transfer-free mode.
fn make_fixed(sc: &mut SolverControls, sweeps: Label) {
    sc.fixed_iters = true;
    sc.max_iter = sweeps;
    sc.report_residuals = false;
}

// ==========================================================================
//  The starting flux
// ==========================================================================

/// Where the STARTING `phi` came from.
///
/// Unlike `ofgpu-plume`, where the flux is the whole physics and is frozen for
/// the run, here it is only an initial condition: the first pressure
/// correction replaces it with one built off the pressure operator, and from
/// then on `phi` is conservative to the pressure solver's tolerance whatever
/// it started as. It is still worth starting from something sensible, because
/// the very first momentum convection term is assembled on it.
enum FluxSource {
    File,
    Potential(PotentialFlowResult),
    Interpolated,
}

/// Establish the starting `phi`, and with it the `U` that goes with it.
///
/// In order of preference:
///
/// 1. a `phi` in the time directory - it came out of a solver that satisfied
///    discrete continuity, and nothing here improves on that;
/// 2. `interpolate(U) & Sf`, whenever the `U` on disk is already moving. That
///    is a RESTART, and the developed field describes the flow far better than
///    any potential flow does; it is not conservative, and it does not have to
///    be, because the first pressure correction projects it onto the flux the
///    pressure operator implies;
/// 3. potential flow, for a cold start from rest. `interpolate(0) & Sf` is
///    zero everywhere, which would leave the first momentum equation with no
///    convection at all and the burner injecting mass into a domain with no
///    path out of it.
///
/// Case 3 overwrites `u`'s internal field, and that is the point: the velocity
/// has to be the one the flux implies. Case 2 must NOT, which is why the two
/// are distinguished at all - handing a restarted run a potential-flow
/// velocity would throw away the solution it was restarted from.
///
/// Takes free-standing fields rather than the ones inside [`Simple`] because
/// [`solve_potential_flow`] needs `phi` and `U` mutably at the same time and
/// `Simple` cannot hand out two mutable borrows of itself. They are copied in
/// afterwards; it is setup, and it happens once.
#[allow(clippy::too_many_arguments)]
fn establish_flux(
    gpu: &Gpu,
    phi: &mut GpuSurfaceScalarField,
    u: &mut GpuVectorField,
    hm: &HostMesh,
    mesh: &GpuMesh,
    t_dir: &Path,
    o: &Options,
    ctrl: &SolverControls,
    u_at_rest: bool,
) -> Result<FluxSource> {
    let path = t_dir.join("phi");

    if path.exists() {
        load_phi(gpu, phi, hm, &path)?;
        return Ok(FluxSource::File);
    }

    if u_at_rest && !o.no_potential {
        if let Some(r) = potential_flux(gpu, phi, u, hm, mesh, o, ctrl)? {
            return Ok(FluxSource::Potential(r));
        }
    }

    compute_phi_from_u(gpu, phi, u, hm)?;
    println!(
        "phi started from interpolate(U) & Sf{} - not conservative, and the first \
         pressure correction is what fixes that",
        if u_at_rest { "" } else { ", because the U on disk is already moving" }
    );
    Ok(FluxSource::Interpolated)
}

/// Solve for the flux, or explain why this case cannot be solved for and
/// return `None`.
fn potential_flux(
    gpu: &Gpu,
    phi: &mut GpuSurfaceScalarField,
    u: &mut GpuVectorField,
    hm: &HostMesh,
    mesh: &GpuMesh,
    o: &Options,
    ctrl: &SolverControls,
) -> Result<Option<PotentialFlowResult>> {
    for name in [&o.inlet_patch, &o.outlet_patch] {
        if !hm.patches.iter().any(|p| &p.name == name) {
            println!("no patch named \"{name}\", so the starting flux cannot be solved for");
            println!("  pass -inletPatch/-outletPatch to name the openings, or -noPotential");
            return Ok(None);
        }
    }

    let u_in = mean_inflow_speed(&gpu.download(&u.bf)?, hm, &o.inlet_patch)?;

    if !u_in.is_finite() || u_in <= 0.0 {
        println!(
            "the {} patch carries no inflow (mean normal velocity {}), so there is \
             nothing for potential flow to distribute",
            o.inlet_patch,
            g(f64::from(u_in))
        );
        return Ok(None);
    }

    let spec = PotentialFlowSpec {
        inlet_patch: o.inlet_patch.clone(),
        inlet_normal_velocity: u_in,
        outlet_patch: o.outlet_patch.clone(),
    };

    Ok(Some(solve_potential_flow(gpu, hm, mesh, phi, u, &spec, ctrl)?))
}

/// Read a `phi` written by an earlier run.
fn load_phi(
    gpu: &Gpu,
    phi: &mut GpuSurfaceScalarField,
    hm: &HostMesh,
    path: &Path,
) -> Result<()> {
    let raw = read_scalar_field(path, hm.n_internal_faces)?;
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

/// The start-up log line for whichever flux the run started from.
fn report_flux(source: &FluxSource) {
    if let FluxSource::Potential(r) = source {
        println!(
            "phi seeded from potential flow: laplacian(Phi) = 0 in {} iterations, residual {}",
            r.iterations,
            sci(f64::from(r.final_residual), 3)
        );
        println!(
            "  inlet flux {} m3/s   outlet flux {} m3/s   imbalance {} m3/s",
            sci(f64::from(r.inlet_flux), 6),
            sci(f64::from(r.outlet_flux), 6),
            sci(f64::from(r.imbalance()), 3)
        );
    }
}

// ==========================================================================
//  The stratification probe
// ==========================================================================

/// What `-probe` measures: the mean temperature of the top and the bottom
/// third of the domain.
#[derive(Debug, Clone, Copy)]
struct Layers {
    upper: Scalar,
    lower: Scalar,
    /// Height of the two cut planes, in metres along the up direction.
    lower_top: Scalar,
    upper_bottom: Scalar,
}

impl Layers {
    /// The number the case is judged on. Positive means the hot gas went up.
    fn contrast(&self) -> Scalar {
        self.upper - self.lower
    }
}

/// Mass-weighted mean `T` in the upper and lower thirds of the domain.
///
/// "Up" is `-g` normalised rather than `+z`, so the probe still means
/// something on a case whose gravity is not axis-aligned; on the plume case it
/// is exactly `+z`.
///
/// The weight is `V*rho/rho_ref = V*TRef/T`, the SAME density ratio the
/// buoyancy term uses, so this really is a mass-weighted mean rather than a
/// volume-weighted one wearing the name. It matters here: hot cells hold less
/// mass, so volume weighting would overstate the upper third by several kelvin
/// on a developed plume and flatter the very result being tested.
///
/// Cell centres decide which third a cell is in, so a cell straddling a cut
/// plane belongs to one third only and the two never double-count. `T` is
/// clamped at 1 K in the weight for the same reason `BuoyancyCoeffs` clamps
/// it: a corrupted zero must not make one cell weigh infinity.
fn stratification(t: &[Scalar], hm: &HostMesh, up: Vec3, t_ref: Scalar) -> Result<Layers> {
    if t.len() != hm.n_cells || hm.c.len() != hm.n_cells || hm.v.len() != hm.n_cells {
        return Err(Error::Config(format!(
            "stratification: {} temperatures, {} centres and {} volumes for {} cells",
            t.len(),
            hm.c.len(),
            hm.v.len(),
            hm.n_cells
        )));
    }
    if hm.n_cells == 0 {
        return Err(Error::Config("stratification: empty mesh".to_string()));
    }

    let h: Vec<Scalar> = hm.c.iter().map(|c| c.dot(up)).collect();
    let mut lo = h[0];
    let mut hi = h[0];
    for v in &h {
        lo = lo.min(*v);
        hi = hi.max(*v);
    }

    let third = (hi - lo) / 3.0;
    let lower_top = lo + third;
    let upper_bottom = hi - third;

    // (weighted sum, weight) for the lower and the upper third.
    let mut acc = [(0.0 as Scalar, 0.0 as Scalar); 2];

    for c in 0..hm.n_cells {
        // Two independent tests rather than an if/else chain. With a real
        // extent `lower_top < upper_bottom` and a cell can satisfy at most one
        // of them, so nothing is double-counted; with NO extent along `up` -
        // a 2-D case whose one cell layer is perpendicular to gravity - both
        // are satisfied by every cell, both thirds are the whole domain, and
        // the contrast comes out exactly zero. That is the honest answer for a
        // mesh in which stratification cannot be measured, and it is much
        // better than an error that would abort an otherwise valid run.
        let w = hm.v[c] * t_ref / t[c].max(1.0);

        if h[c] <= lower_top {
            acc[0].0 += w * t[c];
            acc[0].1 += w;
        }
        if h[c] >= upper_bottom {
            acc[1].0 += w * t[c];
            acc[1].1 += w;
        }
    }

    let mean = |(sum, w): (Scalar, Scalar)| if w > 0.0 { sum / w } else { Scalar::NAN };

    Ok(Layers {
        lower: mean(acc[0]),
        upper: mean(acc[1]),
        lower_top,
        upper_bottom,
    })
}

/// The probe line, at a write.
///
/// A negative contrast DURING the run is marked but not shouted about: the
/// plume needs roughly `H/w` seconds to reach the ceiling - about a second and
/// a half on this case - and until it does the upper third is honestly still
/// ambient. Shouting at every early write would train a reader to ignore the
/// one place it matters, which is [`print_verdict`].
fn print_layers(l: &Layers, up: Vec3) {
    println!(
        "    T upper third {} K   lower third {} K   contrast {} K   \
         (cut at {} and {} m along ({}, {}, {})){}",
        g(f64::from(l.upper)),
        g(f64::from(l.lower)),
        g(f64::from(l.contrast())),
        g(f64::from(l.lower_top)),
        g(f64::from(l.upper_bottom)),
        g(f64::from(up.x)),
        g(f64::from(up.y)),
        g(f64::from(up.z)),
        if l.contrast() > 0.0 { "" } else { "   <- upper third NOT hotter yet" }
    );
}

/// The closing statement on whether this run was a plume at all.
///
/// Loud on purpose when the answer is no. A buoyant run that produces a
/// beautiful field with no stratification in it is the exact failure this
/// driver exists to catch, and a reader skimming a log will not notice a quiet
/// number.
fn print_verdict(l: &Layers) {
    if l.contrast() > 0.0 {
        println!(
            "  the upper third is {} K hotter than the lower third: the hot gas rose, \
             spread under the ceiling, and left the bottom of the room near ambient.",
            g(f64::from(l.contrast()))
        );
        return;
    }

    let rule = "!".repeat(74);
    println!("{rule}");
    println!(
        "!!  BUOYANCY IS NOT WORKING. The upper third is {} K, the lower third {} K:",
        g(f64::from(l.upper)),
        g(f64::from(l.lower))
    );
    println!("!!  the hot gas has not risen, so nothing in this run is a plume.");
    println!("!!  Check the sign of constant/g, TRef in physicalProperties, that the T");
    println!("!!  field really is hotter at the burner than the ambient - and that the");
    println!("!!  run was long enough for the plume to cross the room in the first place.");
    println!("{rule}");
}

// ==========================================================================
//  Writing a time directory
// ==========================================================================

/// Where results go, and the input fields whose boundary *type* strings every
/// written directory has to carry.
///
/// Bundled because a transient run writes many directories and every one of
/// them must be seeded from the same originals: `harvest_scalar_field` only
/// invents a type where none is set, so a directory written without the seeds
/// comes out `calculated` everywhere and cannot start another run.
struct FieldWriter<'a> {
    case_dir: &'a Path,
    raw_u: &'a RawVectorField,
    raw_p: &'a RawScalarField,
    raw_k: &'a RawScalarField,
    raw_e: &'a RawScalarField,
    raw_t: &'a RawScalarField,
    output: &'a [OutputFormat],
}

impl FieldWriter<'_> {
    #[allow(clippy::too_many_arguments)]
    fn write(
        &self,
        gpu: &Gpu,
        name: &str,
        step: usize,
        time: Scalar,
        s: &Simple<'_>,
        model: &KEpsilon<'_>,
        heat: &ScalarTransport<'_>,
        hm: &HostMesh,
    ) -> Result<PathBuf> {
        let out_dir = self.case_dir.join(name);

        let mut out_u = seed_vector_types(self.raw_u);
        let mut out_p = seed_types(self.raw_p);
        let mut out_k = seed_types(self.raw_k);
        let mut out_e = seed_types(self.raw_e);
        let mut out_t = seed_types(self.raw_t);
        let mut out_nut = RawScalarField::default();
        // `phi` has no input file to inherit boundary types from - it is the
        // pressure equation's output, not the case's input - so the harvest
        // invents them.
        let mut out_phi = RawScalarField::default();

        harvest_vector_field(gpu, &mut out_u, s.u(), hm)?;
        harvest_scalar_field(gpu, &mut out_p, s.p(), hm)?;
        harvest_scalar_field(gpu, &mut out_k, model.k(), hm)?;
        harvest_scalar_field(gpu, &mut out_e, model.epsilon(), hm)?;
        harvest_scalar_field(gpu, &mut out_t, heat.field(), hm)?;
        harvest_scalar_field(gpu, &mut out_nut, model.nut(), hm)?;
        harvest_surface_scalar_field(gpu, &mut out_phi, s.phi(), hm)?;

        out_u.dimensions = "[0 1 -1 0 0 0 0]".to_string();
        // Kinematic pressure: p/rho, which is what an incompressible momentum
        // equation carries and what `simple.rs` solves for.
        out_p.dimensions = "[0 2 -2 0 0 0 0]".to_string();
        out_k.dimensions = "[0 2 -2 0 0 0 0]".to_string();
        out_e.dimensions = "[0 2 -3 0 0 0 0]".to_string();
        out_nut.dimensions = "[0 2 -1 0 0 0 0]".to_string();
        out_t.dimensions = if self.raw_t.dimensions.is_empty() {
            "[0 0 0 1 0 0 0]".to_string()
        } else {
            self.raw_t.dimensions.clone()
        };

        // One seam call per requested format, replacing what used to be six
        // scattered `fields::write_*` sites - see `ofgpu::io::writer`.
        let foam_fields = [
            ofgpu::io::FoamField::vector("U", &out_u),
            ofgpu::io::FoamField::scalar("p", &out_p),
            ofgpu::io::FoamField::scalar("k", &out_k),
            ofgpu::io::FoamField::scalar("epsilon", &out_e),
            ofgpu::io::FoamField::scalar("T", &out_t),
            ofgpu::io::FoamField::scalar("nut", &out_nut),
            // The conservative flux, so a restart from this directory begins
            // on the flux the pressure equation produced rather than on
            // `interpolate(U)·Sf` - SPEC-LIT §5.1 and the restart check of §22.
            ofgpu::io::FoamField::surface("phi", &out_phi),
        ];
        let vis_fields = [
            ofgpu::io::OutputField::vector("U", &out_u.internal),
            ofgpu::io::OutputField::scalar("p", &out_p.internal),
            ofgpu::io::OutputField::scalar("k", &out_k.internal),
            ofgpu::io::OutputField::scalar("epsilon", &out_e.internal),
            ofgpu::io::OutputField::scalar("T", &out_t.internal),
            ofgpu::io::OutputField::scalar("nut", &out_nut.internal),
        ];
        let cart = ofgpu::pressure::cartesian::detect(hm)
            .ok()
            .map(|c| ofgpu::io::cartesian_info(hm, &c));
        let ctx = ofgpu::io::WriteCtx {
            time,
            step,
            name,
            mesh: hm,
            cart: cart.as_ref(),
            fields: &vis_fields,
            foam: &foam_fields,
        };
        let mut writers = build_writers(self.case_dir, "buoyant", self.output)?;
        for w in &mut writers {
            w.write_step(&ctx)?;
        }

        Ok(out_dir)
    }
}

/// A destination field carrying only the source's boundary *type* strings.
fn seed_types(src: &RawScalarField) -> RawScalarField {
    // Through `types_only`, so a `boundaryField { ".*" {...} }` pattern comes
    // across with the types and `harvest_scalar_field` expands it into one
    // explicit entry per patch on the way out.
    src.types_only()
}

fn seed_vector_types(src: &RawVectorField) -> RawVectorField {
    src.types_only()
}

/// Everything one restart interval needs: `U`, `p`, `k`, `epsilon`, `T` and
/// `phi`, internal AND boundary values, plus `p0` (the mean pressure level -
/// see `common::mean`'s doc for why nothing here reads it back).
#[allow(clippy::too_many_arguments)]
fn write_restart_checkpoint(
    gpu: &Gpu,
    case_dir: &Path,
    mesh_hash: u64,
    t: Scalar,
    s: &Simple<'_>,
    model: &KEpsilon<'_>,
    heat: &ScalarTransport<'_>,
    hm: &HostMesh,
) -> Result<()> {
    let u_i = gpu.download(&s.u().f)?;
    let u_b = gpu.download(&s.u().bf)?;
    let p_i = gpu.download(&s.p().f)?;
    let p_b = gpu.download(&s.p().bf)?;
    let k_i = gpu.download(&model.k().f)?;
    let k_b = gpu.download(&model.k().bf)?;
    let e_i = gpu.download(&model.epsilon().f)?;
    let e_b = gpu.download(&model.epsilon().bf)?;
    let t_i = gpu.download(&heat.field().f)?;
    let t_b = gpu.download(&heat.field().bf)?;
    let phi_i = gpu.download(&s.phi().f)?;
    let phi_b = gpu.download(&s.phi().bf)?;

    let p0 = mean(&p_i);
    let mut data = restart_shell(mesh_hash, t, p0, hm);
    data.fields.push(restart_vector("U", &u_i, &u_b));
    data.fields.push(restart_scalar("p", &p_i, &p_b));
    data.fields.push(restart_scalar("k", &k_i, &k_b));
    data.fields.push(restart_scalar("epsilon", &e_i, &e_b));
    data.fields.push(restart_scalar("T", &t_i, &t_b));
    data.fields.push(restart_surface("phi", &phi_i, &phi_b));

    let path = case_dir.join("restart.mcr");
    restart::write_restart(&path, &data)?;
    println!(
        "    restart checkpoint written to {} (t = {}, p0 = {})",
        path.display(),
        g(f64::from(t)),
        sci(f64::from(p0), 3)
    );
    Ok(())
}

// ==========================================================================
//  The unit of work
// ==========================================================================

/// What the log line for one unit of work prints. Zeroes throughout in
/// `-fixedIters` mode, where nothing is read back.
#[derive(Default, Clone, Copy)]
struct Residuals {
    u: [Scalar; 3],
    u_iters: [usize; 3],
    p: Scalar,
    p_iters: usize,
    k: Scalar,
    k_iters: usize,
    eps: Scalar,
    eps_iters: usize,
    t: Scalar,
    t_iters: usize,
    /// `max_c |sum_f phi_f|` after the last flux correction.
    continuity: Scalar,
}

impl Residuals {
    /// The largest of the seven, for the steady convergence test a case with
    /// no `residualControl` falls back to.
    fn worst(&self) -> Scalar {
        self.u
            .iter()
            .copied()
            .chain([self.p, self.k, self.eps, self.t])
            .fold(0.0 as Scalar, Scalar::max)
    }

    /// The same numbers, labelled with the field names a `residualControl`
    /// dictionary uses.
    ///
    /// `U` carries the largest of its three components: the entry names the
    /// vector, so the vector is converged when all of it is.
    fn by_field(&self) -> [(&'static str, Scalar); 5] {
        [
            ("U", self.u.iter().copied().fold(0.0 as Scalar, Scalar::max)),
            ("p", self.p),
            ("k", self.k),
            ("epsilon", self.eps),
            ("T", self.t),
        ]
    }
}

/// One pass: `n_correctors` SIMPLE iterations, then turbulence, then the
/// scalar both of them diffuse.
///
/// The order is fixed and this is one function rather than three call sites
/// because the capture path and the ordinary path must record exactly the same
/// sequence. It is also the physical order: `k-epsilon` needs the velocity
/// SIMPLE just produced, and `T` needs the flux SIMPLE just made conservative
/// and the `nut` k-epsilon just produced from it.
fn one_pass(
    gpu: &Gpu,
    s: &mut Simple<'_>,
    model: &mut KEpsilon<'_>,
    heat: &mut ScalarTransport<'_>,
    backend: &mut dyn PressureBackend,
    n_correctors: Label,
) -> Result<Residuals> {
    let mut r = Residuals::default();
    let _ = n_correctors;

    // ONE call, which runs `nOuterCorrectors` outer correctors internally,
    // each of them `nCorrectors` PISO correctors, each of those
    // `nNonOrthogonalCorrectors + 1` pressure solves - SPEC-LIT 14.
    //
    // It used to be a host-side loop calling `Simple::correct` N times, and
    // because `correct` refreshed `U^{n-1}` on entry that loop advanced the
    // VELOCITY by N Euler sub-steps of dt per time step while `T`, `k` and
    // `epsilon` advanced by one. The fields came apart in time and nothing in
    // the output said so. `Simple::correct_outer` now stores the old level in
    // a steady run only; the transient one gets it from `begin_time_step`.
    {
        let perf = s.solve_step(gpu, backend, model.nut(), heat.field())?;

        // The residuals reported are the FIRST outer corrector's: they measure
        // the error the step started with, which is what a convergence history
        // means. The iteration counts are the last one's, because those are
        // what the run actually spent.
        for c in 0..3 {
            r.u[c] = perf.first.u[c].initial_residual;
            r.u_iters[c] = perf.last.u[c].n_iterations;
        }
        r.p = perf.first.p.initial_residual;
        r.p_iters = perf.last.p.n_iterations;
        r.continuity = perf.last.continuity_error;
    }

    let flow = s.flow_state();
    // The temperature reaches the turbulence model, which is what makes the
    // buoyancy production of SPEC-LIT 17 possible at all. `heat.field()` has
    // had its boundary conditions evaluated by the previous `heat.correct`,
    // which grad(T) reads directly.
    let (eps, k) = model.correct_buoyant(gpu, &flow, Some(heat.field()))?;
    let t = heat.correct(gpu, &flow, model.nut())?;

    r.k = k.initial_residual;
    r.k_iters = k.n_iterations;
    r.eps = eps.initial_residual;
    r.eps_iters = eps.n_iterations;
    r.t = t.initial_residual;
    r.t_iters = t.n_iterations;

    Ok(r)
}

/// One unit of work: `outer` passes of [`one_pass`].
///
/// The residuals returned are the LAST pass's - the only pass whose systems
/// were assembled from coefficients the step had already settled on.
fn one_step(
    gpu: &Gpu,
    s: &mut Simple<'_>,
    model: &mut KEpsilon<'_>,
    heat: &mut ScalarTransport<'_>,
    backend: &mut dyn PressureBackend,
    outer: Label,
    n_correctors: Label,
    dt: Scalar,
) -> Result<Residuals> {
    // ONE rotation of the time levels per TIME STEP, before the correctors -
    // not one per corrector, which would collapse U^{n-2} onto U^{n-1} and
    // make `ddtSchemes backward` quietly first order (SPEC-LIT 13.3).
    s.begin_time_step(gpu, dt)?;

    let mut r = Residuals::default();
    for _ in 0..outer.max(1) {
        r = one_pass(gpu, s, model, heat, backend, n_correctors)?;
    }
    Ok(r)
}

/// The residual block, three lines, the same in both loops.
fn print_report(head: &str, r: &Residuals, t_stats: (Scalar, Scalar)) {
    println!("{head}");
    println!(
        "    res  Ux {} ({})  Uy {} ({})  Uz {} ({})  p {} ({})",
        sci(f64::from(r.u[0]), 3),
        r.u_iters[0],
        sci(f64::from(r.u[1]), 3),
        r.u_iters[1],
        sci(f64::from(r.u[2]), 3),
        r.u_iters[2],
        sci(f64::from(r.p), 3),
        r.p_iters
    );
    println!(
        "         k {} ({})  eps {} ({})  T {} ({})   T[min,max] {} {} K   \
         max |sum_f phi| {} m3/s",
        sci(f64::from(r.k), 3),
        r.k_iters,
        sci(f64::from(r.eps), 3),
        r.eps_iters,
        sci(f64::from(r.t), 3),
        r.t_iters,
        g(f64::from(t_stats.0)),
        g(f64::from(t_stats.1)),
        sci(f64::from(r.continuity), 3)
    );
}

// ==========================================================================
//  The schedule
// ==========================================================================

/// The resolved plan. Every field is validated before it is built, so neither
/// loop ever has to ask whether a number makes sense.
struct Schedule {
    transient: bool,
    dt: f64,
    n_steps: i64,
    /// Non-positive means "at the end only".
    write_interval: f64,
    outer_iters: Label,
    n_correctors: Label,
    check_every: Label,
    convergence_tol: Scalar,
    /// `SIMPLE/residualControl`. Empty when the case gave none, in which case
    /// `convergence_tol` is what stops the run.
    residual_control: ofgpu::io::case::ResidualControl,
    /// Whether a residual was read back at all.
    ///
    /// `-fixedIters` leaves every residual at zero, and zero is below any
    /// tolerance: without this the steady loop would declare victory on its
    /// first check having measured nothing. A fixed-iteration run therefore
    /// runs its full iteration count, which is the only honest thing it can
    /// do.
    residuals_read: bool,
    do_write: bool,
    graph: bool,
    probe_layers: bool,
    /// Directory the final write lands in.
    final_name: String,
    /// `-restartWrite N` - write a `.mcr` checkpoint every N steps.
    restart_write: Option<u64>,
    mesh_hash: u64,
    /// The simulated time the run starts counting from - `0` for a cold
    /// start, the checkpoint's own time for `-restartFrom`. `t` at step `n`
    /// is `t0 + n*dt`, never `n*dt` alone: an `-endTime` after a restart
    /// names the ABSOLUTE time to reach, not an additional duration, and a
    /// `t` that forgot `t0` would silently redo the steps the checkpoint
    /// already covers.
    t0: f64,
}

/// What the summary block needs out of the loop.
struct RunReport {
    steps: i64,
    writes: usize,
    /// Wall clock of the stepping alone. The stats read-back and the field
    /// writes are timed separately and subtracted, because "how fast does it
    /// step" and "how long did the run take" are different questions and both
    /// were asked.
    loop_wall: f64,
    io_wall: f64,
    converged: bool,
}

/// Is `t` a multiple of the write interval, to within half a step?
///
/// Half a step rather than an exact comparison because `t` is `step*dt` in
/// binary floating point and will not land exactly on `W`. Half a step is also
/// the widest window that cannot match two consecutive steps, which is what
/// stops one write time producing two directories.
fn is_write_time(t: f64, w: f64, dt: f64) -> bool {
    if w <= 0.0 {
        return false;
    }
    let n = (t / w).round();
    n >= 1.0 && (t - n * w).abs() <= 0.5 * dt
}

/// Everything the loop borrows and does not own.
struct Fields<'a, 'm> {
    s: &'a mut Simple<'m>,
    model: &'a mut KEpsilon<'m>,
    heat: &'a mut ScalarTransport<'m>,
    backend: &'a mut dyn PressureBackend,
}

/// The one loop. `Schedule::transient` decides whether a unit of work is a
/// time step or a SIMPLE outer iteration, and nothing else differs: the models
/// do not branch on the regime either - `steady` makes `r_delta_t()` zero and
/// `fvm_ddt_euler` then writes nothing at all.
#[allow(clippy::too_many_arguments)]
fn run_loop(
    gpu: &Gpu,
    f: Fields<'_, '_>,
    hm: &HostMesh,
    sched: &Schedule,
    writer: &FieldWriter<'_>,
    up: Vec3,
    t_ref: Scalar,
) -> Result<RunReport> {
    let Fields { s, model, heat, backend } = f;

    // Without -graph nothing is ever captured, so "warm-up" is the whole run.
    let warmup = if sched.graph {
        GRAPH_WARMUP.min(sched.n_steps)
    } else {
        sched.n_steps
    };

    let mut graph: Option<Graph> = None;
    let mut writes = 0usize;
    let mut io_wall = 0.0f64;
    let mut converged = false;
    let mut done = 0i64;

    gpu.sync()?;
    let t_loop = Instant::now();

    for step in 1..=sched.n_steps {
        done = step;

        // A capture executes nothing, so the step it happens on is advanced by
        // the replay immediately below.
        if sched.graph && graph.is_none() && step > warmup {
            gpu.sync()?;
            let t_cap = Instant::now();

            // Reborrowed rather than passed straight in: the closure is
            // `FnOnce` and takes its captures by value, and a `&mut` moved out
            // of a variable inside a loop cannot be used on the next pass.
            let captured = {
                let sm = &mut *s;
                let mm = &mut *model;
                let hm2 = &mut *heat;
                let bk = &mut *backend;
                gpu.capture(move |_| {
                    one_step(
                        gpu,
                        sm,
                        mm,
                        hm2,
                        bk,
                        sched.outer_iters,
                        sched.n_correctors,
                        sched.dt as Scalar,
                    )?;
                    Ok(())
                })?
            };

            let Some(mut gr) = captured else {
                return Err(Error::Config(
                    "capture produced an empty graph - one unit of work launched \
                     nothing"
                        .to_string(),
                ));
            };
            gr.upload()?;
            gpu.sync()?;

            println!(
                "  captured one unit ({} x [{} x SIMPLE + k-epsilon + T]) in {} s; \
                 replaying it. Residuals are not read back in this mode.",
                sched.outer_iters,
                sched.n_correctors,
                g(t_cap.elapsed().as_secs_f64())
            );

            graph = Some(gr);
        }

        let r = match &graph {
            Some(gr) => {
                gr.launch()?;
                Residuals::default()
            }
            None => one_step(
                gpu,
                s,
                model,
                heat,
                backend,
                sched.outer_iters,
                sched.n_correctors,
                sched.dt as Scalar,
            )?,
        };

        let t = sched.t0 + step as f64 * sched.dt;
        let last = step == sched.n_steps;

        // A steady run also stops when it stops moving, and it has to test for
        // that on the check interval rather than only at a write.
        let checking = !sched.transient && step % sched.check_every.max(1) as i64 == 0;
        // The case's own SIMPLE/residualControl decides, per field, on the
        // INITIAL residual - the residual of the system as it stood before
        // this iteration's linear solve, which is what measures the outer
        // iteration rather than the linear one. A case that gives none falls
        // back to the single hard-coded tolerance this driver always used.
        let residual_test = if sched.residual_control.is_empty() {
            r.worst() < sched.convergence_tol
        } else {
            sched.residual_control.all_satisfied(&r.by_field())
        };
        if checking && sched.residuals_read && step > 1 && residual_test {
            converged = true;
        }

        let reporting = if sched.transient {
            last || is_write_time(t, sched.write_interval, sched.dt)
        } else {
            last || checking || converged || step == 1
        };
        if !reporting {
            continue;
        }

        // Everything from here is host work, so the loop clock stops for it.
        gpu.sync()?;
        let t_io = Instant::now();

        let temps = gpu.download(&heat.field().f)?;
        let stats = weighted_stats(&temps, &hm.v)?;

        let head = if sched.transient {
            format!(
                "t = {:.3} s   step {step}   wall {:.1} s",
                t,
                t_loop.elapsed().as_secs_f64()
            )
        } else {
            format!(
                "iteration {step}   wall {:.1} s",
                t_loop.elapsed().as_secs_f64()
            )
        };
        print_report(&head, &r, (stats.min, stats.max));

        if sched.probe_layers {
            print_layers(&stratification(&temps, hm, up, t_ref)?, up);
        }

        // `inletOutlet` switches on the sign of the flux, and unlike the
        // frozen-flow drivers this one changes the flux every iteration. It is
        // a host round trip, so it is refreshed here - where the clock is
        // already stopped - rather than in the loop, which would put a
        // transfer back into the path `-fixedIters` exists to empty.
        update_inlet_outlet(gpu, model.k_mut(), s.phi(), hm)?;
        update_inlet_outlet(gpu, model.epsilon_mut(), s.phi(), hm)?;
        update_inlet_outlet(gpu, heat.field_mut(), s.phi(), hm)?;

        let write_now = sched.do_write && (sched.transient || last || converged);
        if write_now {
            let name = if last || converged {
                sched.final_name.clone()
            } else {
                format_time_name(t as Scalar)
            };
            let dir = writer.write(gpu, &name, step as usize, t as Scalar, s, model, heat, hm)?;
            writes += 1;
            println!("    written to {}", dir.display());
        }

        if let Some(interval) = sched.restart_write {
            if (step as u64) % interval == 0 {
                write_restart_checkpoint(
                    gpu,
                    writer.case_dir,
                    sched.mesh_hash,
                    t as Scalar,
                    s,
                    model,
                    heat,
                    hm,
                )?;
            }
        }

        io_wall += t_io.elapsed().as_secs_f64();

        if converged {
            if sched.residual_control.is_empty() {
                println!(
                    "converged: every residual below {}",
                    g(f64::from(sched.convergence_tol))
                );
            } else {
                println!("converged: every residualControl entry met");
            }
            break;
        }
    }

    gpu.sync()?;
    let elapsed = t_loop.elapsed().as_secs_f64();

    Ok(RunReport {
        steps: done,
        writes,
        loop_wall: (elapsed - io_wall).max(0.0),
        io_wall,
        converged,
    })
}

/// The numbers the run was asked for, in a block nothing else prints into.
fn print_summary(rep: &RunReport, sched: &Schedule, total_wall: f64) {
    let steps = rep.steps.max(1) as f64;

    let rule = "=".repeat(74);
    let thin = "-".repeat(74);

    println!("\n{rule}");
    println!(
        "  {} TIMING",
        if sched.transient { "TRANSIENT" } else { "STEADY" }
    );
    println!("{thin}");
    println!("  units of work                          {}", rep.steps);
    println!("  writes                                 {}", rep.writes);
    if sched.transient {
        println!(
            "  simulated time                         {} s   (dt = {} s, {} x [{} x SIMPLE \
             + k-epsilon + T] per step)",
            g(rep.steps as f64 * sched.dt),
            g(sched.dt),
            sched.outer_iters,
            sched.n_correctors
        );
    } else {
        println!(
            "  work per unit                          {} x [{} x SIMPLE + k-epsilon + T]",
            sched.outer_iters, sched.n_correctors
        );
    }
    println!("{thin}");
    println!(
        "  wall clock, TIME LOOP ALONE            {:.3} s   (no setup, no IO)",
        rep.loop_wall
    );
    println!(
        "  wall clock, INCLUDING setup + writes   {:.3} s   (of which IO {:.3} s)",
        total_wall, rep.io_wall
    );
    if sched.transient {
        let sim = rep.steps as f64 * sched.dt;
        println!(
            "  wall seconds per simulated second      {:.4}     (time loop alone)",
            rep.loop_wall / sim
        );
        println!(
            "  wall seconds per simulated second      {:.4}     (whole run)",
            total_wall / sim
        );
    }
    println!(
        "  ms per unit of work                    {:.4}",
        rep.loop_wall / steps * 1e3
    );
    println!("{rule}");
}

// ==========================================================================
//  Backend selection
// ==========================================================================

/// One backend by name, or the whole measured selection.
///
/// The probe is taken off the REAL assembled system - `Simple::correct` has
/// already run once by the time this is called - because every structural fact
/// a backend depends on is a fact about that matrix. Probing an all-zero
/// matrix would say the coefficient is constant and the operator symmetric,
/// which is true and useless.
fn pick_backend(
    gpu: &Gpu,
    hm: &HostMesh,
    mesh: &GpuMesh,
    s: &Simple<'_>,
    pick: BackendPick,
    p_ctrl: SolverControls,
) -> Result<Box<dyn PressureBackend>> {
    let (rauf_mag_sf, b_rauf_mag_sf) = s.pressure_laplacian_coeffs();
    let probe = SystemProbe::probe(gpu, hm, s.p(), s.pressure_matrix(), rauf_mag_sf, b_rauf_mag_sf)?;

    println!(
        "pressure system: {} cells | cartesian {} | separable BCs {} | symmetric {} | \
         constant coefficient {}",
        probe.n_cells,
        match probe.uniform_cartesian {
            Some((nx, ny, nz, dx, dy, dz)) => format!(
                "{nx}x{ny}x{nz}, h = ({}, {}, {})",
                g(f64::from(dx)),
                g(f64::from(dy)),
                g(f64::from(dz))
            ),
            None => format!("no ({})", probe.non_cartesian_reason),
        },
        if probe.separable_bcs {
            "yes".to_string()
        } else {
            format!("no ({})", probe.non_separable_reason)
        },
        probe.symmetric,
        probe.constant_coefficient
    );

    if pick == BackendPick::Auto {
        let candidates: Vec<Box<dyn PressureBackend>> = vec![
            Box::new(PbicgstabBackend::new(p_ctrl)),
            Box::new(FftBackend::new()),
            Box::new(AmgxBackend::new()),
        ];
        let (chosen, choice) =
            choose_pressure_backend(gpu, hm, mesh, s.pressure_matrix(), &probe, candidates)?;
        print!("{}", choice.report());
        return Ok(chosen);
    }

    let mut b: Box<dyn PressureBackend> = match pick {
        BackendPick::Pbicgstab | BackendPick::Auto => Box::new(PbicgstabBackend::new(p_ctrl)),
        BackendPick::Fft => Box::new(FftBackend::new()),
        BackendPick::Amgx => Box::new(AmgxBackend::new()),
    };

    // Applicability is a hard constraint even when the user asked by name. A
    // backend that cannot represent this system does not produce a slightly
    // worse answer, it produces a wrong one.
    if !b.applicable(&probe) {
        return Err(Error::Config(format!(
            "-backend {} cannot solve this system: {}",
            b.name(),
            b.why_not(&probe)
        )));
    }

    b.setup(gpu, hm, mesh, &probe)?;
    println!("pressure backend: {}   (named on the command line, not measured)", b.name());
    Ok(b)
}

// ==========================================================================
//  Driver
// ==========================================================================

fn run(o: &Options) -> Result<()> {
    // Started before anything else, because one of the numbers asked for is
    // the wall clock of the whole run - device init and mesh read included.
    let t_total = Instant::now();

    // ---- device -----------------------------------------------------------
    let gpu = Gpu::new(0)?;
    println!("{}", device_banner(&gpu, "buoyant")?);

    // ---- mesh -------------------------------------------------------------
    let t0 = Instant::now();

    let raw_mesh = read_poly_mesh(&o.case_dir)?;
    let hm = build_host_mesh(&raw_mesh)?;
    hm.print_report();

    let mesh = GpuMesh::upload(&gpu, &hm)?;
    gpu.sync()?;

    println!("mesh uploaded in {} s", g(t0.elapsed().as_secs_f64()));

    let mesh_hash = restart::mesh_hash(&hm);
    let restart_data: Option<RestartData> = match &o.restart_from {
        Some(p) => Some(restart::read_restart(p, mesh_hash)?),
        None => None,
    };
    if let Some(rd) = &restart_data {
        println!(
            "restart: loaded {} (t = {}, mesh hash 0x{:016x} matches)",
            o.restart_from.as_ref().unwrap().display(),
            g(rd.time),
            mesh_hash
        );
    }

    // ---- controls ---------------------------------------------------------
    let mut cc = read_case_controls(&o.case_dir)?;

    if o.n_iters > 0 {
        cc.turb.n_outer_iterations = o.n_iters;
    }
    cc.turb.convergence_check_every = o.check_every;

    // The whole of transient mode, and it has to happen HERE: every model
    // takes its controls BY VALUE, so a later change to `cc.turb` would be
    // invisible to them.
    let transient = o.end_time > 0.0;
    if transient {
        cc.turb.steady = false;
        if o.delta_t > 0.0 {
            cc.turb.delta_t = o.delta_t as Scalar;
        }
        if !(cc.turb.delta_t > 0.0 && cc.turb.delta_t.is_finite()) {
            return Err(Error::Config(format!(
                "-endTime needs a positive time step: no -deltaT was given and \
                 controlDict's deltaT is {}",
                g(f64::from(cc.turb.delta_t))
            )));
        }
        if o.n_iters > 0 {
            eprintln!(
                "[ofgpu-buoyant] -iters is ignored in transient mode: the step count \
                 comes from -endTime / -deltaT"
            );
        }
    }

    let mut simple_ctrl = read_simple_controls(&o.case_dir, &cc)?;
    let mut t_ctrl = read_t_controls(&o.case_dir, &cc.turb)?;

    // `-nCorrectors N` is the OUTER corrector count: N re-linearisations of
    // the momentum-pressure system per unit of work, which is PIMPLE's outer
    // loop (SPEC-LIT 14). It overrides `nOuterCorrectors` from fvSolution
    // because it is the more explicit request of the two - and the override is
    // printed below rather than applied in silence.
    simple_ctrl.n_outer_correctors = o.n_correctors.max(1) as usize;
    // residualControl stops the outer loop early on the INITIAL residuals.

    if o.fixed_iters > 0 {
        // The genuinely transfer-free mode, so the residual read-backs go too:
        // with all of them off nothing at all crosses the bus between upload
        // and write, which is also the precondition for -graph.
        for sc in [
            &mut cc.turb.k_solver,
            &mut cc.turb.epsilon_solver,
            &mut t_ctrl.k_solver,
            &mut simple_ctrl.momentum.u_solver,
            &mut simple_ctrl.p_solver,
        ] {
            make_fixed(sc, o.fixed_iters);
        }
        simple_ctrl.report_continuity = false;
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

    let t_coeffs = read_prandtl(&o.case_dir, &cc)?;
    let buoy: BuoyancyCoeffs = cc.buoyancy;

    println!(
        "nu = {} | Cmu {} C1 {} C2 {} sigmak {} sigmaEps {}",
        g(f64::from(cc.nu)),
        g(f64::from(coeffs.cmu)),
        g(f64::from(coeffs.c1)),
        g(f64::from(coeffs.c2)),
        g(f64::from(coeffs.sigmak)),
        g(f64::from(coeffs.sigma_eps))
    );
    println!(
        "T: Pr {} Prt {} -> alphaEff = nu/Pr + nut/Prt, laminar part {}",
        g(f64::from(t_coeffs.pr)),
        g(f64::from(t_coeffs.prt)),
        g(f64::from(cc.nu / t_coeffs.pr))
    );

    // The line that decides whether the plume rises or sinks, printed with the
    // one number that settles it rather than left for a picture to reveal.
    let hot = buoy.at(1173.15);
    println!(
        "buoyancy: g = ({}, {}, {}) m/s2, TRef = {} K -> b = g*(TRef/T - 1), NOT Boussinesq",
        g(f64::from(buoy.g.x)),
        g(f64::from(buoy.g.y)),
        g(f64::from(buoy.g.z)),
        g(f64::from(buoy.t_ref))
    );
    println!(
        "  at 1173.15 K: b = ({}, {}, {}) m/s2   |   at TRef: b = 0 exactly",
        g(f64::from(hot.x)),
        g(f64::from(hot.y)),
        g(f64::from(hot.z))
    );

    // `-g` normalised. Zero gravity leaves the probe with no axis to measure
    // along, and +z is the only defensible guess.
    let up = if buoy.g.mag() > 0.0 {
        let n = buoy.g.normalised();
        Vec3::new(-n.x, -n.y, -n.z)
    } else {
        Vec3::new(0.0, 0.0, 1.0)
    };

    // What the loop will DO, not which dictionary it was read from - a
    // `PIMPLE` dictionary with `nOuterCorrectors 1` runs PISO (SPEC-LIT 14).
    // The EFFECTIVE settings, after `-nCorrectors` has had its say: a line
    // that named the dictionary's `nOuterCorrectors` while the run used
    // another number would describe the file rather than the run.
    {
        let mut effective = cc.algorithm;
        effective.n_outer_correctors = simple_ctrl.n_outer_correctors;
        effective.n_correctors = simple_ctrl.n_correctors;
        effective.n_non_orth_correctors = simple_ctrl.n_non_orth_correctors;
        effective.momentum_predictor = simple_ctrl.momentum_predictor;
        effective.consistent = simple_ctrl.momentum.simplec;
        println!("algorithm: {}", effective.describe(cc.turb.steady));
    }
    if simple_ctrl.n_outer_correctors != cc.algorithm.n_outer_correctors {
        println!(
            "  -nCorrectors {} overrides nOuterCorrectors {} from fvSolution",
            simple_ctrl.n_outer_correctors, cc.algorithm.n_outer_correctors
        );
    }
    println!(
        "  relax U {} p {}{}",
        g(f64::from(simple_ctrl.momentum.u_relax)),
        g(f64::from(simple_ctrl.p_relax)),
        if cc.turb.steady {
            ""
        } else {
            ", both switched off on the final outer corrector"
        }
    );

    // ---- fields -----------------------------------------------------------
    let t = find_start_time(&o.case_dir)?;
    let t_dir = o.case_dir.join(&t);

    for name in ["U", "p", "T", "k", "epsilon"] {
        if !t_dir.join(name).exists() {
            return Err(Error::Config(format!(
                "{} has no {name} field; ofgpu-buoyant solves for U, p and T on top \
                 of k-epsilon, so the start time must provide all five",
                t_dir.display()
            )));
        }
    }

    let raw_u = read_vector_field(&t_dir.join("U"), hm.n_cells)?;
    let raw_p = read_scalar_field(&t_dir.join("p"), hm.n_cells)?;
    let mut raw_k = read_scalar_field(&t_dir.join("k"), hm.n_cells)?;
    let mut raw_e = read_scalar_field(&t_dir.join("epsilon"), hm.n_cells)?;
    let raw_t = read_scalar_field(&t_dir.join("T"), hm.n_cells)?;

    // SPEC-LIT 29.1: the per-field wall types must form one consistent row,
    // on this route exactly as on the JSONC route.
    {
        let nut_path = t_dir.join("nut");
        let mut raw_nut_row = if nut_path.exists() {
            Some(read_scalar_field(&nut_path, hm.n_cells)?)
        } else {
            None
        };
        ofgpu::field_setup::validate_wall_rows(
            &hm.patches,
            raw_nut_row.as_mut(),
            Some(&mut raw_k),
            Some(&mut raw_e),
            None,
        )?;
    }

    let fk = FieldKernels::new(&gpu)?;

    // ---- the starting flux ------------------------------------------------
    //
    // Solved for in free-standing fields and copied in, because
    // `solve_potential_flow` wants `phi` and `U` mutably at once and `Simple`
    // owns both. Two host round trips, at setup, once.
    let mut seed_u = GpuVectorField::zeros(&gpu, &mesh, "U")?;
    setup_vector_field(&gpu, &mut seed_u, &raw_u, &hm)?;
    correct_boundary_conditions_vector(&gpu, &fk, &mut seed_u, &mesh)?;

    let mut seed_phi = GpuSurfaceScalarField::zeros(&gpu, &mesh, "phi")?;
    let phi_ctrl = read_phi_controls(&o.case_dir)?;

    // Read off the FILE, not off the device: a cold start is a `0/U` whose
    // interior is all zeros, and that is a property of what was written rather
    // than of anything the boundary evaluation has since put on the faces.
    let u_at_rest = raw_u.internal.iter().all(|v| v.mag_sqr() == 0.0);

    if let Some(rd) = &restart_data {
        // The conservative flux this restart was written with - SPEC-LIT
        // 5.1. Skips `establish_flux` entirely, which would otherwise fall
        // back to potential flow or `interpolate(U) & Sf` - exactly the
        // non-conservative starting points a restart exists to avoid.
        let phi = find_restart_field(rd, "phi")?;
        gpu.write(&mut seed_phi.f, &from_restart_scalars(&phi.internal))?;
        gpu.write(&mut seed_phi.bf, &from_restart_scalars(&phi.boundary))?;
        println!("phi loaded from the restart checkpoint - not re-derived from U");
    } else {
        let source = establish_flux(
            &gpu,
            &mut seed_phi,
            &mut seed_u,
            &hm,
            &mesh,
            &t_dir,
            o,
            &phi_ctrl,
            u_at_rest,
        )?;
        report_flux(&source);
    }

    println!(
        "starting max |sum_f phi| per cell = {}   (0 means the flux is discretely \
         conservative)",
        g(f64::from(max_div_phi(&gpu, &seed_phi, &hm)?))
    );

    // ---- the coupled system ------------------------------------------------
    let mut s = Simple::new(&gpu, &hm, &mesh, simple_ctrl, buoy)?;

    // residualControl stops the PIMPLE outer loop early on the INITIAL
    // residuals - but only when there ARE initial residuals. In `-fixedIters`
    // mode the linear solvers report nothing, every residual reads back as
    // zero, and a control tested against zero would declare the very first
    // outer corrector converged and stop. Off is the honest state there: the
    // run does exactly the work it was told to.
    if o.fixed_iters <= 0 {
        s.set_residual_control(cc.residual_control.clone());
    }

    setup_vector_field(&gpu, s.u_mut(), &raw_u, &hm)?;
    setup_scalar_field(&gpu, s.p_mut(), &raw_p, &hm)?;

    {
        // `seed_u` still holds the file's velocity unless potential flow
        // replaced it, so this copy is a no-op on a restart and the initial
        // guess on a cold start.
        let u_int = gpu.download(&seed_u.f)?;
        gpu.write(&mut s.u_mut().f, &u_int)?;
        let phi_f = gpu.download(&seed_phi.f)?;
        let phi_bf = gpu.download(&seed_phi.bf)?;
        gpu.write(&mut s.phi_mut().f, &phi_f)?;
        gpu.write(&mut s.phi_mut().bf, &phi_bf)?;
    }

    if let Some(rd) = &restart_data {
        // Overwrite the INTERNAL cell values with the restart's exact
        // numbers - `raw_u`/`raw_p` only gave the boundary condition TYPES
        // and the (irrelevant, since overwritten) start-time values.
        let u = find_restart_field(rd, "U")?;
        gpu.write(&mut s.u_mut().f, &from_restart_vectors(&u.internal))?;
        let p = find_restart_field(rd, "p")?;
        gpu.write(&mut s.p_mut().f, &from_restart_scalars(&p.internal))?;
    }

    s.initialise(&gpu)?;

    if let Some(rd) = &restart_data {
        // `initialise` re-derives every boundary cell generically
        // (`correct_boundary_conditions[_vector]`), which assumes a cold
        // start and does not first re-run the inlet/outlet direction switch
        // - see `ofgpu-vof`'s identical fix and `Vof::alpha_phi_mut`'s doc
        // for the failure this closes. The checkpoint's own boundary values
        // do not have that problem, so they overwrite whatever `initialise`
        // just computed.
        let u = find_restart_field(rd, "U")?;
        gpu.write(&mut s.u_mut().bf, &from_restart_vectors(&u.boundary))?;
        let p = find_restart_field(rd, "p")?;
        gpu.write(&mut s.p_mut().bf, &from_restart_scalars(&p.boundary))?;
    }

    if s.pressure_is_pinned() {
        println!(
            "the pressure has no fixedValue anywhere, so cell 0 is pinned to zero: \
             only its GRADIENTS mean anything in what follows"
        );
    }

    // SPEC-LIT 15.5: nut's own patch types decide nu_t's wall treatment.
    let raw_nut_for_walls = {
        let p = t_dir.join("nut");
        if p.exists() {
            Some(read_scalar_field(&p, hm.n_cells)?)
        } else {
            None
        }
    };
    let wf_faces = WallFaces::from_case(&raw_e, raw_nut_for_walls.as_ref(), &hm)?;
    let roughness = NutRoughness::from_case(raw_nut_for_walls.as_ref(), &hm)?;

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

    setup_scalar_field(&gpu, model.k_mut(), &raw_k, &hm)?;
    setup_scalar_field(&gpu, model.epsilon_mut(), &raw_e, &hm)?;

    let nut_path = t_dir.join("nut");
    if nut_path.exists() {
        let raw_nut = read_scalar_field(&nut_path, hm.n_cells)?;
        setup_scalar_field(&gpu, model.nut_mut(), &raw_nut, &hm)?;
    }


    // SPEC-LIT 17: the buoyancy production. A k-epsilon run on a 1173 K plume
    // in 293 K air with no G_b is missing a leading-order term - buoyancy is
    // where most of that flow's turbulence comes from, and the stratification
    // above the fire is where the rest of it is destroyed.
    //
    //     G_b = (nu_t/Pr_t) g . grad(T) / T
    //
    // Pr_t is the SAME turbulent Prandtl number the temperature equation
    // diffuses with, read from the case once: two different values of one
    // constant in one run is exactly the kind of quiet inconsistency
    // SPEC-LIT 15.6 is about.
    if buoy.is_active() {
        model.set_buoyancy(BuoyancyProduction {
            g: buoy.g,
            prt: t_coeffs.prt,
            // The Henkes form, which SPEC-LIT 17 marks *DESIGN* and defaults
            // to. `RAS { C3Buoyancy <number>; }` overrides it with a constant,
            // 0 being the other convention section 17 names.
            //
            // NOT spelled `C3`: the standard k-epsilon model of SPEC-LIT 6.1
            // already has a constant of that name - the coefficient of the
            // DILATATION term (2/3 C_1 - C_3) div(u) - and two different
            // constants answering to one key is precisely the silent
            // substitution SPEC-LIT 15.6 and 13.4 are about.
            c3: match ofgpu::io::case::model_coeff(&cc, "C3Buoyancy", Scalar::NAN) {
                v if v.is_nan() => C3Mode::Henkes,
                v => C3Mode::Constant(v),
            },
            ..BuoyancyProduction::default()
        })?;
        println!(
            "turbulence buoyancy: G_b = (nut/Prt) g.grad(T)/T, Prt {}, {}",
            g(f64::from(t_coeffs.prt)),
            model
                .buoyancy()
                .map(|b| b.c3.describe())
                .unwrap_or_default()
        );
        println!(
            "  stable stratification gives G_b < 0 (destroys k); above a heat \
             source G_b > 0 (makes it)"
        );
    } else {
        println!("turbulence buoyancy: no gravity in the case, so G_b is identically zero");
    }

    let mut heat = ScalarTransport::new(&gpu, &hm, &mesh, "T", t_coeffs, t_ctrl)?;
    // `div(phi,T)`, by its own name - see `read_simple_controls`.
    heat.set_convection(cc.schemes.div("div(phi,T)")?);
    setup_scalar_field(&gpu, heat.field_mut(), &raw_t, &hm)?;

    update_inlet_outlet(&gpu, model.k_mut(), s.phi(), &hm)?;
    update_inlet_outlet(&gpu, model.epsilon_mut(), s.phi(), &hm)?;
    update_inlet_outlet(&gpu, heat.field_mut(), s.phi(), &hm)?;

    {
        let flow = s.flow_state();
        model.initialise(&gpu, &flow)?;
    }
    heat.initialise(&gpu)?;

    if let Some(rd) = &restart_data {
        // Same reasoning as `U`/`p` above: the internal values are the
        // restart's exact numbers, and the boundary is restored AFTER
        // `initialise` so a flow-direction-dependent condition is not left
        // evaluating the wrong branch.
        let k = find_restart_field(rd, "k")?;
        gpu.write(&mut model.k_mut().f, &from_restart_scalars(&k.internal))?;
        gpu.write(&mut model.k_mut().bf, &from_restart_scalars(&k.boundary))?;
        let e = find_restart_field(rd, "epsilon")?;
        gpu.write(&mut model.epsilon_mut().f, &from_restart_scalars(&e.internal))?;
        gpu.write(&mut model.epsilon_mut().bf, &from_restart_scalars(&e.boundary))?;
        let t_field = find_restart_field(rd, "T")?;
        gpu.write(&mut heat.field_mut().f, &from_restart_scalars(&t_field.internal))?;
        gpu.write(&mut heat.field_mut().bf, &from_restart_scalars(&t_field.boundary))?;
    }

    // ---- volumetric sources (SPEC-LIT 18) ---------------------------------
    //
    // A fire is a HEAT RELEASE, not only a hot inlet. Until `constant/fvSources`
    // existed there was no way to put a watt into any equation in this solver,
    // so the only way to model one was to blow hot gas in through a patch - and
    // that prescribes a mass flux the fire does not have.
    //
    // Each source names the equation it acts on. A `type` this solver cannot
    // apply, or a zone that selects no cells, is an ERROR here rather than a
    // line quietly skipped (SPEC-LIT 13.4).
    {
        let specs = ofgpu::sources::read_sources(&o.case_dir)?;
        if specs.is_empty() {
            println!("sources: none (no constant/fvSources)");
        }
        for spec in &specs {
            let src = spec.build(&gpu, &hm)?;
            println!("source on {}: {}", spec.field, src.describe());
            match spec.field.as_str() {
                "T" => heat.sources_mut().push(src),
                "U" => s.momentum_mut().sources_mut().push(src),
                other => {
                    return Err(ofgpu::Error::Config(format!(
                        "source \"{}\": field \"{other}\" - this driver solves \
                         for U, p and T, so a source on anything else has no \
                         equation to act on (SPEC-LIT 18)",
                        spec.name
                    )))
                }
            }
        }
    }

    // ---- the pressure backend ----------------------------------------------
    //
    // One SIMPLE iteration first, and it is not thrown away: the selector has
    // to measure on the real assembled matrix, and the matrix only exists once
    // something has assembled it. The state it leaves behind is a legitimate
    // starting state - it is simply the run's first iteration, performed with
    // the fallback backend before the fast one had been chosen.
    println!("\nassembling the pressure system (one SIMPLE iteration with the fallback)");
    {
        let mut boot = PbicgstabBackend::new(simple_ctrl.p_solver);
        boot.setup(&gpu, &hm, &mesh, &SystemProbe::default())?;
        s.correct(&gpu, &mut boot, model.nut(), heat.field())?;
    }

    let mut backend = pick_backend(&gpu, &hm, &mesh, &s, o.backend, simple_ctrl.p_solver)?;

    let writer = FieldWriter {
        case_dir: &o.case_dir,
        raw_u: &raw_u,
        raw_p: &raw_p,
        raw_k: &raw_k,
        raw_e: &raw_e,
        raw_t: &raw_t,
        output: &o.output,
    };

    // ---- what will actually run --------------------------------------------
    ofgpu::io::case::print_effective_settings(&cc);

    // ---- the schedule -------------------------------------------------------
    let dt = f64::from(cc.turb.delta_t);
    // `-endTime` names the ABSOLUTE time to reach, so a restart's own `t0`
    // comes off the step count it asks for - see `Schedule::t0`'s doc.
    let t0 = restart_data.as_ref().map_or(0.0, |d| d.time);
    let n_steps = if transient {
        (((o.end_time - t0) / dt).round() as i64).max(1)
    } else {
        cc.turb.n_outer_iterations.max(1) as i64
    };

    if transient && o.write_interval > 0.0 && o.write_interval < dt {
        eprintln!(
            "[ofgpu-buoyant] -writeInterval {} is shorter than -deltaT {}; every step \
             will be a write time",
            g(o.write_interval),
            g(dt)
        );
    }

    let sim_end = t0 + n_steps as f64 * dt;
    let sched = Schedule {
        transient,
        dt,
        n_steps,
        write_interval: o.write_interval,
        outer_iters: o.outer_iters,
        n_correctors: o.n_correctors,
        check_every: o.check_every,
        convergence_tol: cc.turb.convergence_tol,
        residual_control: cc.residual_control.clone(),
        residuals_read: o.fixed_iters <= 0,
        do_write: o.do_write,
        graph: o.graph,
        probe_layers: o.probe_layers,
        final_name: if !o.write_time.is_empty() {
            o.write_time.clone()
        } else if transient {
            format_time_name(sim_end as Scalar)
        } else {
            cc.write_time.clone()
        },
        restart_write: o.restart_write,
        mesh_hash,
        t0,
    };

    if transient {
        println!(
            "\ntransient: endTime {} s, deltaT {} s -> {} steps x {} pass(es) of \
             [{} x SIMPLE + k-epsilon + T], writing {}",
            g(sim_end),
            g(dt),
            n_steps,
            sched.outer_iters,
            sched.n_correctors,
            if sched.write_interval > 0.0 {
                format!("every {} s", g(sched.write_interval))
            } else {
                "at the end time only".to_string()
            }
        );
        println!(
            "  ddt Euler implicit, 1/dt = {} | relax U {} p {} k {} epsilon {} T {}{}",
            g(f64::from(cc.turb.r_delta_t())),
            g(f64::from(simple_ctrl.momentum.u_relax)),
            g(f64::from(simple_ctrl.p_relax)),
            g(f64::from(cc.turb.k_relax)),
            g(f64::from(cc.turb.eps_relax)),
            g(f64::from(t_ctrl.k_relax)),
            if cc.turb.k_solver.fixed_iters {
                " | fixed-iteration solvers: zero host transfers"
            } else {
                ""
            }
        );
        if (sim_end - o.end_time).abs() > 1e-9 * dt.max(1.0) {
            println!(
                "  endTime rounded to a whole number of steps: {} s asked for, {} s run",
                g(o.end_time),
                g(sim_end)
            );
        }
    } else {
        println!(
            "\nsteady: {} {} outer iterations of {} x [{} x SIMPLE + k-epsilon + T]{}",
            if sched.residuals_read { "up to" } else { "exactly" },
            n_steps,
            sched.outer_iters,
            sched.n_correctors,
            if sched.residuals_read {
                format!(
                    ", stopping when every residual is below {}",
                    g(f64::from(sched.convergence_tol))
                )
            } else {
                " | fixed-iteration solvers: no residual is read, so none can stop the run"
                    .to_string()
            }
        );
    }
    println!();

    let rep = run_loop(
        &gpu,
        Fields { s: &mut s, model: &mut model, heat: &mut heat, backend: &mut *backend },
        &hm,
        &sched,
        &writer,
        up,
        buoy.t_ref,
    )?;

    // ---- what the plume did -------------------------------------------------
    let temps = gpu.download(&heat.field().f)?;
    let stats = weighted_stats(&temps, &hm.v)?;
    println!(
        "\nT: min {}  max {}  volume-weighted mean {}",
        g(f64::from(stats.min)),
        g(f64::from(stats.max)),
        g(f64::from(stats.mean))
    );

    // The physics check, printed whether or not `-probe` was asked for: a run
    // that produced no stratification has produced no plume, and that is worth
    // one read-back at the very end even when nobody asked for the history.
    let l = stratification(&temps, &hm, up, buoy.t_ref)?;
    print_layers(&l, up);
    print_verdict(&l);

    print_summary(&rep, &sched, t_total.elapsed().as_secs_f64());

    if !sched.transient && !rep.converged && sched.residuals_read {
        println!(
            "\nnot converged in {} iterations; the residuals above are where it stopped",
            rep.steps
        );
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
//  Tests
// ==========================================================================

/// Named `buoyant_tests` rather than `tests` because `common/mod.rs` is
/// included by `#[path]` and already contributes a `tests` module to this
/// crate.
#[cfg(test)]
mod buoyant_tests {
    use super::*;
    use ofgpu::mesh::PatchInfo;

    fn argv(v: &[&str]) -> Vec<String> {
        std::iter::once("ofgpu-buoyant".to_string())
            .chain(v.iter().map(|s| (*s).to_string()))
            .collect()
    }

    #[test]
    fn defaults_leave_the_steady_path_untouched() {
        let o = parse(&argv(&["case"])).expect("a bare case directory is a valid command line");

        assert!(o.end_time <= 0.0, "no -endTime must mean steady");
        assert_eq!(o.n_correctors, DEFAULT_N_CORRECTORS);
        assert_eq!(o.outer_iters, DEFAULT_OUTER_ITERS);
        assert_eq!(o.backend, BackendPick::Auto);
        assert!(!o.probe_layers);
        assert!(o.do_write);
        assert!(!o.graph);
    }

    #[test]
    fn the_backend_flag_takes_four_names_and_nothing_else() {
        assert_eq!(BackendPick::from_name("auto").expect("auto"), BackendPick::Auto);
        assert_eq!(
            BackendPick::from_name("PBiCGStab").expect("case insensitive"),
            BackendPick::Pbicgstab
        );
        assert_eq!(BackendPick::from_name("fft").expect("fft"), BackendPick::Fft);
        assert_eq!(BackendPick::from_name("amgx").expect("amgx"), BackendPick::Amgx);
        assert!(BackendPick::from_name("gamg").is_err());
        assert!(BackendPick::from_name("").is_err());
    }

    #[test]
    fn flags_are_rejected_before_they_reach_the_gpu() {
        assert!(parse(&argv(&["case", "-endTime", "2", "-deltaT", "0.01"])).is_ok());
        assert!(parse(&argv(&["case", "-endTime", "0"])).is_err());
        assert!(parse(&argv(&["case", "-deltaT", "0"])).is_err());
        assert!(parse(&argv(&["case", "-writeInterval", "0"])).is_err());
        assert!(parse(&argv(&["case", "-outerIters", "0"])).is_err());
        assert!(parse(&argv(&["case", "-nCorrectors", "0"])).is_err());
        assert!(parse(&argv(&["case", "-backend", "nope"])).is_err());

        // A graph cannot capture a read-back, and every backend but PBiCGStab
        // performs one inside solve().
        assert!(parse(&argv(&["case", "-graph"])).is_err());
        assert!(parse(&argv(&["case", "-graph", "-fixedIters", "8"])).is_err());
        assert!(parse(&argv(&[
            "case",
            "-graph",
            "-fixedIters",
            "8",
            "-backend",
            "pbicgstab"
        ]))
        .is_ok());
    }

    #[test]
    fn a_time_flag_is_a_number_or_an_error() {
        assert!(parse_time("-deltaT", "0.001").is_ok());
        assert!(parse_time("-endTime", "1e-3").is_ok());
        assert!(parse_time("-deltaT", "").is_err());
        assert!(parse_time("-deltaT", "abc").is_err());
        assert!(parse_time("-deltaT", "nan").is_err());
        assert!(parse_time("-deltaT", "inf").is_err());
    }

    #[test]
    fn write_times_land_on_multiples_of_the_interval() {
        let dt = 0.02;
        let w = 0.5;

        assert!(is_write_time(25.0 * dt, w, dt));
        assert!(!is_write_time(24.0 * dt, w, dt));
        assert!(!is_write_time(26.0 * dt, w, dt));
        assert!(!is_write_time(0.0, w, dt), "t = 0 is the start, never a write");
        assert!(!is_write_time(25.0 * dt, 0.0, dt));
    }

    /// A column of cells, hot at the top, so the probe has a mesh with a known
    /// answer. Only the four fields `stratification` reads are filled in.
    fn column(n: usize) -> HostMesh {
        let mut m = HostMesh { n_cells: n, ..Default::default() };
        m.v = vec![1.0; n];
        m.c = (0..n)
            .map(|k| Vec3::new(0.5, 0.5, k as Scalar + 0.5))
            .collect();
        m.patches = vec![PatchInfo {
            name: "sides".into(),
            type_name: "wall".into(),
            kind: ofgpu::PatchKind::Wall,
            start: 0,
            size: 0,
            nbr_patch: None,
        }];
        m
    }

    /// The measurement the whole driver exists to make: hot on top reads
    /// positive, hot on the bottom reads negative, and uniform reads zero.
    #[test]
    fn the_probe_reports_which_third_is_hotter() {
        let up = Vec3::new(0.0, 0.0, 1.0);
        let t_ref: Scalar = 293.15;
        let hm = column(9);

        // Thirds by cell centre: z in [0.5, 1.5] is lower, [7.5, 8.5] upper.
        let mut hot_top = vec![293.15 as Scalar; 9];
        hot_top[8] = 600.0;
        let l = stratification(&hot_top, &hm, up, t_ref).expect("probe");
        assert!(l.contrast() > 0.0, "hot at the top must read positive: {l:?}");

        let mut hot_bottom = vec![293.15 as Scalar; 9];
        hot_bottom[0] = 600.0;
        let l = stratification(&hot_bottom, &hm, up, t_ref).expect("probe");
        assert!(l.contrast() < 0.0, "hot at the bottom must read negative: {l:?}");

        let uniform = vec![400.0 as Scalar; 9];
        let l = stratification(&uniform, &hm, up, t_ref).expect("probe");
        assert_eq!(l.contrast(), 0.0, "a uniform field is not stratified");
        assert_eq!(l.upper, 400.0);
    }

    /// The weighting is by MASS, not by volume, and at 3.5x the reference
    /// temperature the difference is not academic: a cell at 1000 K carries
    /// less than a third of the mass of one at 293 K.
    #[test]
    fn the_probe_weights_by_mass_and_not_by_volume() {
        let up = Vec3::new(0.0, 0.0, 1.0);
        let t_ref: Scalar = 293.15;
        let hm = column(6);

        // Upper third is cells 4 and 5: one at 1000 K, one at TRef.
        let mut t = vec![t_ref; 6];
        t[5] = 1000.0;

        let l = stratification(&t, &hm, up, t_ref).expect("probe");

        // Volume weighting would give the plain average.
        let by_volume = 0.5 * (t_ref + 1000.0);
        let w_hot = t_ref / 1000.0;
        let by_mass = (t_ref + w_hot * 1000.0) / (1.0 + w_hot);

        assert!(
            (l.upper - by_mass).abs() < 1e-9,
            "expected the mass-weighted {by_mass}, got {}",
            l.upper
        );
        assert!(
            (l.upper - by_volume).abs() > 100.0,
            "mass and volume weighting must differ here, or the test proves nothing"
        );
    }

    /// Gravity decides which way is up, not the z axis. A case lying on its
    /// side must still be probed along its own vertical.
    #[test]
    fn the_probe_follows_gravity_rather_than_the_z_axis() {
        let t_ref: Scalar = 293.15;
        let mut hm = column(9);
        // Lay the column along +x instead.
        hm.c = (0..9)
            .map(|i| Vec3::new(i as Scalar + 0.5, 0.5, 0.5))
            .collect();

        let mut t = vec![t_ref; 9];
        t[8] = 600.0;

        let along_x = stratification(&t, &hm, Vec3::new(1.0, 0.0, 0.0), t_ref).expect("probe");
        assert!(along_x.contrast() > 0.0);

        // Probed along z the whole column is one layer, so both thirds contain
        // every cell and the contrast vanishes - the honest answer for a mesh
        // with no extent along the probe direction, rather than a NaN or an
        // abort.
        let along_z = stratification(&t, &hm, Vec3::new(0.0, 0.0, 1.0), t_ref).expect("probe");
        assert_eq!(along_z.contrast(), 0.0);
        assert_eq!(along_z.upper, along_z.lower);
    }

    #[test]
    fn a_mismatched_field_is_refused_rather_than_indexed() {
        let hm = column(4);
        assert!(stratification(&[300.0, 300.0], &hm, Vec3::new(0.0, 0.0, 1.0), 293.15).is_err());
        assert!(stratification(&[], &column(0), Vec3::new(0.0, 0.0, 1.0), 293.15).is_err());
    }
}
