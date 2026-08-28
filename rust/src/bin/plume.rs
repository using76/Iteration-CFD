// meteor-cfd - Copyright (c) 2026 주식회사 메테오시뮬레이션 (Meteo Simulation Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

//! `ofgpu-plume` - k-epsilon and a transported temperature, together.
//!
//! ```text
//! ofgpu-plume <caseDir> [options]
//!
//!   -iters N          outer iterations (default: from controlDict endTime)
//!   -fixedIters N     run every linear solver for exactly N sweeps and never
//!                     read a residual, so the time loop performs ZERO host
//!                     transfers of any kind
//!   -check N          test the convergence measure every N iterations
//!   -write NAME       time directory to write into (default: from controlDict)
//!   -noWrite          do not write fields; useful when timing
//!   -graph            capture ONE unit of work as a CUDA graph and replay it;
//!                     requires -fixedIters
//!   -noPotential      do not solve for the flux; fall back to the old
//!                     interpolate(U) & Sf, which does NOT conserve mass
//!   -inletPatch NAME  patch the flow enters through (default: inlet)
//!   -outletPatch NAME patch it leaves through (default: outlet)
//!
//!   transient mode - additions. Without -endTime none of them is read and the
//!   driver behaves exactly as it always has:
//!
//!   -endTime T        physical end time in seconds. Switches ddt from
//!                     steadyState to Euler implicit, so the run is a real
//!                     transient rather than a relaxed march to steady state
//!   -deltaT dt        time step in seconds (default: controlDict deltaT)
//!   -writeInterval W  write every W seconds of PHYSICAL time
//!                     (default: one write, at the end time)
//!   -outerIters N     outer iterations per time step (default 2)
//! ```
//!
//! Like `ofgpu-k-epsilon` this holds the velocity field frozen. What it adds is
//! a passive scalar riding that flow:
//!
//! ```text
//! d(T)/dt + div(phi, T) - laplacian(nu/Pr + nut/Prt, T) = 0
//! ```
//!
//! `nut` is whatever k-epsilon produced this iteration, so the temperature sees
//! the turbulent mixing the model predicts - which is the entire point of
//! running the two together rather than transporting `T` on a laminar
//! diffusivity.
//!
//! # Where the flux comes from
//!
//! `T` rides `phi`, so `phi` had better conserve mass: if more enters a cell
//! than leaves it, the transport equation has a source nobody wrote down. A
//! `phi` on disk was written by a solver that satisfied discrete continuity
//! and always wins. Failing that, this driver SOLVES for one - potential flow,
//! `ofgpu::potential_flow` - rather than interpolating the cell-centred `U` in
//! `0/U` onto the faces. The old interpolation is still reachable with
//! `-noPotential`, and the `max |sum_f phi|` line printed at start-up is the
//! difference between the two: 1e-18 against 1e-2 on the plume case.
//!
//! One outer iteration is `KEpsilon::correct` followed by
//! `ScalarTransport::correct`, and neither touches the host. `-graph` exploits
//! that: it records the work once and replays the recording, so the driver
//! submits one graph launch per unit instead of some ninety kernel launches.
//! See `src/bin/graph_bench.rs`, which measures what that is worth.
//!
//! # Steady and transient are the same loop with a different `ddt`
//!
//! Nothing in the models branches on the regime. `TurbulenceControls::steady`
//! makes `r_delta_t()` return zero, `fvm_ddt_euler` then writes nothing at all,
//! and the run is `simpleFoam`'s: under-relaxation is the only thing limiting
//! how far a "step" moves. Clearing `steady` and setting `delta_t` is therefore
//! the whole of transient mode - see `io/case.rs`, which sets the same two
//! fields from `ddtSchemes` and `controlDict` when the case asks for it. This
//! driver only lets the command line say so as well.
//!
//! The unit of work is then a *time step* rather than an iteration, and one
//! time step is `outerIters` passes of (k-epsilon, then T).
//!
//! ## What an outer iteration does to the old-time level
//!
//! `KEpsilon::correct` and `ScalarTransport::correct` each call
//! `store_old_time` on entry, so `psi.f0` is refreshed by *every* pass, not
//! once per time step. With `-outerIters 1` that is exactly Euler implicit.
//! With more than one pass each pass differences against the previous pass
//! rather than against the start of the step, so a step of `n` passes advances
//! the transported fields like `n` Euler sub-steps of `dt`. It is stable and it
//! reaches the same steady state, but it is not the PIMPLE semantics of
//! one time level per step, and a run that needs those should use
//! `-outerIters 1` with a correspondingly smaller `-deltaT`. Fixing it properly
//! means lifting `store_old_time` out of `correct` into a new
//! `store_time_level` on both models, which is a library change and not this
//! driver's to make.
//!
//! Provenance: ORIGINAL driver code over LITERATURE numerics. The buoyancy
//! treatment and the transport equations it drives are cited in the library
//! modules and SPEC-LIT.md; this file is argument parsing, case loading, the
//! iteration order and the reporting loop, which are this project's own
//! (`PROVENANCE.md`, `src/bin/*`). No GPL-licensed source was consulted.

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use ofgpu::error::IoContext;
use ofgpu::field::{GpuSurfaceScalarField, GpuVectorField};
use ofgpu::field_ops::{correct_boundary_conditions_vector, FieldKernels};
use ofgpu::field_setup::{
    compute_phi_from_u, harvest_scalar_field, harvest_surface_scalar_field, max_div_phi,
    setup_scalar_field, setup_vector_field,
    update_inlet_outlet, wall_coeffs_from_case, NutRoughness, WallFaces,
};
use ofgpu::io::case::{
    find_start_time, format_time_name, model_coeff, read_case_controls, CaseControls,
    SolverControls, TurbulenceControls,
};
use ofgpu::io::dict::FoamDict;
use ofgpu::io::fields::{read_scalar_field, read_vector_field, RawScalarField};
use ofgpu::io::polymesh::{build_host_mesh, read_poly_mesh};
use ofgpu::models::{KEpsilon, KEpsilonCoeffs};
use ofgpu::turbulence::{BuoyancyProduction, C3Mode};
use ofgpu::potential_flow::{
    mean_inflow_speed, solve_potential_flow, PotentialFlowResult, PotentialFlowSpec,
};
use ofgpu::scalar_transport::{weighted_stats, ScalarTransport, ScalarTransportCoeffs};
use ofgpu::turbulence::FlowState;
use ofgpu::{Error, Gpu, GpuMesh, Graph, HostMesh, Label, Result, Scalar};

#[path = "common/mod.rs"]
mod common;

use common::{atoi, device_banner, g, next_arg, sci};

/// Units of work run the ordinary way before a graph is captured.
///
/// The very first call of each kernel pays for module loading and the first
/// solve pays for the preconditioner setup; capturing either would bake a
/// one-off into a recording that is replayed thousands of times.
const GRAPH_WARMUP: i64 = 5;

/// Outer iterations per time step when the command line does not say.
const DEFAULT_OUTER_ITERS: Label = 2;

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

    /// Fall back to `interpolate(U) & Sf` instead of solving for the flux.
    /// Only useful for reproducing the non-conservative behaviour on purpose.
    no_potential: bool,
    inlet_patch: String,
    outlet_patch: String,
}

fn usage() {
    eprintln!(
        "usage: ofgpu-plume <caseDir> [-iters N] [-fixedIters N] [-check N] \
         [-write NAME] [-noWrite] [-graph]\n       \
         [-endTime T] [-deltaT dt] [-writeInterval W] [-outerIters N]
                [-noPotential] [-inletPatch NAME] [-outletPatch NAME] [-permissive]"
    );
}

/// A physical time from the command line.
///
/// Deliberately stricter than [`atoi`], which every other flag uses: `-deltaT
/// 0.001` read by an integer parser is zero, and a zero timestep does not fail
/// - it makes `1/dt` infinite and fills every matrix with NaNs several hundred
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
        no_potential: false,
        inlet_patch: "inlet".to_string(),
        outlet_patch: "outlet".to_string(),
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
            "-noPotential" => o.no_potential = true,
            "-inletPatch" => o.inlet_patch = next_arg(args, &mut i)?,
            "-outletPatch" => o.outlet_patch = next_arg(args, &mut i)?,
            other => {
                usage();
                return Err(Error::Config(format!("unknown option {other}")));
            }
        }
        i += 1;
    }

    // Caught here rather than left to the driver: a capture containing a
    // read-back fails somewhere inside cudarc with a message about stream
    // capture state, which tells the user nothing about what they typed. Still
    // required in transient mode - the capture is bigger there, not different.
    if o.graph && o.fixed_iters <= 0 {
        usage();
        return Err(Error::Config(
            "-graph needs -fixedIters N: a CUDA graph cannot capture the adaptive \
             solver's convergence read-back, and -fixedIters is what removes it"
                .to_string(),
        ));
    }

    if o.outer_iters < 1 {
        usage();
        return Err(Error::Config(format!(
            "-outerIters must be at least 1, got {}",
            o.outer_iters
        )));
    }

    // A non-positive end time is the sentinel for "steady", so a user who
    // typed one meaning something else is told rather than silently handed a
    // steady run.
    let given = |flag: &str| args.iter().any(|a| a == flag);

    if o.end_time <= 0.0 && given("-endTime") {
        usage();
        return Err(Error::Config(format!(
            "-endTime must be positive, got {}",
            o.end_time
        )));
    }
    if o.delta_t <= 0.0 && given("-deltaT") {
        usage();
        return Err(Error::Config(format!(
            "-deltaT must be positive, got {}",
            o.delta_t
        )));
    }
    if o.write_interval <= 0.0 && given("-writeInterval") {
        usage();
        return Err(Error::Config(format!(
            "-writeInterval must be positive, got {}",
            o.write_interval
        )));
    }

    if o.end_time <= 0.0 && (given("-deltaT") || given("-writeInterval")) {
        eprintln!(
            "[ofgpu-plume] -deltaT and -writeInterval do nothing without -endTime; \
             this run is steady"
        );
    }

    Ok(o)
}

/// Where `phi` came from. Printed, because the three differ by orders of
/// magnitude in how well they conserve mass and a reader has to be able to
/// tell which one produced the numbers below.
enum FluxSource {
    /// Read from the time directory. Written by a solver that satisfied
    /// discrete continuity, so it always wins.
    File,
    /// Solved for: `laplacian(Phi) = 0`, flux read straight off the operator.
    Potential(PotentialFlowResult),
    /// `interpolate(U) & Sf`. Conservative only by accident.
    Interpolated,
}

/// Establish `phi`, and with it the `U` that goes with it.
///
/// The order of preference is the order of trustworthiness:
///
/// 1. a `phi` in the time directory - it came out of a solver that satisfied
///    discrete continuity, and nothing here can improve on that;
/// 2. potential flow, which SOLVES for a conservative flux and reconstructs
///    the cell velocity from it;
/// 3. `interpolate(U) & Sf`, which is what `-noPotential` selects and what
///    this driver used to do unconditionally. Nothing constrains the `U` in a
///    `0/` directory to satisfy discrete continuity, so nothing constrains its
///    interpolant to either; on the plume case it leaves `max |sum_f phi|` at
///    1.4e-2 m^3/s, mass enters at the burner and never leaves, and `T`
///    equilibrates to the inlet value because there is no mean transport to
///    carry heat out. `bounded` hides that rather than fixing it.
///
/// Case 2 overwrites `u`'s internal field. That is the point - the velocity
/// has to be the one the flux implies, or `grad(U)` and the flux describe
/// different flows.
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
) -> Result<FluxSource> {
    let path = t_dir.join("phi");

    if path.exists() {
        load_phi(gpu, phi, hm, &path)?;
        return Ok(FluxSource::File);
    }

    // A `None` here means the case cannot be solved for and the reason has
    // already been printed; fall through rather than abort, so a case this
    // driver was never meant for still runs.
    if !o.no_potential {
        if let Some(r) = potential_flux(gpu, phi, u, hm, mesh, o, ctrl)? {
            return Ok(FluxSource::Potential(r));
        }
    }

    compute_phi_from_u(gpu, phi, u, hm)?;
    println!("phi reconstructed as interpolate(U) & Sf - NOT conservative");
    Ok(FluxSource::Interpolated)
}

/// Solve for the flux, or explain why this case cannot be solved for and
/// return `None`.
///
/// The inflow speed comes from the case's own evaluated `0/U` on the inlet
/// patch rather than from a constant here, so the burner has one description
/// and not two that can drift apart.
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
            println!("no patch named \"{name}\", so the flux cannot be solved for");
            println!("  pass -inletPatch/-outletPatch to name the openings, or -noPotential");
            return Ok(None);
        }
    }

    let u_in = mean_inflow_speed(&gpu.download(&u.bf)?, hm, &o.inlet_patch)?;

    if !u_in.is_finite() || u_in <= 0.0 {
        println!(
            "the {} patch carries no inflow (mean normal velocity {}), so there is nothing for potential flow to distribute",
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

/// `fvSolution`'s `solvers/Phi`, for the one-off potential-flow solve.
///
/// Its own entry rather than a borrowed one: the Laplace matrix is symmetric
/// and much stiffer than any of the transport equations, so the turbulence
/// solver's `relTol 0.01` would stop it long before the flux was conservative,
/// which is the only thing this solve is for. The defaults are therefore a
/// tight absolute tolerance and a generous iteration budget, both paid once.
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
    let d = FoamDict::read(&p)?;

    ofgpu::io::case::read_solver_controls(&mut sc, &d, "Phi")?;

    Ok(sc)
}

/// The start-up log line for whichever flux the run ended up with.
///
/// The inlet and outlet fluxes are the ones that matter: both are signed
/// OUTWARD, so a conservative flux makes them cancel, and their sum is the
/// mass the domain is losing or inventing per second.
fn report_flux(source: &FluxSource) {
    match source {
        FluxSource::File => {}
        FluxSource::Interpolated => {}
        FluxSource::Potential(r) => {
            println!(
                "phi solved as potential flow: laplacian(Phi) = 0 in {} iterations, residual {}",
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

    ofgpu::io::case::read_solver_controls(&mut t.k_solver, &d, "T")?;

    t.k_relax = d.scalar("relaxationFactors/equations/T", t.k_relax);

    Ok(t)
}

/// Every linear solver switched into the transfer-free mode.
fn make_fixed(sc: &mut SolverControls, sweeps: Label) {
    sc.fixed_iters = true;
    sc.max_iter = sweeps;
    sc.report_residuals = false;
}

// ==========================================================================
//  Writing a time directory
// ==========================================================================

/// Where results go, and the three input fields whose boundary *type* strings
/// every written directory has to carry.
///
/// Bundled because a transient run writes many directories and every one of
/// them must be seeded from the same originals: `harvest_scalar_field` only
/// invents a type where none is set, so a directory written without the seeds
/// comes out `calculated` everywhere and cannot start another run.
struct FieldWriter<'a> {
    case_dir: &'a Path,
    raw_k: &'a RawScalarField,
    raw_e: &'a RawScalarField,
    raw_t: &'a RawScalarField,
}

impl FieldWriter<'_> {
    fn write(
        &self,
        gpu: &Gpu,
        name: &str,
        model: &KEpsilon<'_>,
        heat: &ScalarTransport<'_>,
        phi: &GpuSurfaceScalarField,
        hm: &HostMesh,
    ) -> Result<PathBuf> {
        let out_dir = self.case_dir.join(name);
        std::fs::create_dir_all(&out_dir).path(&out_dir)?;

        let mut out_k = seed_types(self.raw_k);
        let mut out_e = seed_types(self.raw_e);
        let mut out_t = seed_types(self.raw_t);
        let mut out_nut = RawScalarField::default();
        let mut out_phi = RawScalarField::default();

        harvest_scalar_field(gpu, &mut out_k, model.k(), hm)?;
        harvest_scalar_field(gpu, &mut out_e, model.epsilon(), hm)?;
        harvest_scalar_field(gpu, &mut out_t, heat.field(), hm)?;
        harvest_scalar_field(gpu, &mut out_nut, model.nut(), hm)?;
        harvest_surface_scalar_field(gpu, &mut out_phi, phi, hm)?;

        out_k.dimensions = "[0 2 -2 0 0 0 0]".to_string();
        out_e.dimensions = "[0 2 -3 0 0 0 0]".to_string();
        out_nut.dimensions = "[0 2 -1 0 0 0 0]".to_string();
        // Kelvin. Keep whatever the input said if it said anything, so a case
        // transporting a dimensionless tracer under the name T is not
        // relabelled behind the user's back.
        out_t.dimensions = if self.raw_t.dimensions.is_empty() {
            "[0 0 0 1 0 0 0]".to_string()
        } else {
            self.raw_t.dimensions.clone()
        };

        ofgpu::io::fields::write_scalar_field(&out_dir.join("k"), &out_k, name)?;
        ofgpu::io::fields::write_scalar_field(&out_dir.join("epsilon"), &out_e, name)?;
        ofgpu::io::fields::write_scalar_field(&out_dir.join("T"), &out_t, name)?;
        ofgpu::io::fields::write_scalar_field(&out_dir.join("nut"), &out_nut, name)?;
        // The flux this run was driven by, so a restart reproduces it exactly
        // instead of re-deriving one - SPEC-LIT §5.1.
        ofgpu::io::fields::write_surface_scalar_field(&out_dir.join("phi"), &out_phi, name)?;

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

// ==========================================================================
//  The transient schedule
// ==========================================================================

/// The resolved transient plan. Every field is validated before it is built,
/// so the loop never has to ask whether a number makes sense.
struct Transient {
    dt: f64,
    n_steps: i64,
    /// Non-positive means "at the end time only".
    write_interval: f64,
    outer_iters: Label,
    do_write: bool,
    graph: bool,
    /// Directory the final write lands in. `-write NAME` wins here, so the
    /// old single-directory habit still works in transient mode; the interval
    /// writes are always named by their own time.
    final_name: String,
}

/// What the summary block needs out of the loop.
struct TransientReport {
    steps: i64,
    writes: usize,
    /// Wall clock of the stepping alone. The stats read-back and the field
    /// writes are timed separately and subtracted, because "how fast does it
    /// step" and "how long did the run take" are different questions and both
    /// were asked.
    loop_wall: f64,
    io_wall: f64,
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

// ==========================================================================
//  The unit of work
// ==========================================================================

/// One outer iteration: turbulence first, then the scalar it diffuses.
///
/// `nut` must be this iteration's, which is why the order is fixed and why
/// this is one function rather than two calls at each site - the capture path
/// and the ordinary path have to record exactly the same sequence.
fn one_iteration(
    gpu: &Gpu,
    model: &mut KEpsilon<'_>,
    heat: &mut ScalarTransport<'_>,
    flow: &FlowState,
) -> Result<Residuals> {
    // SPEC-LIT 17: the temperature reaches the turbulence model, so the
    // buoyancy production is in the k and epsilon equations.
    let (eps, k) = model.correct_buoyant(gpu, flow, Some(heat.field()))?;
    let t = heat.correct(gpu, flow, model.nut())?;

    Ok(Residuals {
        eps_res: eps.initial_residual,
        eps_iters: eps.n_iterations,
        k_res: k.initial_residual,
        k_iters: k.n_iterations,
        t_res: t.initial_residual,
        t_iters: t.n_iterations,
    })
}

/// One time step: `outer` passes of [`one_iteration`].
///
/// The residuals returned are the LAST pass's. That is the pass a reader
/// wants: it is the only one whose systems were assembled from coefficients
/// the step had already settled on.
fn one_step(
    gpu: &Gpu,
    model: &mut KEpsilon<'_>,
    heat: &mut ScalarTransport<'_>,
    flow: &FlowState,
    outer: Label,
) -> Result<Residuals> {
    let mut r = Residuals::default();
    for _ in 0..outer.max(1) {
        r = one_iteration(gpu, model, heat, flow)?;
    }
    Ok(r)
}

/// What the log line for one unit of work prints. Zeroes throughout in
/// `-fixedIters` mode, where nothing is read back.
#[derive(Default, Clone, Copy)]
struct Residuals {
    eps_res: Scalar,
    eps_iters: usize,
    k_res: Scalar,
    k_iters: usize,
    t_res: Scalar,
    t_iters: usize,
}

fn print_residuals(it: Label, r: &Residuals, change: Scalar) {
    println!(
        "{it:>7}  epsilon res {} ({})  k res {} ({})  T res {} ({})  max dk/k {}",
        sci(f64::from(r.eps_res), 3),
        r.eps_iters,
        sci(f64::from(r.k_res), 3),
        r.k_iters,
        sci(f64::from(r.t_res), 3),
        r.t_iters,
        sci(f64::from(change), 3)
    );
}

// ==========================================================================
//  Steady loops
// ==========================================================================

/// The ordinary loop: every kernel submitted by the driver, residuals read
/// back at the check interval. Returns the number of iterations performed.
fn run_per_launch(
    gpu: &Gpu,
    model: &mut KEpsilon<'_>,
    heat: &mut ScalarTransport<'_>,
    flow: &FlowState,
    ctrl: &TurbulenceControls,
    n_iters: Label,
) -> Result<Label> {
    let every = ctrl.convergence_check_every.max(1);
    let mut done: Label = 0;

    for it in 1..=n_iters {
        let r = one_iteration(gpu, model, heat, flow)?;
        done = it;

        if it % every == 0 || it == 1 {
            let change = model.convergence_measure(gpu)?;
            print_residuals(it, &r, change);

            if it > 1 && change < ctrl.convergence_tol {
                println!("converged");
                break;
            }
        }
    }

    Ok(done)
}

/// Capture one outer iteration and replay it.
///
/// The capture executes nothing, so the recording has to be replayed once per
/// iteration exactly as the per-launch loop calls `one_iteration` once per
/// iteration; the two histories are then identical and so are the answers.
/// Only the warm-up iterations before the capture do real work outside the
/// graph, and they are counted.
///
/// The convergence measure still runs between replays - it is a handful of
/// launches and one eight-byte read-back at the check interval, and putting it
/// inside the graph is impossible for exactly that reason.
fn run_graph(
    gpu: &Gpu,
    model: &mut KEpsilon<'_>,
    heat: &mut ScalarTransport<'_>,
    flow: &FlowState,
    ctrl: &TurbulenceControls,
    n_iters: Label,
) -> Result<Label> {
    let every = ctrl.convergence_check_every.max(1);
    let warmup = (GRAPH_WARMUP as Label).min(n_iters);
    let mut done: Label = 0;

    for it in 1..=warmup {
        one_iteration(gpu, model, heat, flow)?;
        done = it;
    }
    gpu.sync()?;

    let t_cap = Instant::now();
    let graph = gpu.capture(|_| {
        one_iteration(gpu, model, heat, flow)?;
        Ok(())
    })?;

    let Some(mut graph) = graph else {
        return Err(Error::Config(
            "capture produced an empty graph - one outer iteration launched no work".to_string(),
        ));
    };
    graph.upload()?;
    gpu.sync()?;

    println!(
        "  captured one outer iteration (k-epsilon + T) in {} s; replaying it. \
         Residuals are not read back in this mode.",
        g(t_cap.elapsed().as_secs_f64())
    );

    for it in (warmup + 1)..=n_iters {
        graph.launch()?;
        done = it;

        if it % every == 0 {
            let change = model.convergence_measure(gpu)?;
            println!("{it:>7}  max dk/k {}", sci(f64::from(change), 3));

            if change < ctrl.convergence_tol {
                println!("converged");
                break;
            }
        }
    }

    Ok(done)
}

// ==========================================================================
//  Transient loop
// ==========================================================================

/// March to the end time, reporting and writing at every write time.
///
/// `-graph` captures ONE WHOLE TIME STEP - all `outer_iters` passes - and
/// replays that per step. Capturing a single pass and replaying it
/// `outer_iters` times would record the same sequence, but it would submit
/// `outer_iters` graph launches per step and so recover proportionally less of
/// the launch overhead the graph exists to remove.
///
/// There is no convergence test: a transient run stops at its end time, not
/// when the fields stop moving, and a `dk/k` read-back per step would put a
/// host transfer back into the loop that `-fixedIters` exists to empty.
fn run_transient(
    gpu: &Gpu,
    model: &mut KEpsilon<'_>,
    heat: &mut ScalarTransport<'_>,
    flow: &FlowState,
    hm: &HostMesh,
    tr: &Transient,
    writer: &FieldWriter<'_>,
) -> Result<TransientReport> {
    // Without -graph nothing is ever captured, so "warm-up" is the whole run.
    let warmup = if tr.graph {
        GRAPH_WARMUP.min(tr.n_steps)
    } else {
        tr.n_steps
    };

    let mut graph: Option<Graph> = None;
    let mut writes = 0usize;
    let mut io_wall = 0.0f64;

    gpu.sync()?;
    let t_loop = Instant::now();

    for step in 1..=tr.n_steps {
        // A capture executes nothing, so the step it happens on is advanced by
        // the replay immediately below - exactly as in the steady path.
        if tr.graph && graph.is_none() && step > warmup {
            gpu.sync()?;
            let t_cap = Instant::now();

            // Reborrowed rather than passed straight in: the closure is
            // `FnOnce` and takes its captures by value, and a `&mut` moved out
            // of a variable inside a loop cannot be used on the next pass.
            let captured = {
                let m = &mut *model;
                let h = &mut *heat;
                gpu.capture(move |_| {
                    one_step(gpu, m, h, flow, tr.outer_iters)?;
                    Ok(())
                })?
            };

            let Some(mut gr) = captured else {
                return Err(Error::Config(
                    "capture produced an empty graph - one time step launched no work".to_string(),
                ));
            };
            gr.upload()?;
            gpu.sync()?;

            println!(
                "  captured one time step ({} x [k-epsilon + T]) in {} s; replaying it. \
                 Residuals are not read back in this mode.",
                tr.outer_iters,
                g(t_cap.elapsed().as_secs_f64())
            );

            graph = Some(gr);
        }

        let r = match &graph {
            Some(gr) => {
                gr.launch()?;
                Residuals::default()
            }
            None => one_step(gpu, model, heat, flow, tr.outer_iters)?,
        };

        let t = step as f64 * tr.dt;
        let last = step == tr.n_steps;

        // The end time always reports, so a run whose end time is not a whole
        // number of write intervals still leaves its final state on disk.
        if !(last || is_write_time(t, tr.write_interval, tr.dt)) {
            continue;
        }

        // Everything from here is host work, so the loop clock stops for it.
        gpu.sync()?;
        let t_io = Instant::now();

        let stats = weighted_stats(&gpu.download(&heat.field().f)?, &hm.v)?;

        println!(
            "t = {:.2} s   step {step}   k {}  eps {}  T {}   T[min,max] {} {}   wall {:.1} s",
            t,
            sci(f64::from(r.k_res), 2),
            sci(f64::from(r.eps_res), 2),
            sci(f64::from(r.t_res), 2),
            g(f64::from(stats.min)),
            g(f64::from(stats.max)),
            t_loop.elapsed().as_secs_f64()
        );

        if tr.do_write {
            let name = if last {
                tr.final_name.clone()
            } else {
                format_time_name(t as Scalar)
            };
            let dir = writer.write(gpu, &name, model, heat, flow.phi, hm)?;
            writes += 1;
            println!("    written to {}", dir.display());
        }

        io_wall += t_io.elapsed().as_secs_f64();
    }

    gpu.sync()?;
    let elapsed = t_loop.elapsed().as_secs_f64();

    Ok(TransientReport {
        steps: tr.n_steps,
        writes,
        loop_wall: (elapsed - io_wall).max(0.0),
        io_wall,
    })
}

/// The numbers the run was asked for, in a block nothing else prints into.
fn print_transient_summary(rep: &TransientReport, tr: &Transient, total_wall: f64) {
    let sim = rep.steps as f64 * tr.dt;
    let steps = rep.steps.max(1) as f64;

    let rule = "=".repeat(74);
    let thin = "-".repeat(74);

    println!("\n{rule}");
    println!("  TRANSIENT TIMING");
    println!("{thin}");
    println!("  time steps                             {}", rep.steps);
    println!("  writes                                 {}", rep.writes);
    println!(
        "  simulated time                         {} s   (dt = {} s, {} outer iters/step)",
        g(sim),
        g(tr.dt),
        tr.outer_iters
    );
    println!("{thin}");
    println!(
        "  wall clock, TIME LOOP ALONE            {:.3} s   (no setup, no IO)",
        rep.loop_wall
    );
    println!(
        "  wall clock, INCLUDING setup + writes   {:.3} s   (of which IO {:.3} s)",
        total_wall, rep.io_wall
    );
    println!(
        "  wall seconds per simulated second      {:.4}     (time loop alone)",
        rep.loop_wall / sim
    );
    println!(
        "  wall seconds per simulated second      {:.4}     (whole run)",
        total_wall / sim
    );
    println!(
        "  ms per time step                       {:.4}",
        rep.loop_wall / steps * 1e3
    );
    println!("{rule}");
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
    println!("{}", device_banner(&gpu, "plume")?);

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

    // The whole of transient mode, and it has to happen HERE: `KEpsilon::new`
    // and `ScalarTransport::new` both take `TurbulenceControls` by value, so a
    // later change to `cc.turb` would be invisible to them.
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
                "[ofgpu-plume] -iters is ignored in transient mode: the step count comes \
                 from -endTime / -deltaT, and -outerIters sets the passes per step"
            );
        }
    }

    let mut t_ctrl = read_t_controls(&o.case_dir, &cc.turb)?;

    if o.fixed_iters > 0 {
        // The genuinely transfer-free mode, so the residual read-back goes
        // too: with both off nothing at all crosses the bus between upload and
        // write, which is also the precondition for -graph.
        for sc in [
            &mut cc.turb.k_solver,
            &mut cc.turb.epsilon_solver,
            &mut t_ctrl.k_solver,
        ] {
            make_fixed(sc, o.fixed_iters);
        }
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

    // ---- fields -----------------------------------------------------------
    let t = find_start_time(&o.case_dir)?;
    let t_dir = o.case_dir.join(&t);

    let t_path = t_dir.join("T");
    if !t_path.exists() {
        return Err(Error::Config(format!(
            "{} has no T field; ofgpu-plume transports one, so the start time \
             must provide it",
            t_dir.display()
        )));
    }

    let raw_u = read_vector_field(&t_dir.join("U"), hm.n_cells)?;
    let raw_k = read_scalar_field(&t_dir.join("k"), hm.n_cells)?;
    let raw_e = read_scalar_field(&t_dir.join("epsilon"), hm.n_cells)?;
    let raw_t = read_scalar_field(&t_path, hm.n_cells)?;

    let fk = FieldKernels::new(&gpu)?;

    let mut u = GpuVectorField::zeros(&gpu, &mesh, "U")?;
    setup_vector_field(&gpu, &mut u, &raw_u, &hm)?;
    correct_boundary_conditions_vector(&gpu, &fk, &mut u, &mesh)?;

    let mut phi = GpuSurfaceScalarField::zeros(&gpu, &mesh, "phi")?;
    let phi_ctrl = read_phi_controls(&o.case_dir)?;
    let source = establish_flux(
        &gpu, &mut phi, &mut u, &hm, &mesh, &t_dir, o, &phi_ctrl,
    )?;
    report_flux(&source);

    println!(
        "max |sum_f phi| per cell = {}   (0 means the flux is discretely conservative)",
        g(f64::from(max_div_phi(&gpu, &phi, &hm)?))
    );

    // SPEC-LIT 15.5: nut's own patch types decide nu_t's wall treatment.
    let mut raw_nut_for_walls = {
        let p = t_dir.join("nut");
        if p.exists() {
            Some(read_scalar_field(&p, hm.n_cells)?)
        } else {
            None
        }
    };
    let mut raw_k = raw_k;
    let mut raw_e = raw_e;
    // SPEC-LIT 29.1: the per-field wall types must form one consistent row.
    ofgpu::field_setup::validate_wall_rows(
        &hm.patches,
        raw_nut_for_walls.as_mut(),
        Some(&mut raw_k),
        Some(&mut raw_e),
        None,
    )?;
    let wf_faces = WallFaces::from_case(&raw_e, raw_nut_for_walls.as_ref(), &hm)?;
    let roughness = NutRoughness::from_case(raw_nut_for_walls.as_ref(), &hm)?;

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
    if cc.buoyancy.is_active() {
        model.set_buoyancy(BuoyancyProduction {
            g: cc.buoyancy.g,
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
    setup_scalar_field(&gpu, heat.field_mut(), &raw_t, &hm)?;

    update_inlet_outlet(&gpu, model.k_mut(), &phi, &hm)?;
    update_inlet_outlet(&gpu, model.epsilon_mut(), &phi, &hm)?;
    update_inlet_outlet(&gpu, heat.field_mut(), &phi, &hm)?;

    model.initialise(&gpu, &flow)?;
    heat.initialise(&gpu)?;

    let writer = FieldWriter {
        case_dir: &o.case_dir,
        raw_k: &raw_k,
        raw_e: &raw_e,
        raw_t: &raw_t,
    };

    // ---- transient: device only between writes ----------------------------
    if transient {
        let dt = f64::from(cc.turb.delta_t);
        let n_steps = ((o.end_time / dt).round() as i64).max(1);
        let sim_end = n_steps as f64 * dt;

        // A write interval shorter than a step fires on every step; say so
        // rather than quietly producing one directory per timestep.
        if o.write_interval > 0.0 && o.write_interval < dt {
            eprintln!(
                "[ofgpu-plume] -writeInterval {} is shorter than -deltaT {}; \
                 every step will be a write time",
                g(o.write_interval),
                g(dt)
            );
        }

        let tr = Transient {
            dt,
            n_steps,
            write_interval: o.write_interval,
            outer_iters: o.outer_iters,
            do_write: o.do_write,
            graph: o.graph,
            final_name: if o.write_time.is_empty() {
                format_time_name(sim_end as Scalar)
            } else {
                o.write_time.clone()
            },
        };

        println!(
            "\ntransient: endTime {} s, deltaT {} s -> {} steps x {} outer iteration(s), \
             writing {}",
            g(sim_end),
            g(dt),
            n_steps,
            tr.outer_iters,
            if tr.write_interval > 0.0 {
                format!("every {} s", g(tr.write_interval))
            } else {
                "at the end time only".to_string()
            }
        );
        println!(
            "  ddt Euler implicit, 1/dt = {} | relax k {} epsilon {} T {}{}",
            g(f64::from(cc.turb.r_delta_t())),
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
        println!();

        let rep = run_transient(&gpu, &mut model, &mut heat, &flow, &hm, &tr, &writer)?;

        // Volume weighting is mass weighting at constant density, so this is
        // the mean a reader can compare against the inlet value: with a
        // conservative flux and no source it must not have drifted.
        let stats = weighted_stats(&gpu.download(&heat.field().f)?, &hm.v)?;
        println!(
            "\nT: min {}  max {}  mass-weighted mean {}",
            g(f64::from(stats.min)),
            g(f64::from(stats.max)),
            g(f64::from(stats.mean))
        );

        print_transient_summary(&rep, &tr, t_total.elapsed().as_secs_f64());

        return Ok(());
    }

    // ---- steady: device only from here to the write -----------------------
    let n_iters = cc.turb.n_outer_iterations;

    println!(
        "\niterating {n_iters} times, relax k {} epsilon {} T {}{}",
        g(f64::from(cc.turb.k_relax)),
        g(f64::from(cc.turb.eps_relax)),
        g(f64::from(t_ctrl.k_relax)),
        if cc.turb.k_solver.fixed_iters {
            " | fixed-iteration solvers: zero host transfers"
        } else {
            ""
        }
    );

    gpu.sync()?;
    let t_loop = Instant::now();

    let done = if o.graph {
        run_graph(&gpu, &mut model, &mut heat, &flow, &cc.turb, n_iters)?
    } else {
        run_per_launch(&gpu, &mut model, &mut heat, &flow, &cc.turb, n_iters)?
    };

    gpu.sync()?;
    let wall = t_loop.elapsed().as_secs_f64();

    // `done` is at least 1 whenever the loop ran; guard anyway, because
    // `-iters 0` is a legal thing for a user to type.
    let n = done.max(1) as f64;

    println!(
        "\n{done} iterations in {} s  ->  {} ms/iteration  ->  {} Mcell-iterations/s",
        sci(wall, 3),
        sci((wall / n) * 1e3, 3),
        sci((hm.n_cells as f64 * n / wall) / 1e6, 3)
    );

    // ---- what the plume did -----------------------------------------------
    let stats = weighted_stats(&gpu.download(&heat.field().f)?, &hm.v)?;
    println!(
        "\nT: min {}  max {}  mass-weighted mean {}",
        g(f64::from(stats.min)),
        g(f64::from(stats.max)),
        g(f64::from(stats.mean))
    );

    // ---- write ------------------------------------------------------------
    if o.do_write {
        let wt = if o.write_time.is_empty() {
            cc.write_time.clone()
        } else {
            o.write_time.clone()
        };
        let out_dir = writer.write(&gpu, &wt, &model, &heat, &phi, &hm)?;
        println!("written to {}", out_dir.display());
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

/// Named `plume_tests` rather than `tests` because `common/mod.rs` is included
/// by `#[path]` and already contributes a `tests` module to this crate.
#[cfg(test)]
mod plume_tests {
    use super::*;

    fn argv(v: &[&str]) -> Vec<String> {
        std::iter::once("ofgpu-plume".to_string())
            .chain(v.iter().map(|s| (*s).to_string()))
            .collect()
    }

    /// The write test is the one piece of transient scheduling that runs on
    /// the host, and getting it wrong either drops every write or writes every
    /// step - neither of which a GPU test would catch any earlier.
    #[test]
    fn write_times_land_on_multiples_of_the_interval() {
        let dt = 0.02;
        let w = 0.5;

        // 25 steps of 0.02 is exactly 0.5 - but only in exact arithmetic.
        let t = 25.0 * dt;
        assert!(is_write_time(t, w, dt), "t = {t} should be a write time");

        assert!(!is_write_time(24.0 * dt, w, dt));
        assert!(!is_write_time(26.0 * dt, w, dt));
        assert!(is_write_time(50.0 * dt, w, dt));

        // t = 0 is the start time, never a write.
        assert!(!is_write_time(0.0, w, dt));

        // No interval means no interval write.
        assert!(!is_write_time(t, 0.0, dt));
        assert!(!is_write_time(t, -1.0, dt));
    }

    #[test]
    fn a_whole_march_writes_once_per_interval() {
        let dt = 0.001;
        let w = 0.25;
        let n_steps = 1000;

        let hits: Vec<i64> = (1..=n_steps)
            .filter(|s| is_write_time(*s as f64 * dt, w, dt))
            .collect();

        // Four writes, no duplicates, no misses. A window wider than half a
        // step would match two adjacent steps and write the same time twice.
        assert_eq!(hits, vec![250, 500, 750, 1000], "write steps were {hits:?}");
    }

    #[test]
    fn a_time_flag_is_a_number_or_an_error() {
        assert!(parse_time("-deltaT", "0.001").is_ok());
        assert!(parse_time("-endTime", " 2 ").is_ok());
        assert!(parse_time("-endTime", "1e-3").is_ok());

        // atoi would read these as 0, and a zero timestep is an infinite
        // 1/dt several hundred launches away from the mistake.
        assert!(parse_time("-deltaT", "").is_err());
        assert!(parse_time("-deltaT", "abc").is_err());
        assert!(parse_time("-deltaT", "nan").is_err());
        assert!(parse_time("-deltaT", "inf").is_err());
    }

    #[test]
    fn transient_flags_are_rejected_before_they_reach_the_gpu() {
        assert!(parse(&argv(&["case", "-endTime", "2", "-deltaT", "0.01"])).is_ok());
        assert!(parse(&argv(&["case", "-endTime", "0"])).is_err());
        assert!(parse(&argv(&["case", "-endTime", "-1"])).is_err());
        assert!(parse(&argv(&["case", "-deltaT", "0"])).is_err());
        assert!(parse(&argv(&["case", "-writeInterval", "0"])).is_err());
        assert!(parse(&argv(&["case", "-outerIters", "0"])).is_err());

        // -graph without -fixedIters was rejected before transient mode
        // existed and has to stay rejected inside it.
        assert!(parse(&argv(&["case", "-endTime", "2", "-graph"])).is_err());
        assert!(parse(&argv(&["case", "-endTime", "2", "-graph", "-fixedIters", "8"])).is_ok());
    }

    #[test]
    fn defaults_leave_the_steady_path_untouched() {
        let o = parse(&argv(&["case"])).expect("a bare case directory is a valid command line");

        assert!(o.end_time <= 0.0, "no -endTime must mean steady");
        assert_eq!(o.outer_iters, DEFAULT_OUTER_ITERS);
        assert_eq!(o.check_every, 25);
        assert_eq!(o.n_iters, -1);
        assert_eq!(o.fixed_iters, -1);
        assert!(o.do_write);
        assert!(!o.graph);
        assert!(o.write_time.is_empty());
    }
}
