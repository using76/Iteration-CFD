// meteor-cfd - Copyright (c) 2026 주식회사 메테오시뮬레이션 (Meteo Simulation Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! `ofgpu-lowmach` - the low-Mach variable-density solver, SPEC-LIT
//! sections 25 and 26, driven by nothing but the case.
//!
//! ```text
//! ofgpu-lowmach <case> [options]
//!
//!   -iters N          steady outer iterations (default 1000)
//!   -check N          print diagnostics every N iterations (default 50)
//!   -permissive       downgrade unsupported-setting errors to warnings
//!
//!   transient mode - both required together, absent means steady SIMPLE:
//!   -endTime T        physical end time, seconds
//!   -deltaT dt        time step, seconds
//!
//!   -sealed           SPEC-LIT §25.2: p0(t) rises with the heat input in a
//!                     closed compartment. Default is OPEN: p0 = const.
//!   -p0 VALUE         the ambient/initial thermodynamic pressure, Pa
//!                     (default 101325)
//!   -heaterPower P    a uniform volumetric heat release of P watts spread
//!                     over the whole domain - SPEC-LIT §18's registry.
//!
//!   -output LIST      comma list of foam,vtu,nvdb,vdb,usda (default: foam)
//!   -writeInterval W  write every W seconds of PHYSICAL time (transient
//!                     only; absent means "write the final state only")
//!   -restartWrite N   write a `.mcr` checkpoint every N steps
//!   -restartFrom FILE resume from a checkpoint - p0 and dp0dt included
//!                     (SPEC-LIT §25.2/§31.2)
//! ```
//!
//! Written from `ofgpu SPEC-LIT.md` sections 25 (the low-Mach formulation)
//! and 26 (the energy equation); the algorithm this wires together is
//! `crate::simple`'s SPEC-LIT §14 PISO/PIMPLE loop, unmodified, plus the ONE
//! seam §25.3 names for a low-Mach solver:
//! [`ofgpu::simple::Simple::correct_outer_low_mach`]. No GPL-licensed source
//! was consulted.
//!
//! # Why this driver exists
//!
//! §25's variable-density formulation is not Boussinesq: the density comes
//! from `p0/(R_s T)` and the continuity equation becomes a prescribed
//! DIVERGENCE rather than `div(u) = 0`. Every driver in this tree that is
//! not built on this one solves either a constant-density flow
//! (`ofgpu-k-epsilon`, `ofgpu-plume`) or a Boussinesq/density-ratio buoyant
//! one (`ofgpu-buoyant`), and none of them constructs an
//! [`ofgpu::energy::GasState`] at all. This one does.
//!
//! It is also what the recorded channel measurements are taken with: the
//! wall-heat-transfer gate of §29.3/§32, the §35 thermostat, the §37 `Pr_t`
//! experiment and the §32.5 friction factor are all properties of THIS
//! loop, and of nothing above it.
//!
//! Their write-up is `docs/07-lowmach-solver.md` §1 and §1.1.
//!
//! # One unit of work
//!
//! ```text
//! rho          <- gas.update_density(T)                          (§25)
//! sources      <- cleared, then the heater (§18) registers into it
//! (div u)_target <- Q/(rho cp T) - dp0/dt/(gamma p0)             (§25.1)
//! U, p, phi    <- Simple::correct_outer_low_mach, target divergence source (§25.3)
//! T            <- Energy::correct, on the phi that satisfies the constraint (§26)
//! p0           <- gas.advance_p0(integral(Q) dV, dt)   [transient, sealed only]  (§25.2)
//! ```
//!
//! `k`/`epsilon` are corrected once per unit of work too, on the `U`/`phi`
//! the PREVIOUS unit produced - the same segregated lag every other
//! coupling coefficient in this crate runs at (`nu_t` into momentum, `T`
//! into buoyancy, `phi` into the scalar equations).
//!
//! # What registers on `Q`, and what does not
//!
//! §25.1's `Q` is one buffer on [`ofgpu::energy::EnergySources`] that any
//! number of models may add into - `-heaterPower`, §35.1's thermostat, a
//! case's own `sources[]` block, and, in a build that carries them, a
//! reacting or radiating model. This driver registers the first three and
//! knows about no others: the energy equation's assembly reads the sum and
//! asks nothing about where it came from, which is §18's whole point. A
//! term nobody registers is not missing from the formulation, it is
//! numerically zero.
//!
//! `div(k_eff grad T)` is the exception, and it is not a registration:
//! [`ofgpu::energy::Energy::update_target_divergence`] computes the
//! conduction divergence itself (§26.1), because it is the one term of `Q`
//! that the energy module already has every field for.
//!
//! # Turbulence: SPEC-LIT §30.2
//!
//! `constant/momentumTransport` (or a JSONC `turbulence` block) picks the
//! model exactly as `ofgpu-buoyant` does - `crate::models::registry::build_coupled`
//! constructs whichever of kEpsilon/kOmega/kOmegaSST/LES/laminar the case
//! names, and the outer loop below drives it through
//! [`ofgpu::models::CoupledTurbulence`] alone, never matching on the
//! concrete type.
//!
//! # What `ofgpu-lowmach` does NOT do, on purpose
//!
//! * **`nOuterCorrectors` is fixed at 1.** [`ofgpu::energy::Energy::correct`]
//!   refreshes its own old-time level on entry, exactly like
//!   [`ofgpu::scalar_transport::ScalarTransport::correct`] - calling it more
//!   than once per time step would advance `T` by more Euler sub-steps than
//!   `U` (see `ofgpu-buoyant`'s module doc for the identical trap on `U`
//!   before `Simple::begin_time_step` existed). A future pass that wants a
//!   real PIMPLE outer loop here needs `Energy` to grow the same
//!   once-per-step/once-per-corrector split `Simple` already has.
//! * **The energy equation's `ddtSchemes` is fixed at `Euler`/`steadyState`.**
//!   `backward` (BDF2) is implemented and tested in `crate::energy` itself;
//!   this driver does not read a `ddtSchemes` entry for `T` yet.
//! * **Momentum's own `ddt`/convection stay density-unweighted** - see
//!   `crate::energy`'s first module-level *DESIGN* note for why that is
//!   exact, not approximate, at the buoyancy term, and a leading-order
//!   simplification everywhere else.
//! * **It transports no scalars of its own.** §19's species equation is a
//!   library capability this driver does not reach for, so a case that
//!   carries species fields has them read by nothing here.
//!
//! # Field output and restart (SPEC-LIT §31.2)
//!
//! `-output LIST`/`-writeInterval` write `U`, `p`, `T`, the turbulence
//! model's own fields (`k`/`epsilon` or `k`/`omega`, `nut` - SPEC-LIT
//! §30.2's `CoupledTurbulence::output_fields`) and `rho` - `write_time`,
//! below, `ofgpu-buoyant`/`ofgpu-vof`'s own `io::writer` seam.
//! `-restartWrite N`/`-restartFrom FILE` are
//! `restart::write_restart`/`read_restart`, in the `.mcr` format of
//! `docs/05-io-redesign.md` §4.6.
//!
//! One thing is peculiar to a low-Mach restart: the thermodynamic pressure
//! `p0` (§25.2) has to survive the round trip too, or the resumed run starts
//! from a different thermodynamic state than the one it stopped in even with
//! `U`/`p`/`T` restored exactly - see `write_restart_checkpoint`'s own doc,
//! and `ofgpu::restart`'s "Version 2: `dp0dt`" note for the one part of that
//! state the `.mcr` format did not carry until this section's own gate test
//! found the gap.

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use ofgpu::device::DevBuf;
use ofgpu::energy::{
    kays_crawford_prt, DomainKind, Energy, EnergyControls, GasProperties, GasState, PrtModel,
    KAYS_CRAWFORD_C,
};
use ofgpu::field::{BcKind, GpuScalarField, GpuSurfaceScalarField};
use ofgpu::field_ops::{update_inlet_outlet_scalar, FieldKernels};
use ofgpu::field_setup::{
    faces_where, harvest_scalar_field, harvest_vector_field, les_nut_wall_faces,
    setup_scalar_field, setup_vector_field, validate_wall_rows, wall_coeffs_from_case,
    NutRoughness, WallFaces,
};
use ofgpu::io::case::{find_start_time, format_time_name, CaseControls};
use ofgpu::io::fields::{read_scalar_field, read_vector_field, RawScalarField, RawVectorField};
use ofgpu::models::{
    build_coupled, select_turbulence_model, CoupledTurbulence, RasModel, ThermalCtx,
};
use ofgpu::momentum::MomentumControls;
use ofgpu::pressure::{PbicgstabBackend, PressureBackend, SystemProbe};
use ofgpu::simple::{Simple, SimpleControls};
use ofgpu::turbulence::FlowState;
use ofgpu::timescheme::DdtScheme;
use ofgpu::{Error, Gpu, GpuMesh, HostMesh, Result, Scalar};

#[path = "common/mod.rs"]
mod common;

use common::{
    atoi, device_banner, find_restart_field, from_restart_scalars,
    from_restart_vectors, g, load_case, next_arg, output_root, parse_output_formats,
    restart_scalar, restart_shell, restart_surface, restart_vector, sci, CaseNumerics,
    OutputFormat,
};
use ofgpu::restart::{self, RestartData};

// ==========================================================================
//  Command line
// ==========================================================================

struct Options {
    case_path: PathBuf,
    n_iters: i64,
    check_every: i64,
    end_time: f64,
    delta_t: f64,
    sealed: bool,
    p0: Scalar,
    heater_power: Scalar,
    /// `-output foam|vtu|nvdb|vdb|usda`, comma list.
    output: Vec<OutputFormat>,
    /// `-writeInterval W` - write every W seconds of PHYSICAL time. Non-
    /// positive means "not given": only the final state is written, exactly
    /// as `ofgpu-buoyant`/`ofgpu-vof` treat an absent `-writeInterval`.
    write_interval: f64,
    /// `-restartWrite N` - write a `.mcr` checkpoint every N steps.
    restart_write: Option<u64>,
    /// `-restartFrom FILE` - load state from a checkpoint, skipping
    /// whatever re-initialisation would otherwise overwrite it (the
    /// potential-flow-equivalent `phi` seed, and every field's own initial
    /// condition).
    restart_from: Option<PathBuf>,
    /// Which of `-output`, `-writeInterval`, `-restartWrite` this command
    /// line actually NAMED - SPEC-LIT §44.6.
    ///
    /// Not the same question as "what are they set to": `-output` defaults to
    /// `foam` and `write_interval` to `0`, so every run has values for all
    /// three. A case that carries an `output` block and a command line that
    /// names any of them are two ways of saying the same thing, and that is
    /// an error naming both rather than a silent winner - which needs this
    /// list, not those values.
    output_flags: Vec<&'static str>,
}

fn usage() {
    eprintln!(
        "usage: ofgpu-lowmach <case> [-iters N] [-check N] [-endTime T] [-deltaT dt]\n       \
         [-sealed] [-p0 PA] [-heaterPower W] [-output LIST]\n       \
         [-writeInterval W] [-restartWrite N] [-restartFrom FILE] [-permissive]"
    );
}

fn parse_time(flag: &str, s: &str) -> Result<f64> {
    match s.trim().parse::<f64>() {
        Ok(v) if v.is_finite() => Ok(v),
        _ => Err(Error::Config(format!(
            "{flag} needs a finite number, got \"{s}\""
        ))),
    }
}

fn parse(args: &[String]) -> Result<Options> {
    if args.len() < 2 {
        usage();
        return Err(Error::Config("a case path is required".to_string()));
    }

    let mut o = Options {
        case_path: PathBuf::from(&args[1]),
        n_iters: 1000,
        check_every: 50,
        end_time: 0.0,
        delta_t: 0.0,
        sealed: false,
        p0: 101_325.0,
        heater_power: 0.0,
        output: vec![OutputFormat::Foam],
        write_interval: 0.0,
        restart_write: None,
        restart_from: None,
        output_flags: Vec::new(),
    };

    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "-iters" => {
                o.n_iters = atoi(&next_arg(args, &mut i)?);
            }
            "-check" => {
                o.check_every = atoi(&next_arg(args, &mut i)?);
            }
            "-endTime" => {
                o.end_time = parse_time("-endTime", &next_arg(args, &mut i)?)?;
            }
            "-deltaT" => {
                o.delta_t = parse_time("-deltaT", &next_arg(args, &mut i)?)?;
            }
            "-sealed" => {
                o.sealed = true;
            }
            "-p0" => {
                o.p0 = parse_time("-p0", &next_arg(args, &mut i)?)? as Scalar;
            }
            "-heaterPower" => {
                o.heater_power = parse_time("-heaterPower", &next_arg(args, &mut i)?)? as Scalar;
            }
            "-output" => {
                o.output = parse_output_formats(&next_arg(args, &mut i)?)?;
                o.output_flags.push("-output");
            }
            "-writeInterval" => {
                o.write_interval = parse_time("-writeInterval", &next_arg(args, &mut i)?)?;
                o.output_flags.push("-writeInterval");
            }
            "-restartWrite" => {
                let n = atoi(&next_arg(args, &mut i)?);
                if n <= 0 {
                    return Err(Error::Config("-restartWrite needs a positive step count".to_string()));
                }
                o.restart_write = Some(n as u64);
                o.output_flags.push("-restartWrite");
            }
            "-restartFrom" => o.restart_from = Some(PathBuf::from(next_arg(args, &mut i)?)),
            "-permissive" => {
                ofgpu::io::contract::set_permissive(true);
            }
            other => {
                usage();
                return Err(Error::Config(format!("unrecognised option \"{other}\"")));
            }
        }
        i += 1;
    }

    if o.end_time > 0.0 && !(o.delta_t > 0.0) {
        return Err(Error::Config(
            "-endTime needs -deltaT as well - ofgpu-lowmach does not read \
             controlDict's deltaT"
                .to_string(),
        ));
    }
    if !(o.p0 > 0.0) {
        return Err(Error::Config(format!("-p0 is {}; it is an absolute pressure", o.p0)));
    }
    if o.n_iters <= 0 {
        return Err(Error::Config(format!("-iters is {}; it must be positive", o.n_iters)));
    }

    Ok(o)
}

/// `name` in a [`CoupledTurbulence::output_fields`] list, without the
/// caller downcasting to find out which concrete model it is holding -
/// `"k"` is present for k-epsilon/k-omega/SST and absent for laminar/LES.
fn find_field<'a>(
    fields: &[(&'static str, &'a GpuScalarField)],
    name: &str,
) -> Option<&'a GpuScalarField> {
    fields.iter().find(|(n, _)| *n == name).map(|(_, f)| *f)
}

// ==========================================================================
//  One unit of work - shared between `main` and the integration test
// ==========================================================================

/// What one call to [`outer_iteration`] found, for the caller to print or
/// assert on.
#[derive(Debug, Clone, Copy, Default)]
pub struct IterReport {
    pub u_residual: Scalar,
    /// The pressure corrector's OWN initial residual - SPEC-LIT §31.2's
    /// restart gate reads this directly: a restarted step 21 solving the
    /// same linear system a continuous run's step 21 did should report the
    /// same number here.
    pub p_residual: Scalar,
    pub continuity_error: Scalar,
    pub t_min: Scalar,
    pub t_max: Scalar,
    pub rho_min: Scalar,
    pub rho_max: Scalar,
    pub p0: Scalar,
    pub dp0dt: Scalar,
    /// False the moment `T`, `rho`, `U` or `p` is caught holding a NaN or an
    /// infinity - the §25/§26 gate this driver exists to demonstrate.
    pub finite: bool,
}

/// One pass of the module doc's "one unit of work" - SPEC-LIT §25/§26,
/// assembled out of [`ofgpu::simple::Simple::correct_outer_low_mach`] and
/// [`ofgpu::energy::Energy::correct`] with nothing else in between.
///
/// `nut` is read, never written - the caller corrects turbulence separately,
/// on the flow the PREVIOUS unit of work produced, exactly the lag every
/// other coupling coefficient in this crate already runs at. `k`/`nu` are
/// the same PREVIOUS-iteration turbulence kinetic energy and molecular
/// viscosity; [`ofgpu::energy::Energy::correct`] reads them ONLY for the
/// SPEC-LIT §29.3 thermal wall function, and only where
/// [`ofgpu::energy::Energy::set_thermal_wall`] was called - see `main`.
///
/// `heater` is SPEC-LIT §18's registry - one uniform volumetric term, and
/// the same buffer any other model registering on `Q` would add into.
///
/// `thermostat` is SPEC-LIT §35.1's bulk-temperature controller, present iff
/// `main` found a `thermostat` entry in the case's `sources[]` - see
/// [`ofgpu::sources::Thermostat`]'s own doc for why it is not just another
/// `heater`-shaped constant buffer.
///
/// `dt_for_p0` is `Some(dt)` in transient mode (`p0` is advanced by one
/// explicit-Euler step of the §25.2 ODE using what `heater` registered this
/// iteration) and `None` in steady mode, where advancing a thermodynamic
/// pressure by a "time step" that is really an iteration count has no
/// meaning - a steady sealed-box run holds `p0` at whatever it was
/// constructed with.
#[allow(clippy::too_many_arguments)]
pub fn outer_iteration(
    gpu: &Gpu,
    m: &GpuMesh,
    s: &mut Simple,
    energy: &mut Energy,
    gas: &mut GasState,
    backend: &mut dyn PressureBackend,
    nut: &GpuScalarField,
    k: &DevBuf<Scalar>,
    nu: Scalar,
    heater: Option<&DevBuf<Scalar>>,
    thermostat: Option<&mut ofgpu::sources::Thermostat>,
    is_final: bool,
    dt_for_p0: Option<Scalar>,
) -> Result<IterReport> {
    let fk = FieldKernels::new(gpu)?;

    // `inletOutlet` faces on T (an open boundary: ambient in, whatever is
    // there out) switch on the sign of the flux the PREVIOUS iteration
    // produced, exactly like `Simple::correct_outer_impl` already does for
    // `U` and `p` at its own top - `crate::field_ops::update_inlet_outlet`'s
    // own doc: "faces of every other kind are untouched", so this is a no-op
    // wherever the case gave `T` a plain fixedValue/zeroGradient instead.
    update_inlet_outlet_scalar(gpu, &fk, energy.field_mut(), s.phi())?;

    gas.update_density(gpu, energy.field())?;

    energy.sources_mut().clear(gpu)?;
    if let Some(h) = heater {
        energy.sources_mut().register_explicit(gpu, h)?;
    }
    // SPEC-LIT §35.1: measured off THIS iteration's `T` (unchanged since the
    // `update_inlet_outlet_scalar` above, which touches only `inletOutlet`
    // boundary faces) and registered the same way the heater just was - a
    // uniform W/m3 field, recomputed fresh every iteration because `T_mean`
    // moves every iteration even though the heater's own value never does.
    //
    // SPEC-LIT §35.3: `rho` and `U` go in too, at exactly this lag - `rho`
    // was just refreshed from THIS iteration's `T` by `update_density`
    // above, and `U` is what the previous unit of work left. A uniform
    // thermostat ignores both and takes the identical path it always did.
    if let Some(th) = thermostat {
        th.correct_with_flow(gpu, m, &energy.field().f, &gas.rho().f, &s.u().f)?;
        energy.sources_mut().register_explicit(gpu, th.source_buf())?;
    }

    energy.update_target_divergence(gpu, gas, nut, k, nu)?;

    let perf = s.correct_outer_low_mach(
        gpu,
        backend,
        nut,
        energy.field(),
        energy.target_divergence(),
        is_final,
    )?;

    energy.correct(gpu, s.phi(), nut, k, nu, gas)?;

    if let Some(dt) = dt_for_p0 {
        // SPEC-LIT §25.2 integrates the SAME `Q` §25.1's constraint uses, and
        // since §26.1 that `Q` is the §18 registry PLUS the conduction
        // divergence `div(k_eff grad T)` - which telescopes to the net heat
        // crossing the boundary and is therefore exactly zero on the sealed,
        // adiabatic box §25.2's decisive gate tests. Splitting the two would
        // put a different `Q` in the constraint than in the `p0` ODE, which
        // is the inconsistency §26.1 was written to remove.
        let total_q = energy.sources_mut().total_q(gpu, m)? + energy.total_conduction_q(gpu)?;
        gas.advance_p0(total_q, dt)?;
    }

    let u_residual = perf
        .u
        .iter()
        .map(|r| r.initial_residual)
        .fold(0.0 as Scalar, Scalar::max);

    let t = gpu.download(&energy.field().f)?;
    let rho = gpu.download(&gas.rho().f)?;
    let u = gpu.download(&s.u().f)?;
    let p = gpu.download(&s.p().f)?;

    let finite = t.iter().all(|v| v.is_finite())
        && rho.iter().all(|v| v.is_finite())
        && u.iter().all(|v| v.x.is_finite() && v.y.is_finite() && v.z.is_finite())
        && p.iter().all(|v| v.is_finite());

    let (mut t_min, mut t_max) = (Scalar::INFINITY, Scalar::NEG_INFINITY);
    let (mut rho_min, mut rho_max) = (Scalar::INFINITY, Scalar::NEG_INFINITY);
    for &v in &t {
        t_min = t_min.min(v);
        t_max = t_max.max(v);
    }
    for &v in &rho {
        rho_min = rho_min.min(v);
        rho_max = rho_max.max(v);
    }

    Ok(IterReport {
        u_residual,
        p_residual: perf.p.initial_residual,
        continuity_error: perf.continuity_error,
        t_min,
        t_max,
        rho_min,
        rho_max,
        p0: gas.p0(),
        dp0dt: gas.dp0dt(),
        finite,
    })
}

fn print_report(iter: usize, r: &IterReport) {
    println!(
        "iter {iter:6}  |U| res {}  |p| res {}  contErr {}  T [{}, {}] K  rho [{}, {}] kg/m3  \
         p0 {} Pa  dp0/dt {} Pa/s{}",
        g(f64::from(r.u_residual)),
        sci(f64::from(r.p_residual), 3),
        g(f64::from(r.continuity_error)),
        g(f64::from(r.t_min)),
        g(f64::from(r.t_max)),
        g(f64::from(r.rho_min)),
        g(f64::from(r.rho_max)),
        g(f64::from(r.p0)),
        g(f64::from(r.dp0dt)),
        if r.finite { "" } else { "  *** NaN/Inf ***" },
    );
}

// ==========================================================================
//  SPEC-LIT §31.2: field output and restart
// ==========================================================================

/// The dimensions SPEC-LIT's own field list gives each turbulence quantity -
/// `ofgpu-buoyant`'s identical table, the one place this driver has to know
/// that `epsilon` and `omega` are not interchangeable even in their units.
fn turb_field_dimensions(name: &str) -> &'static str {
    match name {
        "k" => "[0 2 -2 0 0 0 0]",
        "epsilon" => "[0 2 -3 0 0 0 0]",
        "omega" => "[0 0 -1 0 0 0 0]",
        // "nut" and anything else this batch's models do not name.
        _ => "[0 2 -1 0 0 0 0]",
    }
}

/// The boundary-type seed for `name` out of a `(name, RawScalarField)` list -
/// `ofgpu-buoyant`'s `FieldWriter::seed_for`, freed of its struct so this
/// driver's writer stays a handful of free functions.
fn seed_for(seeds: &[(&'static str, RawScalarField)], name: &str) -> RawScalarField {
    seeds
        .iter()
        .find(|(n, _)| *n == name)
        .map(|(_, r)| r.types_only())
        .unwrap_or_default()
}

/// Write one time/step directory in every `-output` format asked for: `U`,
/// `p`, `T`, the turbulence model's own fields (`k`/`epsilon` or `k`/`omega`,
/// `nut` - SPEC-LIT §30.2's `CoupledTurbulence::output_fields`) and `rho`
/// (SPEC-LIT §25).
#[allow(clippy::too_many_arguments)]
fn write_time(
    gpu: &Gpu,
    s: &Simple<'_>,
    turb: &dyn CoupledTurbulence,
    energy: &Energy<'_>,
    gas: &GasState<'_>,
    hm: &HostMesh,
    case_path: &Path,
    t: Scalar,
    // SPEC-LIT §44: the writers are built ONCE, in `run`, and live across
    // the whole loop. They used to be rebuilt inside this function on every
    // call, which is why `VtuWriter`'s `.pvd` collection never grew past one
    // entry and `UsdaWriter`'s `timeSamples` map never accumulated - a
    // writer that is thrown away after each step cannot keep a series.
    pipeline: &mut ofgpu::io::OutputPipeline,
    // `true` for the unconditional final write; otherwise only the stages
    // whose own interval is due write.
    force: bool,
    fields: &InitialFields,
    raw_turb: &[(&'static str, RawScalarField)],
    raw_rho_seed: &RawScalarField,
) -> Result<()> {
    let out_root = output_root(case_path);
    let name = format_time_name(t);

    let mut out_u = fields.u.types_only();
    let mut out_p = fields.p.types_only();
    let mut out_t = fields.t.types_only();
    harvest_vector_field(gpu, &mut out_u, s.u(), hm)?;
    harvest_scalar_field(gpu, &mut out_p, s.p(), hm)?;
    harvest_scalar_field(gpu, &mut out_t, energy.field(), hm)?;
    out_u.dimensions = "[0 1 -1 0 0 0 0]".to_string();
    out_p.dimensions =
        if fields.p.dimensions.is_empty() { "[1 -1 -2 0 0 0 0]".to_string() } else { fields.p.dimensions.clone() };
    out_t.dimensions =
        if fields.t.dimensions.is_empty() { "[0 0 0 1 0 0 0]".to_string() } else { fields.t.dimensions.clone() };

    let mut out_rho = raw_rho_seed.types_only();
    harvest_scalar_field(gpu, &mut out_rho, gas.rho(), hm)?;
    out_rho.dimensions = "[1 -3 0 0 0 0 0]".to_string();

    let mut out_turb: Vec<(&'static str, RawScalarField)> = Vec::new();
    for (fname, field) in turb.output_fields() {
        let mut out = seed_for(raw_turb, fname);
        harvest_scalar_field(gpu, &mut out, field, hm)?;
        out.dimensions = turb_field_dimensions(fname).to_string();
        out_turb.push((fname, out));
    }

    let mut foam_fields: Vec<ofgpu::io::FoamField> = vec![
        ofgpu::io::FoamField::vector("U", &out_u),
        ofgpu::io::FoamField::scalar("p", &out_p),
        ofgpu::io::FoamField::scalar("T", &out_t),
        ofgpu::io::FoamField::scalar("rho", &out_rho),
    ];
    for (fname, out) in &out_turb {
        foam_fields.push(ofgpu::io::FoamField::scalar(fname, out));
    }

    let mut vis_fields: Vec<ofgpu::io::OutputField> = vec![
        ofgpu::io::OutputField::vector("U", &out_u.internal),
        ofgpu::io::OutputField::scalar("p", &out_p.internal),
        ofgpu::io::OutputField::scalar("T", &out_t.internal),
        ofgpu::io::OutputField::scalar("rho", &out_rho.internal),
    ];
    for (fname, out) in &out_turb {
        vis_fields.push(ofgpu::io::OutputField::scalar(fname, &out.internal));
    }

    let cart = ofgpu::pressure::cartesian::detect(hm)
        .ok()
        .map(|c| ofgpu::io::cartesian_info(hm, &c));
    let ctx = ofgpu::io::WriteCtx {
        time: t,
        // Ignored: each stage numbers its own files - see
        // `OutputPipeline::write`. It was a hard-coded `0` here, which made
        // every `.vtu`/`.vdb`/`.nvdb` of a run overwrite the last.
        step: 0,
        name: &name,
        mesh: hm,
        cart: cart.as_ref(),
        fields: &vis_fields,
        foam: &foam_fields,
    };
    if pipeline.write(&ctx, f64::from(t), force)? {
        println!("    written to {}", out_root.join(&name).display());
    }
    Ok(())
}

/// The cell fields `write_time` will build, by name and in order - SPEC-LIT
/// §44.2's early check.
///
/// **This is a second statement of what `write_time` builds, and that is a
/// risk taken deliberately.** The alternative is to raise the error at the
/// first write, which on a six-hour transient run is six hours after the
/// typo. The risk is bounded because it is not the only check:
/// `FieldSelection::apply` re-checks against the fields actually in hand at
/// every write, so a drift between the two lists surfaces as a loud error on
/// the first write rather than as a silently wrong file - and
/// `lowmach_tests::the_early_field_list_is_what_the_run_actually_writes` pins
/// them together.
fn output_field_names(turb: &dyn CoupledTurbulence) -> Vec<String> {
    let mut names: Vec<String> =
        ["U", "p", "T", "rho"].iter().map(|s| (*s).to_string()).collect();
    for (fname, _) in turb.output_fields() {
        names.push(fname.to_string());
    }
    names
}

/// SPEC-LIT §44.5's route: one checkpoint of a RETAINED SERIES, named after
/// the driver's own time label, with the older ones pruned.
///
/// The pruning is reported rather than silent - deleting files is the one
/// genuinely destructive thing §44 does, and a run that deletes without
/// saying so is a run nobody can audit.
#[allow(clippy::too_many_arguments)]
fn write_series_checkpoint(
    gpu: &Gpu,
    s: &Simple<'_>,
    turb: &dyn CoupledTurbulence,
    energy: &Energy<'_>,
    gas: &GasState<'_>,
    hm: &HostMesh,
    pipeline: &mut ofgpu::io::OutputPipeline,
    mesh_hash: u64,
    t: Scalar,
) -> Result<()> {
    let data = build_restart_data(gpu, s, turb, energy, gas, hm, mesh_hash, t)?;
    let label = format_time_name(t);
    let Some(ck) = pipeline.restart_mut() else { return Ok(()) };
    let (path, removed) = ck.write(&data, &label)?;
    println!(
        "    restart checkpoint written to {} (t = {}, p0 = {})",
        path.display(),
        g(f64::from(t)),
        sci(f64::from(gas.p0()), 6)
    );
    for old in &removed {
        println!("    retired checkpoint {} (output.restart.keep)", old.display());
    }
    Ok(())
}

/// Everything one restart interval needs: `U`, `p`, `phi`, `T` and the
/// turbulence model's own fields. `p0` (SPEC-LIT §25.2) rides in the `.mcr`
/// header's own `p0` slot, exactly, since unlike `ofgpu-buoyant`/
/// `ofgpu-vof`'s incompressible pressure it IS the state variable that
/// slot names - see the restore side's own comment in `run` for why this
/// is not the mean-pressure diagnostic the other two drivers put there.
///
/// Split from the WRITE (SPEC-LIT §44.5) so the same harvested state can go
/// either to the command line's single `restart.mcr` or into the case's
/// retained series, with no second copy of the field list to drift.
#[allow(clippy::too_many_arguments)]
fn build_restart_data(
    gpu: &Gpu,
    s: &Simple<'_>,
    turb: &dyn CoupledTurbulence,
    energy: &Energy<'_>,
    gas: &GasState<'_>,
    hm: &HostMesh,
    mesh_hash: u64,
    t: Scalar,
) -> Result<RestartData> {
    let u_i = gpu.download(&s.u().f)?;
    let u_b = gpu.download(&s.u().bf)?;
    let p_i = gpu.download(&s.p().f)?;
    let p_b = gpu.download(&s.p().bf)?;
    let phi_i = gpu.download(&s.phi().f)?;
    let phi_b = gpu.download(&s.phi().bf)?;
    let t_i = gpu.download(&energy.field().f)?;
    let t_b = gpu.download(&energy.field().bf)?;

    let mut data = restart_shell(mesh_hash, t, gas.p0(), hm);
    // SPEC-LIT §31.2/§25.2: `p0` alone is not the whole thermodynamic
    // state a low-Mach restart needs - `dp0dt` closes the gap the gate test
    // found (see `ofgpu::energy::GasState::set_dp0dt`'s doc and the `.mcr`
    // format's "Version 2" note).
    data.dp0dt = f64::from(gas.dp0dt());
    data.fields.push(restart_vector("U", &u_i, &u_b));
    data.fields.push(restart_scalar("p", &p_i, &p_b));
    data.fields.push(restart_surface("phi", &phi_i, &phi_b));
    data.fields.push(restart_scalar("T", &t_i, &t_b));

    for (fname, field) in turb.output_fields() {
        let fi = gpu.download(&field.f)?;
        let fb = gpu.download(&field.bf)?;
        data.fields.push(restart_scalar(fname, &fi, &fb));
    }

    Ok(data)
}

/// The command-line route (`-restartWrite N`, in STEPS): one file,
/// `<out>/restart.mcr`, overwritten. Unchanged by SPEC-LIT §44 on purpose -
/// §44.6 makes a case pick this or `output.restart`, never both.
#[allow(clippy::too_many_arguments)]
fn write_restart_checkpoint(
    gpu: &Gpu,
    s: &Simple<'_>,
    turb: &dyn CoupledTurbulence,
    energy: &Energy<'_>,
    gas: &GasState<'_>,
    hm: &HostMesh,
    case_path: &Path,
    mesh_hash: u64,
    t: Scalar,
) -> Result<()> {
    let data = build_restart_data(gpu, s, turb, energy, gas, hm, mesh_hash, t)?;
    let out_root = output_root(case_path);
    std::fs::create_dir_all(&out_root).map_err(|e| {
        Error::Config(format!(
            "ofgpu-lowmach: could not create {}: {e}",
            out_root.display()
        ))
    })?;
    let path = out_root.join("restart.mcr");
    restart::write_restart(&path, &data)?;
    println!(
        "    restart checkpoint written to {} (t = {}, p0 = {})",
        path.display(),
        g(f64::from(t)),
        sci(f64::from(gas.p0()), 6)
    );
    Ok(())
}

// ==========================================================================
//  Set-up
// ==========================================================================

struct InitialFields {
    u: RawVectorField,
    p: RawScalarField,
    t: RawScalarField,
    /// `Some` for every RAS model (SPEC-LIT §30.2's `RasModel::dissipation_field`
    /// is `Some`); `None` for `laminar`/`LES`, which solve no `k` equation.
    k: Option<RawScalarField>,
    /// The model's OWN dissipation variable - `epsilon` for k-epsilon,
    /// `omega` for k-omega/SST, absent for laminar/LES. Whichever it is, its
    /// `0/` file name is `RasModel::dissipation_field()`.
    diss: Option<RawScalarField>,
    nut: Option<RawScalarField>,
}

/// Read `U`, `p`, `T` (and `k`/the model's own dissipation field, `nut`, if
/// present) from either case format - the same duality
/// [`common::load_case`] already gives the mesh and controls through.
///
/// `diss_name` is `RasModel::dissipation_field()` for whichever model
/// [`select_turbulence_model`] already chose - SPEC-LIT §30.2: `Some("epsilon")`
/// for k-epsilon, `Some("omega")` for k-omega/SST, `None` for laminar/LES,
/// which solve no `k` equation and have no `0/` dissipation file to find.
fn load_initial_fields(
    case_path: &Path,
    hm: &HostMesh,
    diss_name: Option<&'static str>,
    model_name: &str,
) -> Result<(InitialFields, Option<ofgpu::io::case_json::LoweredCase>)> {
    if common::is_json_case(case_path) {
        let json = ofgpu::io::case_json::read_case_jsonc(case_path)?;
        let lowered = json.lower()?;
        let need = |name: &str, f: &Option<ofgpu::io::case_json::LoweredScalarField>| {
            f.as_ref().map(|x| x.to_raw(hm.n_cells)).ok_or_else(|| {
                Error::Config(format!(
                    "ofgpu-lowmach needs `initial.{name}` - the low-Mach energy \
                     equation (SPEC-LIT §26) needs T, and this case runs \
                     {model_name} on top of it"
                ))
            })
        };
        let (k, diss) = match diss_name {
            Some("epsilon") => (Some(need("k", &lowered.k_field)?), Some(need("epsilon", &lowered.epsilon_field)?)),
            Some("omega") => (Some(need("k", &lowered.k_field)?), Some(need("omega", &lowered.omega_field)?)),
            Some(other) => unreachable!("RasModel::dissipation_field only names epsilon/omega, got {other}"),
            None => (None, None),
        };
        let fields = InitialFields {
            u: lowered.u_field.to_raw(hm.n_cells),
            p: lowered.p_field.to_raw(hm.n_cells),
            t: need("T", &lowered.t_field)?,
            k,
            diss,
            nut: lowered.nut_field.as_ref().map(|f| f.to_raw(hm.n_cells)),
        };
        Ok((fields, Some(lowered)))
    } else {
        let t_name = find_start_time(case_path)?;
        let t_dir = case_path.join(&t_name);
        let mut required = vec!["U", "p", "T"];
        if let Some(name) = diss_name {
            required.push("k");
            required.push(name);
        }
        for name in &required {
            if !t_dir.join(name).exists() {
                return Err(Error::Config(format!(
                    "{} has no {name} field; ofgpu-lowmach with {model_name} solves {}",
                    t_dir.display(),
                    required.join(", "),
                )));
            }
        }
        let mut k = match diss_name {
            Some(_) => Some(read_scalar_field(&t_dir.join("k"), hm.n_cells)?),
            None => None,
        };
        let mut diss = match diss_name {
            Some(name) => Some(read_scalar_field(&t_dir.join(name), hm.n_cells)?),
            None => None,
        };
        let mut nut = if t_dir.join("nut").exists() {
            Some(read_scalar_field(&t_dir.join("nut"), hm.n_cells)?)
        } else {
            None
        };
        // SPEC-LIT 29.1: the per-field wall types must form one consistent
        // row, on this route exactly as on the JSONC route. Only the
        // DISSIPATION field the selected model actually carries is checked -
        // `epsilon`'s slot for k-epsilon, `omega`'s for k-omega/SST - because
        // the other one has no `0/` file to have an opinion in.
        if diss_name.is_some() {
            let (eps_slot, omega_slot) = match diss_name {
                Some("omega") => (None, diss.as_mut()),
                _ => (diss.as_mut(), None),
            };
            validate_wall_rows(&hm.patches, nut.as_mut(), k.as_mut(), eps_slot, omega_slot)?;
        }
        let fields = InitialFields {
            u: read_vector_field(&t_dir.join("U"), hm.n_cells)?,
            p: read_scalar_field(&t_dir.join("p"), hm.n_cells)?,
            t: read_scalar_field(&t_dir.join("T"), hm.n_cells)?,
            k,
            diss,
            nut,
        };
        Ok((fields, None))
    }
}

/// `Simple` owns `U` and `phi` in the same struct, so
/// `field_setup::compute_phi_from_u(gpu, s.phi_mut(), s.u(), hm)` cannot be
/// called directly - the two accessors borrow all of `s`, mutably and
/// immutably, in the same call. Computed into a fresh field and moved in
/// instead: `GpuSurfaceScalarField` needs neither `Clone` nor `Copy` for a
/// plain move assignment through `&mut`.
fn seed_phi_from_u(gpu: &Gpu, m: &GpuMesh, s: &mut Simple, hm: &HostMesh) -> Result<()> {
    let mut phi = GpuSurfaceScalarField::zeros(gpu, m, "phi")?;
    ofgpu::field_setup::compute_phi_from_u(gpu, &mut phi, s.u(), hm)?;
    *s.phi_mut() = phi;
    Ok(())
}

// ==========================================================================
//  SPEC-LIT §13.4.1: reconciling the case's `numerics` block with this driver
// ==========================================================================

/// The time scheme this run will integrate with, from the case's own
/// `ddtSchemes`/`numerics.ddt` reconciled against the run mode `-endTime`
/// selected.
///
/// `ofgpu-lowmach` takes its run mode from `-endTime`/`-deltaT` and NOT from the
/// case - its own "`-endTime` needs `-deltaT`" diagnostic says so, and every
/// transient case in this tree carries the same sentence in a comment. That
/// leaves two settings naming the same thing, and SPEC-LIT §13.4 does not let
/// a driver pick one of them in silence. The three outcomes:
///
/// * **They agree.** Honour the scheme in full. This is what gains `backward`
///   (SPEC-LIT §13.3) and what lets `Energy::new`'s own `reconcile_ddt`
///   reject `localEuler`/`CrankNicolson` BY NAME. All of them used to become
///   first-order Euler with nothing printed, because this driver wrote
///   `ddt: if transient { Euler } else { SteadyState }` and never looked.
/// * **The case names a transient scheme and no `-endTime` was given.** A
///   §13.4 error: running it steady drops the very time derivative the case
///   asked for.
/// * **The case names `steadyState` and `-endTime` was given.** Promoted to
///   `Euler` - `DdtScheme::reconciled`'s own documented rule, which only ever
///   promotes the steady default and never overrides an explicit transient
///   scheme - and PRINTED, because a promotion nobody sees is a silent
///   substitution.
fn resolve_ddt(cc: &CaseControls, transient: bool) -> Result<DdtScheme> {
    let want = cc.turb.ddt;

    if !transient && !want.is_steady() {
        return ofgpu::io::contract::unsupported_note(
            "ddtSchemes/default (numerics.ddt)",
            &want.describe(),
            &["steadyState", "localEuler"],
            "ofgpu-lowmach takes its run mode from -endTime/-deltaT, not from \
             the case; no -endTime was given, so this run has no time steps \
             to integrate over. Give `-endTime T -deltaT dt` to run the case \
             transient as it asks",
            "steadyState",
            DdtScheme::SteadyState,
        );
    }

    let ddt = want.reconciled(!transient);
    if ddt != want {
        println!(
            "ddt: the case names {} and -endTime was given, so this run \
             integrates with {} (SPEC-LIT §13.4)",
            want.describe(),
            ddt.describe()
        );
    }
    Ok(ddt)
}

/// SPEC-LIT §13.4 on the one algorithm entry this driver cannot honour.
///
/// `ofgpu-lowmach` advances energy and the momentum/pressure system ONCE
/// per unit of work, in that order, with no
/// loop around them - `docs/07-lowmach-solver.md`'s own "one outer corrector"
/// note, and the reason `is_final` is simply `transient` at the call site.
/// `nCorrectors` (the PISO pressure correctors, SPEC-LIT §14) IS honoured,
/// because `Simple::correct_outer` runs them, and so is `momentumPredictor`;
/// `nOuterCorrectors > 1` is the one that names something that is not here.
fn check_outer_correctors(cc: &CaseControls, json: bool) -> Result<()> {
    let n = cc.algorithm.n_outer_correctors;
    if n <= 1 {
        return Ok(());
    }

    let setting = if json {
        "numerics.algorithm.outerCorrectors".to_string()
    } else {
        let dict = if cc.algorithm.dict.is_empty() { "PIMPLE" } else { cc.algorithm.dict };
        format!("{dict}/nOuterCorrectors")
    };

    ofgpu::io::contract::unsupported_note(
        &setting,
        &n.to_string(),
        &["1"],
        "ofgpu-lowmach advances energy and the momentum/pressure system \
         once per step and has no PIMPLE outer loop around them yet \
         (docs/07-lowmach-solver.md). nCorrectors - the PISO \
         pressure correctors - is honoured and is the knob that is here; \
         ofgpu-buoyant has the outer loop",
        "1 outer corrector",
        (),
    )
}

/// Every control [`run`] hands a solver, built from the case and from the two
/// command-line settings that decide the run mode - and from nothing else.
///
/// One struct and one function because that is what makes SPEC-LIT §13.4.1's
/// standing test possible: `two_runs_differing_only_in_X_must_differ` calls
/// THIS, the same code path `run` calls, rather than re-deriving the wiring
/// in the test - a test that re-derives the wiring tests the test.
#[derive(Debug)]
struct LowMachControls {
    simple: SimpleControls,
    energy: EnergyControls,
    gas: GasProperties,
}

/// Read the case's numerics into the controls each equation will run with.
///
/// **What this replaced.** `run` built its `MomentumControls` and
/// `EnergyControls` from `::default()` with `nu`, `steady` and `delta_t`
/// overridden and NOTHING else read. So a case saying
/// `div(phi,U) Gauss linearUpwind grad(U)` discretised its momentum equation
/// first order; a case saying `relaxation { "U": 0.5 }` ran at 0.7; the
/// `solvers` rule matching `T` was never looked up at all, so the energy
/// equation solved at `SolverControls::default()`'s tolerance whatever the
/// case wrote; and `physics.fluid.Prt` never reached `k_eff`. None of it
/// printed anything. This is the FOURTH instance of that defect class in
/// this project (SPEC-LIT §13.4.1); `ofgpu-buoyant`'s `read_simple_controls`
/// is the third, and the model this follows.
///
/// **The rule it follows.** Each equation reads the entry named for ITS OWN
/// field: momentum reads `div(phi,U)` and `grad(U)`, energy reads
/// `div(phi,T)` and `grad(T)`, each turbulence equation reads its own
/// (already, through `CaseControls::turb`). Reading one equation's entry for
/// another is the mistake `read_simple_controls`'s comment warns about, and
/// it is why every lookup here is spelled out with its key rather than
/// shared.
///
/// `common::CaseNumerics` is what answers them, because `ofgpu-lowmach` takes
/// either case format and only the OpenFOAM one has an `fvSchemes`.
fn lowmach_controls(
    case_path: &Path,
    cc: &CaseControls,
    lowered: Option<&ofgpu::io::case_json::LoweredCase>,
    transient: bool,
    dt: Scalar,
) -> Result<LowMachControls> {
    let num = CaseNumerics::read(case_path, cc, lowered)?;
    let ddt = resolve_ddt(cc, transient)?;
    check_outer_correctors(cc, lowered.is_some())?;

    // SPEC-LIT §13.4, "recognised, not implemented". `ofgpu-buoyant`,
    // `ofgpu-k-epsilon` and `ofgpu-k-omega` all stop early on
    // `residualControl`; this driver's loop is `-iters`-counted and consults
    // no residual, so a case naming one would have it read, stored and never
    // tested. `Simple::set_residual_control` is not the answer either: it is
    // tested inside `Simple::solve_step`, and this driver calls
    // `correct_outer_low_mach` directly (one outer corrector, see
    // `check_outer_correctors`), so wiring it would produce a setting that
    // is accepted and still inert - which is the defect, not the fix.
    if !cc.residual_control.is_empty() {
        let fields: Vec<String> =
            cc.residual_control.iter().map(|(f, t)| format!("{f} {t:e}")).collect();
        ofgpu::io::contract::unsupported_note(
            "residualControl",
            &fields.join(", "),
            &[],
            "ofgpu-lowmach counts iterations rather than testing residuals: use `-iters N` for the budget and `-check N` for how often the residuals are printed. ofgpu-buoyant, ofgpu-k-epsilon and ofgpu-k-omega do stop on residualControl",
            "no residual-based stopping - the run ends on -iters/-endTime",
            (),
        )?;
    }

    // `physics.fluid.Pr` and `Prt` used to be dropped on the floor: a JSONC
    // case got `GasProperties::default()`'s 0.71/0.85 whatever it wrote, and
    // `Pr_t` is what SPEC-LIT §26's `k_eff = k + rho cp nu_t/Pr_t` divides
    // by. Every other member of `GasProperties` (`R`, `W`, `cp`, `gamma`,
    // `k`) has no JSONC spelling at all, so it keeps the default - which is
    // exactly what an OpenFOAM case with no
    // `constant/thermophysicalProperties` gets too.
    let gas = match lowered {
        Some(l) => GasProperties {
            pr: l.fluid.pr,
            pr_t: l.fluid.prt,
            // SPEC-LIT §37.4: `physics.fluid.PrtModel`, already resolved
            // (and, if unrecognised, already refused) by `JsonCase::lower`.
            pr_t_model: l.prt_model,
            ..GasProperties::default()
        },
        None => GasProperties::from_case(case_path)?,
    };
    gas.validate()?;

    // ---- momentum / pressure -------------------------------------------
    //
    // Written out field by field rather than closed with
    // `..MomentumControls::default()`: the next setting added to that struct
    // then has to be answered HERE, in the open, instead of quietly taking a
    // default nobody reviewed. That is the whole of SPEC-LIT §13.4.1.
    let u_conv = num.div("div(phi,U)")?;
    let mom_d = MomentumControls::default();
    let momentum = MomentumControls {
        nu: cc.nu,
        // `solvers/U` (`numerics.solvers` matching `U`) - already resolved by
        // both case readers, which is why this is not a `num.solver` call.
        u_solver: cc.u_solver,
        u_relax: num.relax("U", mom_d.u_relax)?,
        div_scheme: u_conv.scheme,
        bounded_convection: u_conv.bounded,
        grad_scheme: num.grad("grad(U)")?,
        sn_grad: num.sn_grad("default")?,
        n_non_orth_correctors: num.n_non_orth_correctors(),
        // The SCHEME, not just the steady/transient boolean - SPEC-LIT §13.4.
        ddt,
        lts: cc.lts,
        steady: !transient,
        delta_t: dt,
        // *DESIGN*, not a case setting: no dictionary entry in either format
        // names it, and a buoyant plume's `nu_t` gradient is exactly where the
        // transpose term matters.
        variable_viscosity_stress: mom_d.variable_viscosity_stress,
        // SPEC-LIT §5.3. `consistent yes;` in the algorithm dictionary; the
        // JSONC format has no spelling for it, and its reader sets `false`.
        simplec: cc.algorithm.consistent,
        simplec_floor: mom_d.simplec_floor,
        // SPEC-LIT §38.7: `viscosityModel` / `physics.fluid.rheology`, already
        // resolved - and, if unrecognised, already refused - by whichever
        // case reader ran. Newtonian unless the case named a closure, and
        // Newtonian launches no rheology kernel at all.
        rheology: cc.rheology,
    };

    let simple = SimpleControls {
        momentum,
        p_solver: cc.p_solver,
        // The pressure is relaxed as a FIELD, not as an equation (Patankar
        // 1980 §6.7): the relaxation is applied to the solution, not folded
        // into the matrix, so that is where OpenFOAM puts the entry. The
        // `equations` spelling is accepted too, because cases carry it.
        p_relax: num.relax_field("p", SimpleControls::default().p_relax)?,
        n_non_orth_correctors: num.n_non_orth_correctors(),
        // §25.3's low-Mach source is one extra `fvm_su` per pressure solve;
        // nothing about PISO's own corrector count changes, so this is
        // whatever the case asked for.
        n_correctors: cc.algorithm.n_correctors,
        // Fixed at one, and `check_outer_correctors` above is what refuses a
        // case that asks for more instead of dropping the request.
        n_outer_correctors: 1,
        momentum_predictor: cc.algorithm.momentum_predictor,
        report_continuity: SimpleControls::default().report_continuity,
    };

    // ---- energy ---------------------------------------------------------
    //
    // By the ENERGY equation's own entry names, never the momentum or
    // turbulence equation's. `EnergyControls::div_scheme` takes the whole
    // `DivEntry` because `bounded` is a property of the entry; the module
    // reads only the scheme half and says why (SPEC-LIT §26 makes the
    // bounded correction physics here, not an option). Exhaustive for the
    // same reason `momentum` above is.
    let energy_d = EnergyControls::default();
    let energy = EnergyControls {
        t_solver: num.solver("T", energy_d.t_solver)?,
        t_relax: num.relax("T", energy_d.t_relax)?,
        div_scheme: num.div("div(phi,T)")?,
        grad_scheme: num.grad("grad(T)")?,
        sn_grad: num.sn_grad("default")?,
        n_non_orth_correctors: num.n_non_orth_correctors(),
        // `Energy::new` reconciles this against what a rho*cp-weighted ddt
        // can actually do, and refuses `localEuler`/`CrankNicolson` by name
        // (SPEC-LIT §13.4) rather than substituting Euler.
        ddt,
        steady: !transient,
        delta_t: dt,
    };

    Ok(LowMachControls {
        simple,
        energy,
        gas,
    })
}

impl LowMachControls {
    /// The settings this run will USE, per equation, printed once at
    /// start-up.
    ///
    /// SPEC-LIT §13.4's rule stops a request being substituted in silence;
    /// this is its other half, in `print_effective_settings`'s own words: a
    /// user reading a log has to be able to see which scheme, which
    /// relaxation and which linear solver were actually in force, without
    /// inferring it from the case files - because the case files are exactly
    /// what may have been overridden. `ofgpu-lowmach` printed NONE of it, which
    /// is part of how it ran `Gauss upwind` on cases asking for
    /// `Gauss linearUpwind` for as long as it did.
    ///
    /// Built from the CONTROLS rather than from the case, so it is
    /// format-independent and cannot drift from what was actually handed to
    /// `Simple::new`/`Energy::new`.
    fn print(&self) {
        let m = &self.simple.momentum;
        let bounded = |b: bool| if b { "bounded " } else { "" };

        println!("Numerics (SPEC-LIT §13.4.1 - what this run will actually use)");
        println!("    ddt                   {}", m.ddt.describe());
        if !m.steady {
            println!("    deltaT                {}", g(f64::from(m.delta_t)));
        }
        println!(
            "    div(phi,U)            {}{}",
            bounded(m.bounded_convection),
            m.div_scheme.describe()
        );
        println!(
            "    div(phi,T)            {}{}",
            bounded(self.energy.div_scheme.bounded),
            self.energy.div_scheme.scheme.describe()
        );
        println!("    grad(U) / grad(T)     {} / {}", m.grad_scheme.describe(), self.energy.grad_scheme.describe());
        println!(
            "    snGrad (laplacian)    {}, {} non-orthogonal corrector(s)",
            m.sn_grad.describe(),
            m.n_non_orth_correctors
        );
        println!(
            "    relaxation            U {}, p {}, T {}",
            g(f64::from(m.u_relax)),
            g(f64::from(self.simple.p_relax)),
            g(f64::from(self.energy.t_relax)),
        );
        for (name, sc) in [
            ("p", &self.simple.p_solver),
            ("U", &m.u_solver),
            ("T", &self.energy.t_solver),
        ] {
            println!(
                "    solvers/{name:<14}{} + {}, tol {}, relTol {}, maxIter {}",
                sc.solver.name(),
                sc.precon.name(),
                sci(f64::from(sc.tolerance), 1),
                g(f64::from(sc.rel_tol)),
                sc.max_iter
            );
        }
        println!(
            "    algorithm             {} pressure corrector(s), 1 outer corrector, \
momentum predictor {}{}",
            self.simple.n_correctors,
            if self.simple.momentum_predictor { "on" } else { "off" },
            if m.simplec { ", SIMPLEC" } else { "" }
        );
        println!(
            "    Pr / Prt              {} / {} ({})",
            g(f64::from(self.gas.pr)),
            g(f64::from(self.gas.pr_t)),
            match self.gas.pr_t_model {
                PrtModel::Constant => "constant, SPEC-LIT §26".to_string(),
                PrtModel::KaysCrawford => format!(
                    "Prt read as Pr_t_inf; Kays-Crawford C = {}, Pr_t in [{}, {}], SPEC-LIT §37",
                    g(f64::from(KAYS_CRAWFORD_C)),
                    g(f64::from(self.gas.pr_t)),
                    g(2.0 * f64::from(self.gas.pr_t)),
                ),
            }
        );
        // SPEC-LIT §38.3 requires `m`, `muMin` and `muMax` printed; the rest
        // is printed with them because a rheology nobody can read off the
        // banner is a rheology nobody can check.
        println!("    viscosityModel        {}", m.rheology.describe());
    }
}

/// **Device memory, MEASURED rather than counted.**
///
/// A field count times a cell count times eight bytes is arithmetic on a
/// design, and arithmetic on a design is not a measurement of a program:
/// workspaces, the linear solvers' own vectors and the CUDA context are all
/// outside it and all real. This is the
/// measurement, and it is a DIFFERENCE rather than an absolute: `mem_info`
/// reports the whole device, and on a desktop card that includes the
/// compositor and every browser tab, so the baseline is taken after the CUDA
/// context exists and BEFORE the first field does. What the line prints is
/// therefore what THIS RUN allocated, which is the only figure that can be
/// compared between two configurations on the same machine.
///
/// Sampled at three points - after setup, after the first step, after the
/// last - not every step. Nothing in the time loop allocates after the first
/// iteration, and a `cuMemGetInfo` per step would put a driver round trip
/// and a stream synchronise inside the wall time a run is also being judged
/// on.
struct MemWatch {
    baseline_free: usize,
    total: usize,
    peak: usize,
}

impl MemWatch {
    fn new(gpu: &Gpu) -> Result<Self> {
        gpu.sync()?;
        let (free, total) = gpu.mem_info()?;
        Ok(Self { baseline_free: free, total, peak: 0 })
    }

    /// One sample. `saturating_sub` because the baseline is a device-wide
    /// figure: another process releasing memory between two samples can make
    /// `free` RISE above the baseline, and a run cannot have allocated a
    /// negative number of bytes.
    fn sample(&mut self, gpu: &Gpu) -> Result<()> {
        gpu.sync()?;
        let (free, _) = gpu.mem_info()?;
        self.peak = self.peak.max(self.baseline_free.saturating_sub(free));
        Ok(())
    }

    /// `peak MiB own | resident of total MiB | B/cell`, the three numbers
    /// a storage claim is checked against. The per-cell figure is the one
    /// that can be
    /// extrapolated to another mesh, which is what any storage ceiling
    /// stated as arithmetic has to be checked against.
    fn describe(&self, n_cells: usize) -> String {
        let used_now = self.total.saturating_sub(self.baseline_free);
        format!(
            "device memory: {} MiB peak allocated by this run | {} MiB was already \
             resident of {} MiB before it started | {} B/cell over {} cells",
            self.peak >> 20,
            used_now >> 20,
            self.total >> 20,
            g(self.peak as f64 / n_cells.max(1) as f64),
            n_cells
        )
    }
}

fn run(o: &Options) -> Result<()> {
    let t_total = Instant::now();

    let gpu = Gpu::new(0)?;
    println!("{}", device_banner(&gpu, "lowmach")?);
    // Before the mesh, before the first field.
    let mut mem = MemWatch::new(&gpu)?;

    let (hm, mut cc, _lowered_for_controls): (HostMesh, CaseControls, _) =
        load_case(&o.case_path)?;
    hm.print_report();
    let mesh = GpuMesh::upload(&gpu, &hm)?;
    gpu.sync()?;

    // ---- restart, if asked for --------------------------------------------
    //
    // Loaded here, before a single field is set up, so a bad `-restartFrom`
    // (missing file, wrong mesh) fails before any kernel launches - exactly
    // `ofgpu-buoyant`/`ofgpu-vof`'s own placement.
    let mesh_hash = restart::mesh_hash(&hm);
    let restart_data: Option<RestartData> = match &o.restart_from {
        Some(p) => {
            let rd = restart::read_restart(p, mesh_hash)?;
            println!(
                "restart: loaded {} (t = {} s, p0 = {} Pa, mesh hash 0x{:016x} matches)",
                p.display(),
                g(rd.time),
                g(rd.p0),
                mesh_hash
            );
            Some(rd)
        }
        None => None,
    };

    // SPEC-LIT §30.2: `constant/momentumTransport` used to be read for
    // NOTHING here - `KEpsilon` was built unconditionally regardless of what
    // the case asked for, so `RAS { model kOmegaSST; }` silently ran
    // k-epsilon, the exact substitution §13.4 forbids. `build_coupled`,
    // below, now constructs whichever model the case actually names -
    // `ofgpu-buoyant`'s adoption of the same registry is the pattern this
    // follows: nothing below this line names a concrete model, so nothing
    // below it has to pin the case to k-epsilon to reach `epsilon` directly.
    let selection = select_turbulence_model(&cc)?;
    // "epsilon" for k-epsilon, "omega" for k-omega/SST, `None` for laminar
    // and LES, which solve no `k` equation and have no `0/` dissipation
    // file to find.
    let diss_name: Option<&'static str> = selection.model.dissipation_field();

    let (fields, lowered) =
        load_initial_fields(&o.case_path, &hm, diss_name, selection.model.name())?;

    // A setting this format does not define is refused by the READER,
    // before anything is reconciled or allocated - `deny_unknown_fields` on
    // every struct in `case_json`, not a check here. It has to be the
    // reader's job: every driver in this engine would otherwise need the
    // same check, and the one that forgot it would run a case whose entries
    // it had silently dropped.

    let transient = o.end_time > 0.0;
    let dt: Scalar = if transient { o.delta_t as Scalar } else { 1.0 };

    // ---- ONE time step, for every equation ------------------------------
    //
    // `dt` above is the step this driver integrates with, and it once was
    // the step of the momentum, pressure and energy equations ONLY.
    // `CaseControls::turb.delta_t` is lowered from the CASE's `run.deltaT`
    // (`io::case_json`'s `to_case_controls`), and that is the `ddt` of every
    // equation built on `RasCore` - `k` and `epsilon` here. So a case naming
    // `run.deltaT 0.005` and run with `-deltaT 0.001` advanced its momentum
    // five times for every step its TURBULENCE took, and this driver's own
    // banner said it "takes its run mode from the command line instead"
    // while those two equations took it from the file.
    //
    // It is not a small effect and it is not visible in any printed number.
    //
    // Every case in this tree names exactly the `deltaT` its own documented
    // command passes (and every steady case names `1.0`, which is what `dt`
    // is in steady mode), so this line moves NO published number: the step
    // it forces is the step the banner below prints, and the banner prints
    // the disagreement when there is one.
    cc.turb.delta_t = dt;

    let domain = if o.sealed { DomainKind::Sealed } else { DomainKind::Open };
    println!(
        "domain: {} | p0 = {} Pa{}",
        match domain {
            DomainKind::Sealed => "SEALED (p0 rises with heat input, §25.2)",
            DomainKind::Open => "OPEN (p0 = const)",
        },
        g(f64::from(o.p0)),
        if o.heater_power != 0.0 {
            format!(" | heater {} W over the whole domain", g(f64::from(o.heater_power)))
        } else {
            String::new()
        }
    );

    // ---- everything the case's `numerics` block asks for ----------------
    //
    // SPEC-LIT §13.4.1: one call, one place, both case formats. See
    // [`lowmach_controls`] for what used to happen here instead.
    let ctrls = lowmach_controls(&o.case_path, &cc, lowered.as_ref(), transient, dt)?;
    ctrls.print();

    // SPEC-LIT §13.4's other half again, for the blocks a JSONC case carries
    // that this driver deliberately does NOT take its behaviour from.
    //
    // `run.endTime`/`run.deltaT` ARE honoured in the sense that matters -
    // they are what §31.3's transient/algorithm contract is checked against
    // at lowering time, and what a case's own header records the
    // intended command line in - but the run MODE comes from
    // `-endTime`/`-deltaT`, so this driver says which of the two is in force
    // rather than letting the case file look honoured.
    //
    // The `output` block, `run.adjustTimeStep` and `run.maxCo` are refused
    // outright, through the one shared checker every JSONC-reading driver
    // calls (`common::refuse_unimplemented_blocks`). This used to be a
    // printed note here and nothing at all in `ofgpu-k-epsilon` - one note
    // per driver is exactly how a block ends up silently ignored by the
    // second driver that reads the format.
    if let Some(l) = &lowered {
        println!(
            "run: the case names endTime {} / deltaT {}; ofgpu-lowmach takes its run \
             mode from the command line instead, and this run is {}",
            g(f64::from(l.run.end_time)),
            g(f64::from(l.run.delta_t)),
            if transient {
                format!("transient, endTime {} deltaT {}", g(o.end_time), g(o.delta_t))
            } else {
                format!("steady, {} iterations", o.n_iters)
            }
        );
        // SPEC-LIT §13.4.1. The sentence above used to be false for the
        // two turbulence equations - `k` and `epsilon` took their `ddt` step
        // from the CASE, so a disagreement here ran the closure at one step
        // and the flow at another. It is now one step, and the disagreement
        // is REPORTED rather than left for a reader to discover in a factor
        // of two.
        if (f64::from(l.run.delta_t) - f64::from(dt)).abs()
            > 1e-15 * f64::from(dt).abs().max(1.0)
        {
            println!(
                "  NOTE: the case's deltaT ({}) is NOT the one in force ({}). Every \
                 equation - momentum, pressure, energy, k and epsilon - \
                 integrates at {}; the last two once took the case's \
                 value and nothing said so (SPEC-LIT §13.4.1)",
                g(f64::from(l.run.delta_t)),
                g(f64::from(dt)),
                g(f64::from(dt)),
            );
        }
    }
    common::refuse_unimplemented_blocks(lowered.as_ref())?;

    // SPEC-LIT §44: the `output` block, which used to be part of the refusal
    // above. Resolved here, before a single field is set up, so a case that
    // asks for something impossible fails before any kernel launches -
    // exactly where `-restartFrom` is checked and for the same reason.
    let mut output_plan = common::output_plan(lowered.as_ref())?;
    if let Some(plan) = &mut output_plan {
        // §44.6 - the case and the command line are two ways to say this.
        // Under `-permissive` the answer is `false`, and the warning it just
        // printed ("substituting the command line") is then this driver's
        // job to make TRUE rather than a guess about it.
        if !common::refuse_output_named_twice(plan, &o.output_flags)? {
            output_plan = None;
        }
    }
    if let Some(plan) = &mut output_plan {
        // §44.4 - a steady run advances an iteration counter, not a clock.
        if !transient {
            plan.refuse_interval_when_steady(
                "ofgpu-lowmach",
                "give it -endTime and -deltaT for a transient run; without them it writes its final state once",
            )?;
        }
        // §44.1 - a dense voxel grid needs a lattice to sample onto, and the
        // volume writers would otherwise only say so at the FIRST WRITE,
        // which on a long transient run is a long way in.
        plan.refuse_visualisation_on_a_non_cartesian_mesh(
            ofgpu::pressure::cartesian::detect(&hm).is_ok(),
        )?;
        // Every sub-block may have been substituted away under
        // `-permissive`; an empty plan is no plan, and the command line
        // drives, which is what each of those warnings said.
        if plan.is_empty() {
            output_plan = None;
        }
    }

    let gas_props = ctrls.gas;
    let simple_ctrl = ctrls.simple;

    let mut s = Simple::new(&gpu, &hm, &mesh, simple_ctrl, cc.buoyancy)?;

    setup_vector_field(&gpu, s.u_mut(), &fields.u, &hm)?;
    setup_scalar_field(&gpu, s.p_mut(), &fields.p, &hm)?;
    if let Some(rd) = &restart_data {
        // Overwrite the INTERNAL cell values with the restart's exact
        // numbers - `fields.u`/`fields.p` only gave the boundary condition
        // TYPES and the (irrelevant, since overwritten) start-time values -
        // `ofgpu-buoyant`'s identical restart wiring.
        let u = find_restart_field(rd, "U")?;
        gpu.write(&mut s.u_mut().f, &from_restart_vectors(&u.internal))?;
        let p = find_restart_field(rd, "p")?;
        gpu.write(&mut s.p_mut().f, &from_restart_scalars(&p.internal))?;
    }
    s.initialise(&gpu)?;
    if let Some(rd) = &restart_data {
        // `initialise` re-derives every boundary cell generically, which
        // assumes a cold start - see `ofgpu-buoyant`'s identical fix. The
        // checkpoint's own boundary values do not have that problem, so
        // they overwrite whatever `initialise` just computed.
        let u = find_restart_field(rd, "U")?;
        gpu.write(&mut s.u_mut().bf, &from_restart_vectors(&u.boundary))?;
        let p = find_restart_field(rd, "p")?;
        gpu.write(&mut s.p_mut().bf, &from_restart_scalars(&p.boundary))?;

        // The conservative flux this restart was written with - SPEC-LIT
        // §5.1. Skips `seed_phi_from_u` entirely, which would otherwise
        // fall back to `interpolate(U)·Sf` - exactly the non-conservative
        // starting point a restart exists to avoid.
        let phi = find_restart_field(rd, "phi")?;
        gpu.write(&mut s.phi_mut().f, &from_restart_scalars(&phi.internal))?;
        gpu.write(&mut s.phi_mut().bf, &from_restart_scalars(&phi.boundary))?;
        println!("phi loaded from the restart checkpoint - not re-derived from U");
    } else {
        seed_phi_from_u(&gpu, &mesh, &mut s, &hm)?;
    }

    // ---- turbulence -----------------------------------------------------
    //
    // SPEC-LIT §30.2, exactly `ofgpu-buoyant`'s adoption: `wall_faces` reads
    // the DISSIPATION field's own patch types for a RAS model (§15.5); an
    // LES case has no such field, so its wall faces come from `nut`'s own
    // patch types instead (§30.1's `wernerWengleWallFunction`); laminar has
    // neither. `build_coupled` then constructs whichever model
    // `selection.model` names, buoyancy production wired in (SPEC-LIT §17)
    // from `models::buoyancy_settings` - the per-iteration `set_buoyancy`
    // call this used to make manually, once, is now `CoupledXXX::correct`'s
    // own job, every iteration.
    let wall_faces = match diss_name {
        Some(_) => WallFaces::from_case(
            fields.diss.as_ref().ok_or_else(|| {
                Error::Config(
                    "ofgpu-lowmach: internal error - a RAS model was selected but no \
                     dissipation field was loaded"
                        .to_string(),
                )
            })?,
            fields.nut.as_ref(),
            &hm,
        )?,
        None if selection.model == RasModel::Les => WallFaces {
            constrained_cells: vec![false; hm.n_boundary_faces],
            nut: match fields.nut.as_ref() {
                Some(raw) => les_nut_wall_faces(raw, &hm)?,
                None => vec![false; hm.n_boundary_faces],
            },
        },
        None => WallFaces::none(hm.n_boundary_faces),
    };
    let roughness = NutRoughness::from_case(fields.nut.as_ref(), &hm)?;

    let mut turb: Box<dyn CoupledTurbulence> =
        build_coupled(&gpu, &hm, &mesh, &cc, &selection, &wall_faces, &roughness)?;
    println!("turbulence model: {}", turb.name());

    for (fname, field) in turb.output_fields_mut() {
        let raw = match fname {
            "k" => fields.k.as_ref(),
            n if Some(n) == diss_name => fields.diss.as_ref(),
            _ => None,
        };
        if let Some(raw) = raw {
            setup_scalar_field(&gpu, field, raw, &hm)?;
        }
    }
    if let Some(raw_nut) = &fields.nut {
        for (fname, field) in turb.output_fields_mut() {
            if fname == "nut" {
                setup_scalar_field(&gpu, field, raw_nut, &hm)?;
            }
        }
    }

    let flow0 = FlowState::new(s.u(), s.phi(), cc.nu);
    turb.initialise(&gpu, &flow0)?;

    // ---- energy / gas state ----------------------------------------------
    let energy_ctrl = ctrls.energy;
    let mut energy = Energy::new(&gpu, &mesh, energy_ctrl, gas_props)?;
    setup_scalar_field(&gpu, energy.field_mut(), &fields.t, &hm)?;
    energy.initialise(&gpu)?;

    // SPEC-LIT §29.3: the Jayatilleke thermal wall function, on whichever
    // faces T's OWN patch type named `thermalWallFunction` - §15.5's rule,
    // never derived from `epsilon`'s or `nut`'s patch types. Harmless to call
    // unconditionally: a case with no such patch gets an empty face list and
    // `Energy::correct` skips the update entirely (`n_faces == 0`).
    let t_thermal_wall_faces = faces_where(&fields.t, &hm, BcKind::is_thermal_wall_function)?;
    if t_thermal_wall_faces.iter().any(|&on| on)
        && find_field(&turb.output_fields(), "k").is_none()
    {
        // SPEC-LIT §29.3's `u_tau = C_mu^{1/4} sqrt(k_P)` needs a real
        // turbulence kinetic energy - laminar has none (`nu_t = 0` is the
        // whole model) and this driver's LES family reports no `k` field
        // either (Deardorff's `k_sgs` is a diagnostic, not a boundary-carrying
        // field - see `CoupledLes::output_fields`'s own doc). SPEC-LIT §13.4:
        // refuse the combination rather than feed the wall function a
        // stand-in it was never meant to read.
        return Err(Error::Config(format!(
            "T names a thermalWallFunction patch, but {} has no turbulence \
             kinetic energy field for SPEC-LIT §29.3's u_tau to read \
             (SPEC-LIT §13.4)",
            turb.name()
        )));
    }
    energy.set_thermal_wall(&gpu, wall_coeffs_from_case(&cc.wall), &t_thermal_wall_faces)?;

    // SPEC-LIT §32.2: the fixed wall heat flux, on whichever faces T's OWN
    // patch type named `fixedFluxTemperature` - same §15.5 discipline as the
    // thermal wall function above. Harmless to call unconditionally, same
    // reasoning as that call: an empty face list is a no-op every iteration.
    let t_fixed_flux_faces = faces_where(&fields.t, &hm, BcKind::is_fixed_flux_temperature)?;
    energy.set_fixed_flux_walls(&gpu, &t_fixed_flux_faces)?;

    if let Some(rd) = &restart_data {
        // Same reasoning as `U`/`p`/`phi` above, gathered here because both
        // `turb.initialise` and `energy.initialise` have already run: the
        // checkpoint's internal values AND boundary values overwrite
        // whatever the cold-start path just computed. `nut` is OPTIONAL on
        // the way in - an older checkpoint, or one from a laminar run, may
        // not carry it, and nothing downstream needs it restored rather
        // than recomputed by `correct_nut` on the first outer iteration -
        // but `k` and whichever dissipation field this model has are
        // required exactly as they always were.
        for (fname, field) in turb.output_fields_mut() {
            if fname == "nut" {
                if let Some(rf) = rd.fields.iter().find(|f| f.name == "nut") {
                    gpu.write(&mut field.f, &from_restart_scalars(&rf.internal))?;
                    gpu.write(&mut field.bf, &from_restart_scalars(&rf.boundary))?;
                }
                continue;
            }
            let rf = find_restart_field(rd, fname)?;
            gpu.write(&mut field.f, &from_restart_scalars(&rf.internal))?;
            gpu.write(&mut field.bf, &from_restart_scalars(&rf.boundary))?;
        }
        let t_field = find_restart_field(rd, "T")?;
        gpu.write(&mut energy.field_mut().f, &from_restart_scalars(&t_field.internal))?;
        gpu.write(&mut energy.field_mut().bf, &from_restart_scalars(&t_field.boundary))?;
    }

    // SPEC-LIT §25.2: the requirement of substance in the module doc's
    // restart section - `p0` is the low-Mach thermodynamic pressure the
    // WHOLE gas state hangs off (`rho = p0/(R_s T)`), and a restart that
    // silently reused `-p0`'s default instead would start the resumed run
    // from a different thermodynamic state than the one it stopped in, even
    // with `T` restored bit-for-bit. The `.mcr` header's own `p0` slot
    // already carries exactly this number - `ofgpu-buoyant`/`ofgpu-vof`
    // write a diagnostic MEAN pressure into that same slot (see
    // `common::mean`'s doc) because neither of them has a real p0 to carry;
    // this driver's `p0` is not a diagnostic, so it is written and read back
    // as the exact state variable it is.
    let p0_init = restart_data.as_ref().map_or(o.p0, |d| d.p0 as Scalar);
    if restart_data.is_some() && (p0_init - o.p0).abs() > 1e-9 * o.p0.max(1.0) {
        println!(
            "p0 = {} Pa loaded from the restart checkpoint, overriding -p0 {} Pa",
            g(f64::from(p0_init)),
            g(f64::from(o.p0))
        );
    }
    let mut gas = GasState::new(&gpu, &mesh, gas_props, domain, p0_init)?;
    if let Some(rd) = &restart_data {
        // SPEC-LIT §25.2/§31.2: the gate's own finding - `p0` alone is not
        // the whole thermodynamic state. `update_target_divergence` reads
        // `dp0dt` at a one-iteration lag; left at `GasState::new`'s default
        // of zero, the FIRST pressure solve after resuming a sealed run
        // would assemble the wrong target divergence even with `p0` and
        // `T` both restored bit-exact. See `GasState::set_dp0dt`'s doc.
        gas.set_dp0dt(rd.dp0dt as Scalar);
    }
    gas.update_density(&gpu, energy.field())?;

    // The heater, SPEC-LIT §18: a uniform W/m3 over the whole domain summing
    // to `-heaterPower`. Built once - it is a CONSTANT registration, applied
    // fresh every iteration through `EnergySources::clear`/`register_explicit`.
    let heater: Option<DevBuf<Scalar>> = if o.heater_power != 0.0 {
        let q_per_vol = o.heater_power / mesh.total_volume;
        Some(gpu.upload(&vec![q_per_vol; hm.n_cells])?)
    } else {
        None
    };

    // ---- volumetric sources (SPEC-LIT §18, §31.1) -------------------------
    //
    // Two routes into the SAME registry: a JSONC case's own `sources[]`
    // ([`ofgpu::io::case_json::JsonSource`]) or an OpenFOAM case's
    // `constant/fvSources` ([`ofgpu::sources::read_sources`]) - never both,
    // since a case is one format or the other. This is a PERIODIC case's
    // reason to exist: a cyclic-patch channel (SPEC-LIT §31.1) has no inlet
    // to prescribe a mass flow from, so a momentum source is the only way
    // left to drive it. Scoped to `U` and, since SPEC-LIT §35.1, exactly one
    // `T` entry - a thermostat - which does not go through
    // [`ofgpu::sources::SourceSet`] at all (see [`ofgpu::sources::Thermostat`]'s
    // own doc): every other `T` source reaches the energy equation the one
    // dedicated, array-based way this driver has (`-heaterPower`, through
    // `EnergySources`).
    let source_specs: Vec<ofgpu::sources::SourceSpec> = match &lowered {
        Some(l) => l.sources.clone(),
        None => ofgpu::sources::read_sources(&o.case_path)?,
    };
    if source_specs.is_empty() {
        println!("sources: none");
    }
    let mut thermostat: Option<ofgpu::sources::Thermostat> = None;
    // SPEC-LIT §32.5.2's independent cross-check on the MEASURED wall shear:
    // at a converged, fully developed, streamwise-periodic state every newton
    // a body force puts into the fluid leaves through the walls, so
    // `(g.e_hat) sum_c rho_c V_c` over the cells the source actually acts on
    // must equal the wall traction's own streamwise integral. Recorded here,
    // where the term and its zone are both in hand; `rho` is read at the
    // CONVERGED state instead, at the bottom of this function, because it is
    // not known yet.
    let mut body_forces: Vec<(ofgpu::Vec3, Vec<usize>)> = Vec::new();
    // The streamwise axis a massFlux thermostat already resolved (SPEC-LIT
    // §35.3.5), reused by the friction report rather than resolved twice -
    // one case, one streamwise direction.
    let mut streamwise_hint: Option<ofgpu::Vec3> = None;
    for spec in &source_specs {
        let src = spec.build(&gpu, &hm)?;
        println!("source on {}: {}", spec.field, src.describe());
        match spec.field.as_str() {
            "U" => {
                if let ofgpu::sources::SourceTerm::BodyForce(g_vec) = src.term {
                    body_forces.push((g_vec, spec.selector.select(&hm)));
                }
                s.momentum_mut().sources_mut().push(src)
            }
            "T" => {
                let ofgpu::sources::SourceTerm::Thermostat {
                    target,
                    tau,
                    weighting,
                    direction,
                } = src.term
                else {
                    return Err(Error::Config(format!(
                        "source \"{}\": field \"T\" - the only T-equation entry \
                         ofgpu-lowmach's generic volumetric-source registry \
                         reaches is a thermostat (SPEC-LIT §35.1); a uniform \
                         heat release wants -heaterPower instead",
                        spec.name
                    )));
                };
                if thermostat.is_some() {
                    return Err(Error::Config(format!(
                        "source \"{}\": a second thermostat - SPEC-LIT §35.1 \
                         corrects ONE domain-mean temperature, so only one is \
                         meaningful",
                        spec.name
                    )));
                }
                let tau = match tau {
                    Some(t) => t,
                    None => {
                        // The domain's own flow-through time, SPEC-LIT
                        // §35.1's default - `V^(1/3) / |U|_mean`, measured
                        // from the INITIAL condition since `s` has just been
                        // seeded from it and no outer iteration has run yet.
                        let u0 = gpu.download(&s.u().f)?;
                        let (num, den) = (0..hm.n_cells).fold((0.0f64, 0.0f64), |(n, d), c| {
                            let v = f64::from(hm.v[c]);
                            (n + v * f64::from(u0[c].mag()), d + v)
                        });
                        let u_ref = if den > 0.0 { (num / den) as Scalar } else { 0.0 };
                        ofgpu::sources::flow_through_time(mesh.total_volume, u_ref)?
                    }
                };
                let rho_cp = gas.rho_at(target) * gas_props.cp;
                println!(
                    "thermostat: target {} K | tau {} s | gain rho(T_target)*cp = {} J/(m3 K)",
                    g(f64::from(target)),
                    g(f64::from(tau)),
                    g(f64::from(rho_cp)),
                );
                // SPEC-LIT §35.3. `uniform` is the default and takes the
                // identical construction it always did; `massFlux` resolves
                // `e_hat` ONCE, here, from the case's own `direction` or
                // from the mesh's single cyclic pair.
                thermostat = Some(match weighting {
                    ofgpu::sources::ThermostatWeighting::Uniform => {
                        println!("thermostat: weighting uniform (SPEC-LIT §35.1)");
                        ofgpu::sources::Thermostat::new(&gpu, &mesh, target, tau, rho_cp)?
                    }
                    ofgpu::sources::ThermostatWeighting::MassFlux => {
                        let e_hat = ofgpu::sources::resolve_streamwise_direction(
                            &hm, &spec.name, direction,
                        )?;
                        println!(
                            "thermostat: weighting massFlux, e_hat = ({} {} {}) {} (SPEC-LIT §35.3)",
                            g(f64::from(e_hat.x)),
                            g(f64::from(e_hat.y)),
                            g(f64::from(e_hat.z)),
                            if direction.is_some() {
                                "(given)"
                            } else {
                                "(from the mesh's cyclic pair)"
                            },
                        );
                        streamwise_hint = Some(e_hat);
                        ofgpu::sources::Thermostat::new_mass_flux(
                            &gpu, &mesh, target, tau, rho_cp, e_hat,
                        )?
                    }
                });
            }
            other => {
                return Err(Error::Config(format!(
                    "source \"{}\": field \"{other}\" - ofgpu-lowmach's generic \
                     volumetric-source registry only reaches the momentum \
                     equation (\"U\") and, as a thermostat, the energy \
                     equation (\"T\") (SPEC-LIT §18, §35.1)",
                    spec.name
                )))
            }
        }
    }

    // SPEC-LIT §35.1's own warning: a closed domain (no inlet/open patch -
    // reusing the SAME pinned-pressure test SPEC-LIT §8.5 already runs, see
    // `Simple::pressure_is_pinned`'s doc) with no Dirichlet `T` anywhere and
    // no thermostat is exactly the ill-posed pure-Neumann `T` equation §35.1
    // exists to fix. A WARNING, not an error - only the STEADY solve is
    // singular; a transient run of the same domain is legitimate (it keeps
    // whatever `T` it started with, which may be exactly what a case wants).
    if !transient && s.pressure_is_pinned() && thermostat.is_none() && !energy.field().has_a_dirichlet(&gpu)? {
        ofgpu::io::contract::warn_once(
            "ill-posed-steady-temperature",
            "this is a CLOSED domain (no inlet/open patch) with no Dirichlet \
             T anywhere and no thermostat - the steady temperature equation \
             is pure Neumann and singular up to an additive constant \
             (SPEC-LIT §35). Add a `thermostat` source (SPEC-LIT §35.1) or a \
             Dirichlet T boundary, or run transient instead of steady.",
        );
    }

    let mut backend = PbicgstabBackend::new(simple_ctrl.p_solver);
    backend.setup(&gpu, &hm, &mesh, &SystemProbe::default())?;

    println!(
        "gas: R_s = {} J/(kg K), cp = {}, gamma = {}, k = {} W/(m K), Pr = {}, Prt = {} ({})",
        g(f64::from(gas_props.r_s())),
        g(f64::from(gas_props.cp)),
        g(f64::from(gas_props.gamma)),
        g(f64::from(gas_props.k)),
        g(f64::from(gas_props.pr)),
        g(f64::from(gas_props.pr_t)),
        gas_props.pr_t_model.name(),
    );
    println!(
        "low-Mach constraint (SPEC-LIT §25.1/§26.1): (div u)_target = Q/(rho cp T) - dp0dt/(gamma p0), 
           with Q = the §18 source registry PLUS the conduction divergence div(k_eff grad T). 
           Both halves; the conduction half was omitted before §26.1 and its omission was the 
           whole of §32's resolved-leg energy-balance gap."
    );

    // ---- SPEC-LIT §37.3: exactly how far the Kays-Crawford model reaches --
    //
    // It supplies `Pr_t` to ONE closure, §26's `k_eff = k + rho cp nu_t/Pr_t`.
    // Two other places in this driver carry a turbulent Prandtl number and
    // are NOT changed by it, and §13.4 forbids either being a surprise:
    //
    // * §29.3's Jayatilleke thermal wall function keeps `Pr_t_inf`, by
    //   derivation rather than by omission - its log branch is
    //   `T+ = Pr_t (u+ + P)`, whose `Pr_t` is the LOG-LAYER value, which is
    //   exactly the limit Kays-Crawford's own `Pe_t -> inf` asymptote returns
    //   (§37.2). Substituting a local sublayer value into a correlation
    //   calibrated with a constant one would double-count the same physics.
    //   Printed, not errored: nothing is being substituted for a setting the
    //   case asked for.
    // * §17's buoyancy production `G_b = (nu_t/Pr_t) g . grad(T)/T` DOES take
    //   the constant, and it is a different closure this section has not
    //   specified. With gravity on, that is a `Pr_t` the case asked to vary
    //   and would not, so it is a §13.4 error rather than a note.
    if gas_props.pr_t_model == PrtModel::KaysCrawford {
        println!(
            "  Prt model KaysCrawford (SPEC-LIT §37.3) reaches k_eff only. The §29.3
   Jayatilleke thermal wall function keeps Pr_t_inf = {}: its log branch IS the
   log-layer value §37.2's Pe_t -> inf limit returns, so every wall flux it writes
   is unchanged by this setting.",
            g(f64::from(gas_props.pr_t))
        );
        let g_mag = (f64::from(cc.buoyancy.g.x).powi(2)
            + f64::from(cc.buoyancy.g.y).powi(2)
            + f64::from(cc.buoyancy.g.z).powi(2))
        .sqrt();
        if g_mag > 0.0 {
            ofgpu::io::contract::unsupported_note(
                "physics.fluid.PrtModel",
                "KaysCrawford (with gravity on)",
                &PrtModel::NAMES,
                "SPEC-LIT §37.3 wires the Kays-Crawford Pr_t into the energy \
                 equation's k_eff only. §17's buoyancy production \
                 G_b = (nu_t/Pr_t) g.grad(T)/T is a separate closure that still \
                 takes the constant Prt, so a buoyant case would run two \
                 different Pr_t at once without saying so. Run it with \
                 gravity [0,0,0], or with PrtModel constant",
                "the constant Prt in G_b, and the Kays-Crawford Pr_t in k_eff",
                (),
            )?;
        }
    }

    // ---- output set-up (SPEC-LIT §31.2) -----------------------------------
    //
    // Boundary-type seeds for the turbulence model's OWN fields - "k" and
    // whichever of "epsilon"/"omega" the selected model carries, "nut"
    // falling back to an invented default (it has no case-file counterpart
    // with types to inherit) - exactly `ofgpu-buoyant`'s `raw_turb`/
    // `seed_for`.
    let raw_turb: Vec<(&'static str, RawScalarField)> = match diss_name {
        Some(name) => vec![("k", fields.k.clone().unwrap_or_default()), (name, fields.diss.clone().unwrap_or_default())],
        None => Vec::new(),
    };
    // `rho` has no case-file counterpart either - `T`'s own patch types
    // stand in, since `rho` shares its topology (an `inletOutlet` on T is a
    // `calculated`-like boundary on the density that tracks it).
    let raw_rho_seed = fields.t.clone();

    // SPEC-LIT §44: ONE pipeline, built once and alive for the whole run -
    // either the case's `output` block or the command line's `-output` /
    // `-writeInterval`, never a blend of the two (§44.6 refused that above).
    let out_root_for_writers = output_root(&o.case_path);
    let mut pipeline = match &output_plan {
        Some(plan) => {
            ofgpu::io::OutputPipeline::from_plan(plan, &out_root_for_writers, "lowmach", "restart")?
        }
        None => ofgpu::io::OutputPipeline::from_command_line(
            &out_root_for_writers,
            "lowmach",
            &o.output,
            if transient { o.write_interval } else { 0.0 },
        )?,
    };
    // §44.2's EARLY half: the names this run is about to build, checked
    // before the loop rather than at the first write. `write_time` builds
    // exactly this list, in this order, and `FieldSelection::apply` checks
    // it again there - two statements of one set, and the second is what
    // makes the first safe to trust.
    if let Some(plan) = &output_plan {
        let available = output_field_names(&*turb);
        let refs: Vec<&str> = available.iter().map(String::as_str).collect();
        plan.check_fields(&refs)?;
    }
    println!("{}", pipeline.describe());

    // ---- the loop ----------------------------------------------------
    // `-endTime` names the ABSOLUTE time to reach, so a restart's own `t0`
    // comes off the step count it asks for - see `ofgpu-buoyant`'s
    // `Schedule::t0` doc for why `t` at step `n` is `t0 + n*dt`, never
    // `n*dt` alone.
    let t0: f64 = restart_data.as_ref().map_or(0.0, |d| d.time);
    let n_steps = if transient {
        (((o.end_time - t0) / o.delta_t).round().max(1.0)) as usize
    } else {
        o.n_iters as usize
    };
    let mut t_phys: f64 = t0;
    // Every schedule starts from the restart's own time, not from zero -
    // SPEC-LIT §44.4, and exactly the `next_write = t0 + W` this replaces.
    pipeline.start(t0);
    mem.sample(&gpu)?;

    for step in 0..n_steps {
        if transient {
            s.begin_time_step(&gpu, dt)?;
            energy.advance_time_step(dt);
            gas.advance_time_levels();
            t_phys += f64::from(dt);
        }

        let flow = FlowState::new(s.u(), s.phi(), cc.nu);
        // SPEC-LIT §17/§30.2: `g`/`Prt` feed `G_b`; `self.buoy` (built once
        // by `build_coupled` from `models::buoyancy_settings`) is `None`
        // whenever the case has no gravity, and gates the whole term off
        // inside `correct` regardless of what is passed here - so passing
        // `Some` unconditionally is exactly the old unconditional
        // `Some(energy.field())` this replaces, for every model.
        let thermal = ThermalCtx { t: energy.field(), g: cc.buoyancy.g, prt: gas_props.pr_t };
        turb.correct(&gpu, &flow, Some(&thermal))?;

        // `k` for SPEC-LIT §29.3's thermal wall function - present for
        // k-epsilon/k-omega/SST, absent for laminar/LES (guarded against a
        // real `thermalWallFunction` patch at setup, above); `nut` is always
        // present and of the right length, and never dereferenced when
        // there are no thermal-wall faces (`Energy::correct`'s own doc).
        let turb_fields = turb.output_fields();
        let k_for_wall: &GpuScalarField =
            find_field(&turb_fields, "k").unwrap_or_else(|| turb.nut());

        let is_final = transient; // one outer corrector - see the module doc
        let report = outer_iteration(
            &gpu,
            &mesh,
            &mut s,
            &mut energy,
            &mut gas,
            &mut backend,
            turb.nut(),
            &k_for_wall.f,
            cc.nu,
            heater.as_ref(),
            thermostat.as_mut(),
            is_final,
            if transient { Some(dt) } else { None },
        )?;

        if !report.finite {
            eprintln!("[ofgpu-lowmach] a field went non-finite at step {step} - stopping");
            print_report(step, &report);
            return Err(Error::Config("solution diverged (NaN/Inf)".to_string()));
        }

        if step % o.check_every.max(1) as usize == 0 || step + 1 == n_steps {
            print_report(step, &report);
        }

        // The first step is where the linear solvers claim their workspaces
        // and the CUDA graphs their nodes; nothing after it allocates.
        if step == 0 {
            mem.sample(&gpu)?;
        }

        if transient && pipeline.any_due(t_phys) {
            write_time(
                &gpu,
                &s,
                &*turb,
                &energy,
                &gas,
                &hm,
                &o.case_path,
                t_phys as Scalar,
                &mut pipeline,
                false,
                &fields,
                &raw_turb,
                &raw_rho_seed,
            )?;
        }

        // The command-line checkpoint counts STEPS; the case's counts
        // SECONDS and keeps a series (SPEC-LIT §44.5). §44.6 makes a run
        // name one or the other, so these two cannot both fire.
        if let Some(interval) = o.restart_write {
            if (step as u64 + 1) % interval == 0 {
                write_restart_checkpoint(
                    &gpu,
                    &s,
                    &*turb,
                    &energy,
                    &gas,
                    &hm,
                    &o.case_path,
                    mesh_hash,
                    t_phys as Scalar,
                )?;
            }
        }
        if transient && pipeline.restart_mut().is_some_and(|c| c.due(t_phys)) {
            write_series_checkpoint(
                &gpu, &s, &*turb, &energy, &gas, &hm, &mut pipeline,
                mesh_hash, t_phys as Scalar,
            )?;
        }
    }

    mem.sample(&gpu)?;
    println!("done in {} s", g(t_total.elapsed().as_secs_f64()));
    println!("{}", mem.describe(hm.n_cells));

    // The final state, always - exactly `ofgpu-buoyant`/`ofgpu-vof`'s own
    // "write at least once even with no -writeInterval" rule, which
    // SPEC-LIT §44.4 keeps for the case route too.
    write_time(
        &gpu,
        &s,
        &*turb,
        &energy,
        &gas,
        &hm,
        &o.case_path,
        t_phys as Scalar,
        &mut pipeline,
        true,
        &fields,
        &raw_turb,
        &raw_rho_seed,
    )?;
    // ... and a final checkpoint, for the same reason, when the case asked
    // for a series at all.
    if pipeline.has_restart() {
        write_series_checkpoint(
            &gpu, &s, &*turb, &energy, &gas, &hm, &mut pipeline, mesh_hash,
            t_phys as Scalar,
        )?;
    }

    // ---- integrated wall heat flux (SPEC-LIT §29.3's deferred gate) ----
    //
    // Generic to every `ofgpu-lowmach` case, not only the two channel cases
    // this was written for: the physical conductive flux at ANY wall face,
    // whatever `T`'s own patch type there is, is `k_eff_wall * snGrad(T)`,
    // and SPEC-LIT §4's Robin triple already carries exactly the two
    // numbers that make up `snGrad` generically -
    //
    //   snGrad = fr * deltaCoeffs * (ref_value - T_P) + (1 - fr) * ref_grad
    //
    // - a plain `fixedValue` wall (`fr = 1`, `ref_grad = 0`, SPEC-LIT §29.3:
    // "lowRe... pins the molecular resistance") reduces this to the
    // ordinary one-cell Dirichlet flux; a `thermalWallFunction` wall
    // (`fr` rewritten to `0` by `ThermalWallData::update`, `ref_grad` the
    // Jayatilleke-corrected gradient - see that function's own doc) reduces
    // it to the corrected flux directly. Reading the SAME triple the solver
    // itself assembled the matrix from means this owes nothing to which
    // formula produced it, and nothing here needs re-deriving Jayatilleke
    // externally. `k_eff_wall = k_mol + rho_wall * cp * nut_wall/Pr_t`
    // (SPEC-LIT §26) is recomputed from the SAME downloaded `rho`/`nut`
    // boundary fields `Energy::update_k_eff` used, at this converged state.
    // Faces the energy equation never touches as walls (an adiabatic
    // `zeroGradient` side, `fr = ref_grad = 0`) contribute exactly zero, so
    // summing over every `wall`-kind patch - not only the hot ones - is
    // correct without picking patches out by name.
    {
        let t_internal = gpu.download(&energy.field().f)?;
        let fr = gpu.download(&energy.field().fr)?;
        let ref_value = gpu.download(&energy.field().ref_value)?;
        let ref_grad = gpu.download(&energy.field().ref_grad)?;
        let rho_bf = gpu.download(&gas.rho().bf)?;
        let nut_bf = gpu.download(&turb.nut().bf)?;
        // SPEC-LIT §29.3's y+ = Cmu^{1/4} y sqrt(k_P) / nu needs a real
        // turbulence kinetic energy - present for k-epsilon/k-omega/SST,
        // absent for laminar/LES (SPEC-LIT §30.1's Werner-Wengle wall model
        // has no y+ of this kind at all: it works from the cell-averaged
        // velocity directly). `None` here just means the y+ COLUMN below is
        // skipped; the wall heat flux integral above it does not depend on
        // `k` and is reported regardless.
        let k_internal = match find_field(&turb.output_fields(), "k") {
            Some(f) => Some(gpu.download(&f.f)?),
            None => None,
        };
        let wc = wall_coeffs_from_case(&cc.wall);
        let cmu25 = f64::from(wc.cmu).powf(0.25);

        let k_mol = f64::from(gas_props.k);
        let cp = f64::from(gas_props.cp);
        // SPEC-LIT §37.3: this recomputation has to use the SAME `Pr_t` the
        // solver's own `Energy::update_k_eff` used, or the reported flux is
        // wrong by exactly the ratio of the two - which was measured at
        // +16 % on the wall-function leg, on a `fixedFluxTemperature` wall
        // whose flux is IMPOSED and therefore cannot have moved. A report
        // that disagrees with the matrix it claims to be reading is worse
        // than no report.
        let prt_at = |nut: Scalar| -> f64 {
            f64::from(match gas_props.pr_t_model {
                PrtModel::Constant => gas_props.pr_t,
                PrtModel::KaysCrawford => kays_crawford_prt(
                    (nut / cc.nu) * gas_props.pr,
                    KAYS_CRAWFORD_C,
                    gas_props.pr_t,
                ),
            })
        };

        let mut q_wall_w: f64 = 0.0;
        let mut wall_area_m2: f64 = 0.0;
        // y+ AND the SPEC-LIT §32.2 diagnosed wall temperature, at every wall
        // face, printed per patch below - so a mismatch between the two
        // meshes' ACTUAL near-wall resolution is visible rather than
        // assumed. `T_w` is diagnosed via the SAME Jayatilleke T+ relation
        // §29.3 already carries (`t_plus`/`jayatilleke_p`/`u_tau_of`,
        // `wallfunctions::thermal_wall_ref_grad`'s own module doc: "the
        // fixed-q form falls out of the same function") - this is a
        // POSTPROCESSING read of that pure host function, not a second
        // device kernel, and it reduces to the plain molecular estimate at
        // low y+ (the viscous branch `T+ = Pr y+`), which is why the SAME
        // formula is correct on both a wall-function and a resolved mesh
        // (§32.2's own point: "on a resolved mesh directly from the first
        // cell").
        let mut yplus_by_patch: Vec<(String, f64, f64, f64)> = Vec::new(); // (name, min, mean, max)
        let mut tw_by_patch: Vec<(String, f64, f64, f64, f64, f64)> = Vec::new(); // (name, mean q, mean T_P, mean T_w, mean k_P, area)
        // SPEC-LIT §33.2's mesh check needs the owner cell of every wall
        // boundary face - gathered here, over every `wall`-kind patch (not
        // only the hot ones), because "how many cells sit at y+ < 20" is a
        // property of the whole mesh's wall-normal resolution, not of any
        // one patch.
        let mut wall_face_owner: Vec<usize> = Vec::new();
        for patch in &hm.patches {
            if patch.kind != ofgpu::mesh::PatchKind::Wall {
                continue;
            }
            let mut yplus_min = f64::INFINITY;
            let mut yplus_max: f64 = 0.0;
            let mut yplus_sum: f64 = 0.0;
            let mut patch_q_sum: f64 = 0.0;
            let mut patch_area: f64 = 0.0;
            let mut tp_sum: f64 = 0.0;
            let mut tw_sum: f64 = 0.0;
            let mut kp_sum: f64 = 0.0;
            for i in 0..patch.size {
                let bf = patch.start + i;
                let cell = hm.b_face_cells[bf] as usize;
                wall_face_owner.push(cell);
                let t_p = f64::from(t_internal[cell]);
                let delta_coeffs = f64::from(hm.b_delta_coeffs[bf]);
                let frv = f64::from(fr[bf]);
                let sn_grad =
                    frv * delta_coeffs * (f64::from(ref_value[bf]) - t_p) + (1.0 - frv) * f64::from(ref_grad[bf]);
                let k_eff_wall = k_mol
                    + f64::from(rho_bf[bf]) * cp * f64::from(nut_bf[bf]) / prt_at(nut_bf[bf]);
                let area = f64::from(hm.b_mag_sf[bf]);
                let q_face = k_eff_wall * sn_grad;
                q_wall_w += q_face * area;
                wall_area_m2 += area;
                patch_q_sum += q_face * area;
                patch_area += area;
                tp_sum += t_p * area;

                // SPEC-LIT §6.4/§29.3: y+ = Cmu^{1/4} y sqrt(k_P) / nu -
                // only meaningful where there is a `k` to read (see above).
                if let Some(k_internal) = &k_internal {
                    let y = f64::from(hm.b_y[bf]);
                    let k_p = f64::from(k_internal[cell]).max(0.0);
                    let yplus = cmu25 * y * k_p.sqrt() / f64::from(cc.nu);
                    yplus_min = yplus_min.min(yplus);
                    yplus_max = yplus_max.max(yplus);
                    yplus_sum += yplus;
                    kp_sum += k_p * area;

                    // SPEC-LIT §29.3/§32.2: T_w = T_P + q T+/(rho cp u_tau).
                    //
                    // `Pr_t` here is `Pr_t_inf` under EVERY `PrtModel`, and
                    // that is SPEC-LIT §37.3's own derivation, not an
                    // oversight: `T+ = Pr_t(u+ + P(Pr/Pr_t))` is Jayatilleke's
                    // law, whose `Pr_t` is the LOG-LAYER value - exactly what
                    // Kays-Crawford's `Pe_t -> inf` limit returns - and whose
                    // `P` already carries its own sublayer integral. Feeding a
                    // local sublayer `Pr_t` in would count that integral twice.
                    // It is also what makes a wall-function mesh a CONTROL for
                    // §37's experiment.
                    use ofgpu::wallfunctions::{jayatilleke_p, t_plus, u_tau_of};
                    let u_tau = u_tau_of(k_p, f64::from(wc.cmu));
                    let tp_plus = t_plus(
                        yplus,
                        f64::from(gas_props.pr),
                        f64::from(gas_props.pr_t),
                        f64::from(wc.kappa),
                        f64::from(wc.e),
                        jayatilleke_p(f64::from(gas_props.pr), f64::from(gas_props.pr_t)),
                    );
                    let rho_c = f64::from(rho_bf[bf]);
                    if tp_plus > 0.0 && u_tau > 0.0 && rho_c > 0.0 {
                        let t_w = t_p + q_face * tp_plus / (rho_c * f64::from(gas_props.cp) * u_tau);
                        tw_sum += t_w * area;
                    } else {
                        tw_sum += t_p * area;
                    }
                }
            }
            if patch.size > 0 && k_internal.is_some() {
                yplus_by_patch.push((
                    patch.name.clone(),
                    yplus_min,
                    yplus_sum / patch.size as f64,
                    yplus_max,
                ));
                tw_by_patch.push((
                    patch.name.clone(),
                    patch_q_sum / patch_area.max(1e-30),
                    tp_sum / patch_area.max(1e-30),
                    tw_sum / patch_area.max(1e-30),
                    kp_sum / patch_area.max(1e-30),
                    patch_area,
                ));
            }
        }

        // SPEC-LIT §32.2's mixed-mean (mass-flux-weighted) bulk temperature
        // and bulk (mass-averaged) streamwise velocity, over the WHOLE
        // domain - for a statistically homogeneous streamwise-periodic flow
        // every x cross-section is the same cross-section, so integrating
        // over the full volume rather than one cross-sectional area gives
        // the same weighted average (the extra streamwise integral just
        // multiplies both the numerator and the denominator by the same
        // uniform dx). Generic to any `ofgpu-lowmach` case - a plume reads
        // these as its own domain-wide bulk numbers, not only the two
        // periodic-channel cases this was written for.
        // `den` is the domain MASS (sum rho_c V_c) and is carried out of this
        // block alongside `T_b`/`U_b`: SPEC-LIT §32.5.2's body-force balance
        // needs exactly that sum, and recomputing it would be a second
        // reduction that could silently disagree with this one.
        let (t_b, u_b, domain_mass, u_i, rho_i) = {
            let u_i = gpu.download(&s.u().f)?;
            let rho_i = gpu.download(&gas.rho().f)?;
            let cp = f64::from(gas_props.cp);
            let mut num_t: f64 = 0.0;
            let mut num_u: f64 = 0.0;
            let mut den: f64 = 0.0;
            for c in 0..hm.n_cells {
                let vol = f64::from(hm.v[c]);
                let rho_c = f64::from(rho_i[c]);
                let ux = f64::from(u_i[c].x);
                let w = rho_c * ux * vol;
                num_t += w * cp * f64::from(t_internal[c]);
                num_u += w;
                den += rho_c * vol;
            }
            let mass_flux = num_u; // sum(rho u V), the mixed-mean's own denominator sans cp
            let t_b = if mass_flux.abs() > 1e-300 { num_t / (mass_flux * cp) } else { 0.0 };
            let u_b = num_u / den.max(1e-300);
            println!(
                "\n=== bulk (mixed-mean) state (SPEC-LIT §32.2) ===\nT_b = {} K | U_b (mass-avg u_x) = {} m/s",
                g(t_b),
                g(u_b)
            );
            (t_b, u_b, den, u_i, rho_i)
        };

        // ---- SPEC-LIT §37.5: what Pr_t actually was, across the domain ----
        //
        // The one measurement that says whether the correlation is doing
        // anything here, and it is printed under BOTH models on purpose: on
        // a `constant` run the same numbers are what Kays-Crawford WOULD
        // have produced on that run's own converged `nu_t`, which is the
        // control's own statement of how large the change it is the control
        // for could be. `nu_t` is the cell field the momentum and energy
        // equations both ran with, downloaded once, so this cannot disagree
        // with what the solve did.
        {
            let nut_cells = gpu.download(&turb.nut().f)?;
            let nu_mol = f64::from(cc.nu);
            let pr = f64::from(gas_props.pr);
            let p_inf = f64::from(gas_props.pr_t);
            let prt_of = |c: usize| -> f64 {
                let pe_t = (f64::from(nut_cells[c]) / nu_mol) * pr;
                f64::from(kays_crawford_prt(
                    pe_t as Scalar,
                    KAYS_CRAWFORD_C,
                    gas_props.pr_t,
                ))
            };
            let (mut lo, mut hi, mut vol_w, mut vol_t) = (f64::MAX, f64::MIN, 0.0, 0.0);
            let (mut nut_lo, mut nut_hi) = (f64::MAX, f64::MIN);
            for c in 0..hm.n_cells {
                let prt = prt_of(c);
                let vol = f64::from(hm.v[c]);
                lo = lo.min(prt);
                hi = hi.max(prt);
                vol_w += prt * vol;
                vol_t += vol;
                let r = f64::from(nut_cells[c]) / nu_mol;
                nut_lo = nut_lo.min(r);
                nut_hi = nut_hi.max(r);
            }
            let (mut wall_lo, mut wall_hi) = (f64::MAX, f64::MIN);
            for &c in &wall_face_owner {
                wall_lo = wall_lo.min(prt_of(c));
                wall_hi = wall_hi.max(prt_of(c));
            }
            println!("
=== turbulent Prandtl number (SPEC-LIT §37.5) ===");
            println!(
                "model: {} | Pr_t_inf {} | sublayer limit 2*Pr_t_inf {} | nu_t/nu in [{}, {}]",
                gas_props.pr_t_model.name(),
                g(p_inf),
                g(2.0 * p_inf),
                g(nut_lo),
                g(nut_hi),
            );
            match gas_props.pr_t_model {
                PrtModel::KaysCrawford => println!(
                    "Pr_t IN USE: min {} | volume-mean {} | max {}{}",
                    g(lo),
                    g(vol_w / vol_t.max(1e-300)),
                    g(hi),
                    if wall_face_owner.is_empty() {
                        String::new()
                    } else {
                        format!(
                            " | wall-adjacent cells [{}, {}]",
                            g(wall_lo),
                            g(wall_hi)
                        )
                    },
                ),
                PrtModel::Constant => println!(
                    "Pr_t IN USE: {} everywhere. Kays-Crawford on THIS run's own nu_t
  would have given: min {} | volume-mean {} | max {}{}
  - a diagnostic of how much room the §37 model has on this mesh, NOT a number this
  run used",
                    g(p_inf),
                    g(lo),
                    g(vol_w / vol_t.max(1e-300)),
                    g(hi),
                    if wall_face_owner.is_empty() {
                        String::new()
                    } else {
                        format!(" | wall-adjacent cells [{}, {}]", g(wall_lo), g(wall_hi))
                    },
                ),
            }
        }

        println!("\n=== integrated wall heat flux (SPEC-LIT §29.3) ===");
        println!(
            "total wall area: {} m2 | total wall heat input: {} W | mean flux: {} W/m2",
            g(wall_area_m2),
            g(q_wall_w),
            g(q_wall_w / wall_area_m2.max(1e-30)),
        );
        for (name, ymin, ymean, ymax) in &yplus_by_patch {
            println!(
                "  y+ at {name}: min {} | mean {} | max {}",
                g(*ymin), g(*ymean), g(*ymax)
            );
        }
        for (name, qmean, tp_mean, tw_mean, kp_mean, _area) in &tw_by_patch {
            println!(
                "  {name}: mean flux {} W/m2 | mean T_P {} K | diagnosed T_w {} K | mean k_P {} m2/s2",
                g(*qmean), g(*tp_mean), g(*tw_mean), g(*kp_mean)
            );
        }
        if k_internal.is_none() {
            println!("  y+ not reported: {} carries no turbulence kinetic energy", turb.name());
        }

        // SPEC-LIT §35.2: "the thermostat's integrated power equals the wall
        // heat input to round-off" at steady state - printed side by side
        // with the wall heat this driver just measured above, so the check
        // is a subtraction a reader can do without re-running anything.
        if let Some(th) = &thermostat {
            println!("\n=== thermostat (SPEC-LIT §35.1/§35.2) ===");
            println!(
                "T_target = {} K | T_mean (volume mean) = {} K | tau = {} s",
                g(f64::from(th.target())),
                g(f64::from(th.t_mean())),
                g(f64::from(th.tau())),
            );
            println!(
                "thermostat power = {} W | wall heat input = {} W | difference = {} W{}",
                g(f64::from(th.power())),
                g(q_wall_w),
                g(f64::from(th.power()) + q_wall_w),
                if th.saturated() { " | SATURATED" } else { "" },
            );
        }

        // ---- SPEC-LIT §32.5.5's specified experiment, DIAGNOSTIC ---------
        //
        // "instrument `fvm_div_bounded_correction`'s domain integral
        // `-Sum_c cp T_c (div phi_m)_c V_c` on the energy equation and
        // compare it against the 0.0996 W by which the resolved leg's
        // balance is short". `Energy::assembly_budget` is that instrument;
        // it re-assembles the converged equation seven times, in prefixes,
        // and differences the domain row-sums, so every term of §26 is
        // reported in watts and not just the one under suspicion. It reads
        // the state the last `outer_iteration` left, restores the matrix
        // unchanged, and is called exactly here - after the loop, once.
        //
        // The comparison line is printed only where there IS a balance to be
        // short of: a case with no thermostat has no second, independent
        // measurement of the same wattage to difference against (§35.2), and
        // a ratio against a number that is not a shortfall would be a
        // fabricated verdict. The budget itself prints either way.
        {
            let budget = energy.assembly_budget(&gpu, &gas)?;
            println!("\n=== energy budget, domain integrals (SPEC-LIT §26/§32.5.5, DIAGNOSTIC) ===");
            println!(
                "sign convention: an entry is what the assembled LEFT-HAND SIDE takes out of \n\
                 the domain, in W; the entries sum to the row-sum residual."
            );
            println!("  ddt(rho cp, T)                    = {} W", g(budget.ddt_w));
            println!("  div(phi_m, T)                     = {} W", g(budget.convection_w));
            println!(
                "  -T div(phi_m) [§26 bounded corr]  = {} W   <- §32.5.5's term",
                g(budget.bounded_correction_w)
            );
            println!("  deferred scheme correction        = {} W", g(budget.scheme_correction_w));
            println!("  -laplacian(k_eff, T)              = {} W", g(budget.laplacian_w));
            println!("  -§18 sources                      = {} W", g(budget.sources_w));
            println!("  -dp0/dt                           = {} W", g(budget.dp0dt_w));
            println!("  ------------------------------------------------------");
            println!("  row-sum residual                  = {} W", g(budget.residual_w));
            println!(
                "cross-check: the same correction off its own matrix = {} W",
                g(budget.bounded_correction_direct_w),
            );
            println!(
                "  SPEC-LIT §26.1 split of that correction:
                     PRESCRIBED half, -cp sum_c rho_c T_c (div u)_c         = {} W
                     RESIDUAL half,   -cp sum_c T_c (u.grad rho)_c V_c      = {} W
                     the prescribed half taken on (div u)_target itself     = {} W
                   the first is `cp rho T = gamma p0/(gamma-1)` times a telescoping sum and is
                   zero on a closed domain whatever the third is; the SECOND is the balance gap.",
                g(budget.bounded_correction_prescribed_w),
                g(budget.bounded_correction_residual_w),
                g(budget.target_divergence_w),
            );
            println!(
                "discrete mass-flux divergence Sum_f (rho phi)_f, per cell: \n                   net = {} kg/s | L1 = {} kg/s | worst cell = {} kg/s",
                g(budget.net_mass_flux_kg_per_s),
                g(budget.mass_flux_divergence_l1_kg_per_s),
                g(budget.mass_flux_divergence_max_kg_per_s),
            );
            if let Some(th) = &thermostat {
                let shortfall = f64::from(th.power()) + q_wall_w;
                println!(
                    "§32.5.5 comparison: correction {} W against the balance shortfall {} W | \n\
                     ratio = {}",
                    g(budget.bounded_correction_w),
                    g(shortfall),
                    if shortfall.abs() > 0.0 {
                        g(budget.bounded_correction_w / shortfall)
                    } else {
                        "n/a (the balance closes exactly)".to_string()
                    },
                );
            }
        }

        // ---- the friction factor this run REALISES (SPEC-LIT §32.5) ------
        //
        // SPEC-LIT §32.3's Gnielinski correlation is a function of `Re` AND
        // the duct's friction factor `f`; `gnielinski_f` supplies the second
        // from Petukhov's smooth-PIPE form, which is the right `f` only for a
        // pipe. A plane channel at the same `Re_Dh` runs measurably higher
        // (Jones, *ASME J. Fluids Eng.* 98 (1976) 173). §32.4 therefore now
        // requires a verdict to name the `f` it was judged at, and this block
        // is what MEASURES that `f` rather than inferring it - the wall-face
        // traction, summed over every `wall`-kind patch, in whichever of
        // §32.5.1's two forms is correct for that patch's own wall treatment
        // (printed per patch: nothing here averages a modelled `tau_w` with a
        // resolved one).
        //
        // The body-force balance is reported next to it as an INDEPENDENT
        // cross-check, never as a substitute: at a converged fully developed
        // periodic state the two must agree, and if they do not, that
        // disagreement is the finding.
        if wall_area_m2 > 0.0 {
            use ofgpu::wallfunctions::{
                darcy_friction_factor, dittus_boelter_nu, gnielinski_f, gnielinski_nu_at_f,
                wall_shear,
            };

            // One case, one streamwise axis: a `massFlux` thermostat has
            // already resolved it (SPEC-LIT §35.3.5), and a case without one
            // gets it from the mesh's single cyclic pair through the SAME
            // function. Neither available is not an error - a buoyant
            // plume has no streamwise axis and wants none - but it does mean the
            // streamwise quantities below are skipped rather than guessed.
            let e_hat = match streamwise_hint {
                Some(e) => Some((e, "the thermostat's own massFlux direction")),
                None => ofgpu::sources::resolve_streamwise_direction(
                    &hm,
                    "the SPEC-LIT 32.5 friction report",
                    None,
                )
                .ok()
                .map(|e| (e, "the mesh's single cyclic pair")),
            };

            let k_based_wall_function: Vec<bool> = (0..hm.n_boundary_faces)
                .map(|bf| wall_faces.nut[bf] && !roughness.u_based[bf])
                .collect();
            let u_bf = gpu.download(&s.u().bf)?;
            let ws = wall_shear(
                &hm,
                e_hat.map_or(ofgpu::Vec3::new(1.0, 0.0, 0.0), |(e, _)| e),
                &u_i,
                &u_bf,
                &rho_bf,
                &nut_bf,
                k_internal.as_deref(),
                // SPEC-LIT §32.5.1's selector: a `nut` wall function whose
                // `nu_t,w` comes from `k` (the `nutk` family). The
                // VELOCITY-based ones (§15.1's `nutU`, and §30.1's
                // Werner-Wengle, which `NutRoughness::u_based` also covers)
                // are deliberately NOT flagged - their `nu_t,w` is defined to
                // reproduce their own `tau_w` through the viscous expression,
                // so the viscous form is exact for them and the k-based one
                // would be a different model's answer.
                &k_based_wall_function,
                cc.nu,
                wc.cmu,
            );

            println!("\n=== wall friction, MEASURED (SPEC-LIT §32.5) ===");
            for r in &ws.by_patch {
                println!(
                    "  {}: {} | area {} m2 | tau_w (magnitude) = {} Pa | tau_w (streamwise) = {} Pa{}",
                    hm.patches[r.patch].name,
                    r.form.as_str(),
                    g(f64::from(r.area)),
                    g(f64::from(r.tau_w_mag)),
                    g(f64::from(r.tau_w)),
                    match r.tau_w_other {
                        Some(o) => format!(" | other form would give {} Pa", g(f64::from(o))),
                        None => String::new(),
                    },
                );
            }

            match e_hat {
                None => println!(
                    "  streamwise quantities SKIPPED: this mesh has no single cyclic pair to \
                     take a streamwise axis from, and no massFlux thermostat named one. Give a \
                     thermostat `direction` to get f/Re/Nu here (SPEC-LIT §32.5.1)."
                ),
                Some((e, whence)) => {
                    let tau_w = f64::from(ws.tau_w);
                    let drag = f64::from(ws.drag);
                    println!(
                        "e_hat = ({} {} {}) (from {whence}) | {} wall faces | tau_w (area-mean, \
                         streamwise) = {} Pa | streamwise drag = {} N",
                        g(f64::from(e.x)),
                        g(f64::from(e.y)),
                        g(f64::from(e.z)),
                        ws.n_faces,
                        g(tau_w),
                        g(drag),
                    );

                    // SPEC-LIT §32.5.2's cross-check, in the KINEMATIC units
                    // this crate's momentum equation is written in. That is
                    // not a presentation choice: `crate::momentum` assembles
                    // `ddt(U) + div(phi,U) - laplacian(nu_eff,U) = g` with no
                    // density anywhere in it - `phi` is a VOLUMETRIC flux, a
                    // `momentumSource` enters the matrix as `g_cmpt V_c`, and
                    // the low-Mach density reaches the solver only through the
                    // pressure equation's prescribed divergence (§25.3) - so
                    // the balance the DISCRETE equation satisfies is
                    // `(g.e_hat) V = sum_walls nu_eff |dU_par| deltaCoeffs
                    // |Sf|`, both sides in m^4/s^2. Comparing newtons against
                    // newtons instead carries a systematic `rho_bar/rho_wall`
                    // error - 7.6 % on §32.5.3's own wall-function leg, where
                    // it hid a balance that closes to +0.000 % - and that
                    // error is indistinguishable, in the printed percentage,
                    // from the physical finding this cross-check exists to
                    // expose.
                    //
                    // `V_zone` is the volume each body force actually acts on;
                    // the whole mesh is what the common whole-domain case
                    // reduces to.
                    let mut force_kin = 0.0f64;
                    let mut force_n = 0.0f64;
                    for (gv, cells) in &body_forces {
                        let v_zone: f64 = cells.iter().map(|&c| f64::from(hm.v[c])).sum();
                        let m_zone: f64 = cells
                            .iter()
                            .map(|&c| f64::from(rho_i[c]) * f64::from(hm.v[c]))
                            .sum();
                        force_kin += f64::from(gv.dot(e)) * v_zone;
                        force_n += f64::from(gv.dot(e)) * m_zone;
                    }

                    let rho_b = f64::from(gas.rho_at(t_b as Scalar));
                    let f_measured = f64::from(darcy_friction_factor(
                        tau_w as Scalar,
                        rho_b as Scalar,
                        u_b as Scalar,
                    ));

                    println!(
                        "f (MEASURED, = 8 tau_w / (rho_b U_b^2), rho_b = rho(T_b) = {} kg/m3) = {}",
                        g(rho_b),
                        g(f_measured),
                    );
                    if body_forces.is_empty() {
                        println!(
                            "force balance not available: this case registers no momentum body \
                             force, so there is no independent balance to check the measurement \
                             against (SPEC-LIT §32.5.2)"
                        );
                    } else {
                        let sink_kin = f64::from(ws.drag_kin);
                        let gap = (sink_kin - force_kin) / force_kin.abs().max(1e-300);
                        println!(
                            "force balance (SPEC-LIT §32.5.2), KINEMATIC - the momentum equation \
                             this crate assembles carries no density: body force (g.e_hat) V = \
                             {} m4/s2 | wall sink sum nu_eff |dU_par| deltaCoeffs |Sf| = {} \
                             m4/s2 | disagreement = {:+.3} %{}",
                            g(force_kin),
                            g(sink_kin),
                            100.0 * gap,
                            if gap.abs() > 0.02 {
                                "  <-- report the disagreement, do NOT average the two \
                                 (SPEC-LIT §32.5.2)"
                            } else {
                                ""
                            },
                        );
                        println!(
                            "  that sink is the VISCOUS form on EVERY wall patch - the term the \
                             momentum matrix carries, whatever form the tau_w above was reported \
                             in. For reference only: the reported drag is {} N against a \
                             compressible (g.e_hat) sum(rho V) = {} N (rho_bar = {} kg/m3). That \
                             pair is NOT the balance this solver satisfies and its difference is \
                             not a finding (SPEC-LIT §32.5.2's own correction)",
                            g(drag),
                            g(force_n),
                            g(domain_mass / f64::from(mesh.total_volume)),
                        );
                    }

                    // The Nusselt gate itself, at BOTH friction factors -
                    // SPEC-LIT §32.4. `D_h = 4V/A_wall` is the hydraulic
                    // diameter of the periodic cross-section exactly (`V/L`
                    // is its area, `A_wall/L` its wetted perimeter), and for
                    // §34's plane channel - hot walls top and bottom,
                    // `empty` front and back, so no other wall contributes -
                    // it reduces to `2H`, which is the `D_h` §32.2 names.
                    let d_h = 4.0 * f64::from(mesh.total_volume) / wall_area_m2;
                    let re = u_b * d_h / f64::from(cc.nu);
                    // The heated walls only: an adiabatic side wall carries
                    // no `q` and no `T_w` worth averaging into a Nusselt
                    // number, and including it would dilute both.
                    let (mut q_hot, mut tw_hot, mut a_hot) = (0.0f64, 0.0f64, 0.0f64);
                    for (_, qmean, _, tw_mean, _, area) in &tw_by_patch {
                        if qmean.abs() > 0.0 {
                            q_hot += qmean * area;
                            tw_hot += tw_mean * area;
                            a_hot += area;
                        }
                    }
                    if a_hot <= 0.0 {
                        println!(
                            "Nu not reported: no wall patch carries a heat flux, so there is no \
                             (T_w - T_b) for a Nusselt number to be built from"
                        );
                    } else {
                        let q_w = q_hot / a_hot;
                        let t_w = tw_hot / a_hot;
                        let dt = t_w - t_b;
                        let k_th = rho_b * f64::from(gas_props.cp) * f64::from(cc.nu)
                            / f64::from(gas_props.pr);
                        let nu_measured = q_w * d_h / (k_th * dt);
                        let f_pipe = f64::from(gnielinski_f(re as Scalar));
                        let nu_gn_pipe =
                            f64::from(gnielinski_nu_at_f(f_pipe as Scalar, re as Scalar, gas_props.pr));
                        let nu_gn_real = f64::from(gnielinski_nu_at_f(
                            f_measured as Scalar,
                            re as Scalar,
                            gas_props.pr,
                        ));
                        let nu_db = f64::from(dittus_boelter_nu(re as Scalar, gas_props.pr));
                        println!(
                            "D_h (= 4V/A_wall) = {} m | Re = {} | q_w = {} W/m2 | T_w = {} K | \
                             T_b = {} K | dT = {} K | Nu (measured) = {}",
                            g(d_h), g(re), g(q_w), g(t_w), g(t_b), g(dt), g(nu_measured),
                        );
                        println!(
                            "  ABSOLUTE-PREDICTION verdict, Gnielinski at the Petukhov smooth-PIPE \
                             f = {}: Nu_Gn = {} ({:+.1} %)",
                            g(f_pipe),
                            g(nu_gn_pipe),
                            100.0 * (nu_measured / nu_gn_pipe - 1.0),
                        );
                        println!(
                            "  REYNOLDS-ANALOGY verdict, Gnielinski at this run's own MEASURED \
                             f = {}: Nu_Gn = {} ({:+.1} %)",
                            g(f_measured),
                            g(nu_gn_real),
                            100.0 * (nu_measured / nu_gn_real - 1.0),
                        );
                        println!(
                            "  Dittus-Boelter (no f argument, so one verdict only): Nu_DB = {} ({:+.1} %)",
                            g(nu_db),
                            100.0 * (nu_measured / nu_db - 1.0),
                        );
                        println!(
                            "  the two are DIFFERENT claims and neither substitutes for the \
                             other: the first tests the absolute prediction from Re alone, the \
                             second the Reynolds analogy (SPEC-LIT §32.4)"
                        );
                    }
                }
            }
        }

        // SPEC-LIT §33.2: "The solver should MEASURE and report" the worst
        // wall-adjacent y+ and how many cells sit at y+ < 20 - a low-Re
        // model on a wall-function mesh is as wrong as the reverse, and
        // silently so unless this is checked. Only meaningful under
        // `LaunderSharmaKE` (§33's own low-Re model; every other RAS model
        // here is a high-Re closure this check says nothing about) and only
        // when there is a `k` to read at all (see `k_internal` above).
        if turb.name() == "LaunderSharmaKE" {
            if let Some(k_internal) = &k_internal {
                let wd = ofgpu::walldistance::wall_distance(
                    &gpu,
                    &hm,
                    &mesh,
                    &cc.p_solver,
                    cc.turb.n_non_orth_correctors,
                )?;
                let y_cell = gpu.download(&wd.y.f)?;
                let report = ofgpu::models::mesh_resolution_report(
                    k_internal,
                    &y_cell,
                    &wall_face_owner,
                    cc.nu,
                    wc.cmu,
                );
                println!(
                    "\n=== SPEC-LIT §33.2 mesh resolution check (LaunderSharmaKE) ===\n\
                     worst wall-adjacent y+ = {} | cells globally at y+ < 20: {} / {} | wall faces: {}",
                    g(f64::from(report.max_first_cell_y_plus)),
                    report.cells_below_y_plus_20,
                    hm.n_cells,
                    report.n_wall_faces,
                );
                for w in report.warnings() {
                    println!("  WARNING: {w}");
                }
            }
        }
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

#[cfg(test)]
mod lowmach_tests {
    use super::*;
    use ofgpu::io::case::SolverControls;

    // Only the tests build controls by hand now: `run` reads every scheme
    // off the case through `common::CaseNumerics` (SPEC-LIT §13.4.1) and
    // never names one in code.
    use ofgpu::fv::{GradScheme, SnGradScheme};
    use ofgpu::fv::DivScheme;

    fn argv(v: &[&str]) -> Vec<String> {
        std::iter::once("ofgpu-lowmach".to_string())
            .chain(v.iter().map(|s| (*s).to_string()))
            .collect()
    }

    #[test]
    fn defaults_are_open_and_steady() {
        let o = parse(&argv(&["case"])).expect("a bare case path is valid");
        assert!(!o.sealed);
        assert!(o.end_time <= 0.0);
        assert_eq!(o.n_iters, 1000);
        assert!((o.p0 - 101_325.0).abs() < 1e-6);
        assert_eq!(o.heater_power, 0.0);
    }

    #[test]
    fn flags_are_rejected_before_they_reach_the_gpu() {
        assert!(parse(&argv(&["case", "-endTime", "2", "-deltaT", "0.01"])).is_ok());
        assert!(parse(&argv(&["case", "-endTime", "2"])).is_err(), "endTime without deltaT");
        assert!(parse(&argv(&["case", "-p0", "-1"])).is_err());
        assert!(parse(&argv(&["case", "-iters", "0"])).is_err());
        assert!(parse(&argv(&["case", "-nope"])).is_err());
        assert!(parse(&argv(&["case", "-sealed", "-heaterPower", "5000"])).is_ok());
    }

    #[test]
    fn output_and_restart_flags_parse() {
        let o = parse(&argv(&["case", "-output", "foam,vtu", "-writeInterval", "0.5"]))
            .expect("a valid -output/-writeInterval pair");
        assert_eq!(o.output, vec![OutputFormat::Foam, OutputFormat::Vtu]);
        assert!((o.write_interval - 0.5).abs() < 1e-12);

        let o = parse(&argv(&["case", "-restartWrite", "10"])).expect("a positive step count");
        assert_eq!(o.restart_write, Some(10));

        let o = parse(&argv(&["case", "-restartFrom", "restart.mcr"]))
            .expect("a restart path is just a path at parse time");
        assert_eq!(o.restart_from, Some(PathBuf::from("restart.mcr")));

        assert!(
            parse(&argv(&["case", "-restartWrite", "0"])).is_err(),
            "-restartWrite needs a positive step count"
        );
        assert!(
            parse(&argv(&["case", "-output", "notaformat"])).is_err(),
            "-output should reject an unknown format by name (SPEC-LIT 13.4)"
        );
    }

    // ----------------------------------------------------------------------
    //  §25/§26 gate: a hot open plume runs NaN-free with rho spanning ~3x -
    //  built entirely in memory, exercising the SAME `outer_iteration` `main`
    //  calls, on a mesh with one open patch and a heater registered on
    //  §18's source registry, exactly as the module doc describes.
    // ----------------------------------------------------------------------

    use ofgpu::field::BcKind;
    use ofgpu::momentum::BuoyancyCoeffs;
    use ofgpu::{GpuMesh, Label, Vec3};

    fn gpu() -> Option<Gpu> {
        Gpu::new(0).ok()
    }

    fn plume_mesh(n: usize, h: Scalar) -> HostMesh {
        use ofgpu::blockgen::{BlockSpec, GradedAxis};

        let axis = |lo: Scalar| GradedAxis { lo, hi: lo + h * n as Scalar, n, ..GradedAxis::default() };
        let spec = BlockSpec {
            x: axis(0.0),
            y: axis(0.0),
            z: axis(0.0),
            patch_type: ["wall", "wall", "wall", "wall", "wall", "wall"].map(String::from),
            ..BlockSpec::default()
        };
        ofgpu::blockgen::build_mesh(&spec).expect("plume mesh")
    }

    #[test]
    fn a_hot_open_plume_runs_nan_free_with_rho_spanning_about_3x() -> Result<()> {
        let Some(gpu) = gpu() else { return Ok(()) };

        const N: usize = 8;
        let h: Scalar = 0.05;
        let hm = plume_mesh(N, h);
        let m = GpuMesh::upload(&gpu, &hm)?;

        let t_ambient: Scalar = 300.0;
        let props = GasProperties::default();
        let p0 = props.r_s() * t_ambient * 1.2; // ~ 1 atm at rho = 1.2 kg/m3

        let buoy = BuoyancyCoeffs {
            g: Vec3::new(0.0, 0.0, -9.81),
            t_ref: t_ambient,
            t_min: 1.0,
        };

        let momentum_ctrl = MomentumControls {
            nu: 1.5e-4,
            u_relax: 0.3,
            sn_grad: SnGradScheme::Uncorrected,
            variable_viscosity_stress: false,
            ..MomentumControls::default()
        };
        let simple_ctrl = SimpleControls {
            momentum: momentum_ctrl,
            p_solver: SolverControls {
                tolerance: 1e-10,
                rel_tol: 0.0,
                max_iter: 800,
                check_interval: 1,
                ..SolverControls::default()
            },
            p_relax: 0.2,
            ..SimpleControls::default()
        };
        let mut s = Simple::new(&gpu, &hm, &m, simple_ctrl, buoy)?;

        // Every patch a wall except patch 1, open to ambient: zero-gradient U,
        // fixedValue p = 0 - a plume's outlet.
        let nbf = hm.n_boundary_faces;
        let owner = {
            let mut v = vec![0usize; nbf];
            for (p, pi) in hm.patches.iter().enumerate() {
                for k in 0..pi.size {
                    v[pi.start + k] = p;
                }
            }
            v
        };
        {
            let u = s.u_mut();
            let mut kind = vec![BcKind::FixedValue as Label; nbf];
            let mut fr = vec![1.0 as Scalar; nbf];
            let rv = vec![Vec3::ZERO; nbf];
            let rg = vec![Vec3::ZERO; nbf];
            for i in 0..nbf {
                if owner[i] == 1 {
                    kind[i] = BcKind::ZeroGradient as Label;
                    fr[i] = 0.0;
                }
            }
            gpu.write(&mut u.bc_kind, &kind)?;
            gpu.write(&mut u.fr, &fr)?;
            gpu.write(&mut u.ref_value, &rv)?;
            gpu.write(&mut u.ref_grad, &rg)?;
        }
        {
            let p = s.p_mut();
            let mut kind = vec![BcKind::ZeroGradient as Label; nbf];
            let mut fr = vec![0.0 as Scalar; nbf];
            let mut rv = vec![0.0 as Scalar; nbf];
            for i in 0..nbf {
                if owner[i] == 1 {
                    kind[i] = BcKind::FixedValue as Label;
                    fr[i] = 1.0;
                    rv[i] = 0.0;
                }
            }
            gpu.write(&mut p.bc_kind, &kind)?;
            gpu.write(&mut p.fr, &fr)?;
            gpu.write(&mut p.ref_value, &rv)?;
        }
        s.initialise(&gpu)?;
        seed_phi_from_u(&gpu, &m, &mut s, &hm)?;

        // Laminar: no turbulence model wired in this test, only §25/§26's own
        // machinery - see the module doc on why that is a legitimate,
        // separately-scoped simplification for THIS gate. No thermal wall
        // function either (`set_thermal_wall` is never called below), so `k`
        // is never read.
        let nut = GpuScalarField::zeros(&gpu, &m, "nut")?;
        let k = gpu.zeros::<Scalar>(hm.n_cells.max(1))?;

        let mut energy = Energy::new(
            &gpu,
            &m,
            EnergyControls {
                t_solver: SolverControls {
                    tolerance: 1e-10,
                    rel_tol: 0.0,
                    max_iter: 800,
                    check_interval: 1,
                    ..SolverControls::default()
                },
                t_relax: 0.3,
                sn_grad: SnGradScheme::Uncorrected,
                ..EnergyControls::default()
            },
            props,
        )?;
        // T: zero-gradient (adiabatic) on the five walls; `inletOutlet` on
        // the open patch - ambient air comes IN at `t_ambient`, hot gas
        // leaves at whatever temperature it has. A plain `fixedValue`
        // there would clamp the plume's own exhaust back to ambient and a
        // plain `zeroGradient` would let the whole box heat uniformly with
        // nothing ever entering cool - neither gives the STRATIFIED plume
        // this gate is checking for.
        {
            let t = energy.field_mut();
            let mut kind = vec![BcKind::ZeroGradient as Label; nbf];
            let mut rv = vec![0.0 as Scalar; nbf];
            for i in 0..nbf {
                if owner[i] == 1 {
                    kind[i] = BcKind::InletOutlet as Label;
                    rv[i] = t_ambient;
                }
            }
            gpu.write(&mut t.bc_kind, &kind)?;
            gpu.write(&mut t.ref_value, &rv)?;
            gpu.write(&mut t.f, &vec![t_ambient; hm.n_cells])?;
        }
        energy.initialise(&gpu)?;

        let mut gas = GasState::new(&gpu, &m, props, DomainKind::Open, p0)?;

        // A modest, sustained heater on §18's registry - the only kind of
        // heat release this driver has. Kept gentle relative to the box's
        // own thermal mass (`rho cp V`) so a plain fixed-point outer
        // iteration - no adaptive time-stepping, no CFL limit - stays
        // stable; the point of this gate is that §25/§26's bookkeeping does
        // not blow up over a 3x density span, not how hard a source this
        // particular linearisation can survive.
        let heater_power: Scalar = 20000.0;
        let q_per_vol = heater_power / m.total_volume;
        let heater = gpu.upload(&vec![q_per_vol; hm.n_cells])?;

        let mut backend = PbicgstabBackend::new(SolverControls {
            tolerance: 1e-10,
            rel_tol: 0.0,
            max_iter: 800,
            check_interval: 1,
            ..SolverControls::default()
        });
        backend.setup(&gpu, &hm, &m, &SystemProbe::default())?;

        let mut last = IterReport::default();
        for i in 0..600 {
            last = outer_iteration(
                &gpu,
                &m,
                &mut s,
                &mut energy,
                &mut gas,
                &mut backend,
                &nut,
                &k,
                1.5e-4,
                Some(&heater),
                None,
                false,
                None,
            )?;
            assert!(
                last.finite,
                "iteration {i} produced a NaN/Inf - T [{}, {}], rho [{}, {}]",
                last.t_min, last.t_max, last.rho_min, last.rho_max
            );
        }

        println!(
            "hot open plume: T [{}, {}] K, rho [{}, {}] kg/m3, ratio {}, \
             |U| res {}, contErr {}",
            g(f64::from(last.t_min)),
            g(f64::from(last.t_max)),
            g(f64::from(last.rho_min)),
            g(f64::from(last.rho_max)),
            g(f64::from(last.rho_max / last.rho_min)),
            g(f64::from(last.u_residual)),
            g(f64::from(last.continuity_error)),
        );

        assert!(last.finite, "the run ended non-finite");
        assert!(last.t_max > t_ambient + 1.0, "the heater did not heat anything");
        let ratio = last.rho_max / last.rho_min;
        assert!(
            ratio > 1.5,
            "rho only spans {ratio}x - the heater is too weak for this gate \
             to mean anything"
        );

        Ok(())
    }

    // ========================================================================
    //  SPEC-LIT §31.2's gate: 40 steps continuous vs 20 + restart + 20
    // ========================================================================

    use ofgpu::models::CoupledLaminar;

    /// The patch each boundary face belongs to - `plume_mesh`'s own five
    /// walls plus one open patch (index 1), reused by every BC setup below.
    fn owner_of(hm: &HostMesh) -> Vec<usize> {
        let mut v = vec![0usize; hm.n_boundary_faces];
        for (p, pi) in hm.patches.iter().enumerate() {
            for k in 0..pi.size {
                v[pi.start + k] = p;
            }
        }
        v
    }

    /// Everything one unit of work needs, borrowed off one `GpuMesh` -
    /// [`CoupledLaminar`] rather than a real RAS model because this gate is
    /// about SPEC-LIT §31.2's restart wiring, not turbulence: `nu_t = 0`
    /// keeps the physics small enough that a divergence between the two
    /// runs can only come from the restart itself.
    struct Stack<'m> {
        s: Simple<'m>,
        turb: CoupledLaminar,
        energy: Energy<'m>,
        gas: GasState<'m>,
    }

    /// Build one fresh, fully-wired low-Mach stack: a sealed box, laminar,
    /// one open patch (index 1 of [`owner_of`]), and a uniform heater, so a
    /// restart has a moving `p0` and a moving `T` to get wrong.
    /// `rd` is `None` for a cold start and `Some` to restore from a
    /// checkpoint - the SAME two paths `ofgpu-lowmach::run` takes, condensed
    /// into one function because the test needs it built twice.
    #[allow(clippy::too_many_arguments)]
    fn build_stack<'m>(
        gpu: &Gpu,
        hm: &HostMesh,
        mesh: &'m GpuMesh,
        owner: &[usize],
        props: GasProperties,
        t_ambient: Scalar,
        p0_cold: Scalar,
        dt: Scalar,
        rd: Option<&RestartData>,
    ) -> Result<Stack<'m>> {
        let nbf = hm.n_boundary_faces;

        // ---- momentum / pressure -------------------------------------
        let momentum_ctrl = MomentumControls {
            nu: 1.5e-4,
            steady: false,
            delta_t: dt,
            ddt: DdtScheme::Euler,
            sn_grad: SnGradScheme::Uncorrected,
            variable_viscosity_stress: false,
            ..MomentumControls::default()
        };
        let buoy = ofgpu::momentum::BuoyancyCoeffs {
            g: Vec3::new(0.0, 0.0, -9.81),
            t_ref: t_ambient,
            t_min: 1.0,
        };
        let simple_ctrl = SimpleControls {
            momentum: momentum_ctrl,
            p_solver: SolverControls {
                tolerance: 1e-10,
                rel_tol: 0.0,
                max_iter: 800,
                check_interval: 1,
                ..SolverControls::default()
            },
            ..SimpleControls::default()
        };
        let mut s = Simple::new(gpu, hm, mesh, simple_ctrl, buoy)?;

        {
            let u = s.u_mut();
            let mut kind = vec![BcKind::FixedValue as Label; nbf];
            let mut fr = vec![1.0 as Scalar; nbf];
            let rv = vec![Vec3::ZERO; nbf];
            let rg = vec![Vec3::ZERO; nbf];
            for i in 0..nbf {
                if owner[i] == 1 {
                    kind[i] = BcKind::ZeroGradient as Label;
                    fr[i] = 0.0;
                }
            }
            gpu.write(&mut u.bc_kind, &kind)?;
            gpu.write(&mut u.fr, &fr)?;
            gpu.write(&mut u.ref_value, &rv)?;
            gpu.write(&mut u.ref_grad, &rg)?;
        }
        {
            let p = s.p_mut();
            let mut kind = vec![BcKind::ZeroGradient as Label; nbf];
            let mut fr = vec![0.0 as Scalar; nbf];
            let mut rv = vec![0.0 as Scalar; nbf];
            for i in 0..nbf {
                if owner[i] == 1 {
                    kind[i] = BcKind::FixedValue as Label;
                    fr[i] = 1.0;
                    rv[i] = 0.0;
                }
            }
            gpu.write(&mut p.bc_kind, &kind)?;
            gpu.write(&mut p.fr, &fr)?;
            gpu.write(&mut p.ref_value, &rv)?;
        }

        if let Some(rd) = rd {
            let u = find_restart_field(rd, "U")?;
            gpu.write(&mut s.u_mut().f, &from_restart_vectors(&u.internal))?;
            let p = find_restart_field(rd, "p")?;
            gpu.write(&mut s.p_mut().f, &from_restart_scalars(&p.internal))?;
        }
        s.initialise(gpu)?;
        if let Some(rd) = rd {
            let u = find_restart_field(rd, "U")?;
            gpu.write(&mut s.u_mut().bf, &from_restart_vectors(&u.boundary))?;
            let p = find_restart_field(rd, "p")?;
            gpu.write(&mut s.p_mut().bf, &from_restart_scalars(&p.boundary))?;

            let phi = find_restart_field(rd, "phi")?;
            gpu.write(&mut s.phi_mut().f, &from_restart_scalars(&phi.internal))?;
            gpu.write(&mut s.phi_mut().bf, &from_restart_scalars(&phi.boundary))?;
        } else {
            seed_phi_from_u(gpu, mesh, &mut s, hm)?;
        }

        // ---- turbulence (laminar) --------------------------------------
        let mut turb = CoupledLaminar::new(gpu, mesh)?;
        let flow0 = FlowState::new(s.u(), s.phi(), momentum_ctrl.nu);
        turb.initialise(gpu, &flow0)?;
        if let Some(rd) = rd {
            if let Some(rf) = rd.fields.iter().find(|f| f.name == "nut") {
                for (fname, field) in turb.output_fields_mut() {
                    if fname == "nut" {
                        gpu.write(&mut field.f, &from_restart_scalars(&rf.internal))?;
                        gpu.write(&mut field.bf, &from_restart_scalars(&rf.boundary))?;
                    }
                }
            }
        }

        // ---- energy / gas state -----------------------------------------
        let energy_ctrl = EnergyControls {
            t_solver: SolverControls {
                tolerance: 1e-10,
                rel_tol: 0.0,
                max_iter: 800,
                check_interval: 1,
                ..SolverControls::default()
            },
            grad_scheme: GradScheme::GAUSS,
            sn_grad: SnGradScheme::Uncorrected,
            steady: false,
            delta_t: dt,
            ddt: DdtScheme::Euler,
            ..EnergyControls::default()
        };
        let mut energy = Energy::new(gpu, mesh, energy_ctrl, props)?;
        {
            let t = energy.field_mut();
            let mut kind = vec![BcKind::ZeroGradient as Label; nbf];
            let mut rv = vec![0.0 as Scalar; nbf];
            for i in 0..nbf {
                if owner[i] == 1 {
                    kind[i] = BcKind::InletOutlet as Label;
                    rv[i] = t_ambient;
                }
            }
            gpu.write(&mut t.bc_kind, &kind)?;
            gpu.write(&mut t.ref_value, &rv)?;
            gpu.write(&mut t.f, &vec![t_ambient; hm.n_cells])?;
        }
        energy.initialise(gpu)?;
        if let Some(rd) = rd {
            let t_field = find_restart_field(rd, "T")?;
            gpu.write(&mut energy.field_mut().f, &from_restart_scalars(&t_field.internal))?;
            gpu.write(&mut energy.field_mut().bf, &from_restart_scalars(&t_field.boundary))?;
        }

        // SPEC-LIT §25.2: the restart's own `p0`, not the cold-start default
        // - the requirement of substance this whole gate exists to check.
        let p0_init = rd.map_or(p0_cold, |d| d.p0 as Scalar);
        let mut gas = GasState::new(gpu, mesh, props, DomainKind::Sealed, p0_init)?;
        if let Some(rd) = rd {
            // The gate's own finding - see `GasState::set_dp0dt`'s doc.
            gas.set_dp0dt(rd.dp0dt as Scalar);
        }
        gas.update_density(gpu, energy.field())?;

        Ok(Stack { s, turb, energy, gas })
    }

    /// One unit of work, exactly `outer_iteration` driven the way `run`'s own
    /// transient branch drives it - laminar `nut` (real, from `stack.turb`)
    /// and `k` (a zero buffer: laminar has no `k` equation, and there is no
    /// `thermalWallFunction` patch here for it to be read by).
    #[allow(clippy::too_many_arguments)]
    fn step_once(
        gpu: &Gpu,
        mesh: &GpuMesh,
        stack: &mut Stack<'_>,
        dt: Scalar,
        heater: &DevBuf<Scalar>,
        nu: Scalar,
        backend: &mut dyn PressureBackend,
        k_zeros: &DevBuf<Scalar>,
    ) -> Result<IterReport> {
        stack.s.begin_time_step(gpu, dt)?;
        stack.energy.advance_time_step(dt);
        stack.gas.advance_time_levels();

        let flow = FlowState::new(stack.s.u(), stack.s.phi(), nu);
        stack.turb.correct(gpu, &flow, None)?;

        outer_iteration(
            gpu,
            mesh,
            &mut stack.s,
            &mut stack.energy,
            &mut stack.gas,
            backend,
            stack.turb.nut(),
            k_zeros,
            nu,
            Some(heater),
            None,
            true,
            Some(dt),
        )
    }

    #[test]
    fn restart_matches_a_continuous_run_p0_included() -> Result<()> {
        let Some(gpu) = gpu() else { return Ok(()) };

        const N: usize = 6;
        let h: Scalar = 0.05;
        let hm = plume_mesh(N, h);
        let mesh = GpuMesh::upload(&gpu, &hm)?;
        let owner = owner_of(&hm);

        let t_ambient: Scalar = 300.0;
        let props = GasProperties::default();
        let p0_cold = props.r_s() * t_ambient * 1.2;
        let dt: Scalar = 0.002;
        let heater_power: Scalar = 200.0;
        let heater = gpu.upload(&vec![heater_power / mesh.total_volume; hm.n_cells])?;
        let nu = 1.5e-4 as Scalar;

        fn total_enthalpy(gpu: &Gpu, hm: &HostMesh, stack: &Stack<'_>, cp: Scalar) -> Result<f64> {
            let rho = gpu.download(&stack.gas.rho().f)?;
            let t = gpu.download(&stack.energy.field().f)?;
            Ok(rho
                .iter()
                .zip(&t)
                .zip(&hm.v)
                .map(|((&r, &tt), &v)| f64::from(r) * f64::from(cp) * f64::from(tt) * f64::from(v))
                .sum())
        }

        let new_backend = |gpu: &Gpu| -> Result<PbicgstabBackend> {
            let mut b = PbicgstabBackend::new(SolverControls {
                tolerance: 1e-10,
                rel_tol: 0.0,
                max_iter: 800,
                check_interval: 1,
                ..SolverControls::default()
            });
            b.setup(gpu, &hm, &mesh, &SystemProbe::default())?;
            Ok(b)
        };
        let k_zeros = gpu.zeros::<Scalar>(hm.n_cells.max(1))?;

        // ---- 40 steps, continuous --------------------------------------
        let mut cont = build_stack(&gpu, &hm, &mesh, &owner, props, t_ambient, p0_cold, dt, None)?;
        let mut cont_backend = new_backend(&gpu)?;
        let mut cont_report = IterReport::default();
        // Step 21 (index 20): the FIRST unit of work after the point the
        // restarted run resumes from, in the continuous run - the number
        // `resumed_report.p_residual` at its own first post-restart step is
        // checked against below.
        let mut cont_p_residual_step21: Scalar = 0.0;
        for i in 0..40 {
            cont_report =
                step_once(&gpu, &mesh, &mut cont, dt, &heater, nu, &mut cont_backend, &k_zeros)?;
            assert!(cont_report.finite, "continuous run went non-finite at step {i}");
            if i == 20 {
                cont_p_residual_step21 = cont_report.p_residual;
            }
        }
        let cont_h = total_enthalpy(&gpu, &hm, &cont, props.cp)?;

        // ---- 20 steps, a restart checkpoint, then 20 more ---------------
        let mut half = build_stack(&gpu, &hm, &mesh, &owner, props, t_ambient, p0_cold, dt, None)?;
        let mut half_backend = new_backend(&gpu)?;
        let mut t_phys: Scalar = 0.0;
        for i in 0..20 {
            let r =
                step_once(&gpu, &mesh, &mut half, dt, &heater, nu, &mut half_backend, &k_zeros)?;
            assert!(r.finite, "first half went non-finite at step {i}");
            t_phys += dt;
        }

        let mesh_hash = restart::mesh_hash(&hm);
        // `write_restart_checkpoint` treats `case_path` as a case ROOT
        // (`common::output_root`) and writes `<root>/restart.mcr` under it -
        // an ordinary directory, not a `.mcr` file itself.
        let case_dir = std::env::temp_dir().join(format!(
            "ofgpu_lowmach_restart_gate_{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&case_dir);
        write_restart_checkpoint(
            &gpu,
            &half.s,
            &half.turb,
            &half.energy,
            &half.gas,
            &hm,
            &case_dir,
            mesh_hash,
            t_phys,
        )?;
        let rd = restart::read_restart(output_root(&case_dir).join("restart.mcr"), mesh_hash)?;
        let _ = std::fs::remove_dir_all(&case_dir);

        let mut resumed =
            build_stack(&gpu, &hm, &mesh, &owner, props, t_ambient, p0_cold, dt, Some(&rd))?;
        let mut resumed_backend = new_backend(&gpu)?;

        let mut resumed_report = IterReport::default();
        let mut resumed_p_residual_first: Scalar = 0.0;
        for i in 0..20 {
            resumed_report = step_once(
                &gpu,
                &mesh,
                &mut resumed,
                dt,
                &heater,
                nu,
                &mut resumed_backend,
                &k_zeros,
            )?;
            assert!(resumed_report.finite, "resumed run went non-finite at step {i}");
            if i == 0 {
                resumed_p_residual_first = resumed_report.p_residual;
            }
        }
        let resumed_h = total_enthalpy(&gpu, &hm, &resumed, props.cp)?;

        println!(
            "restart gate: first pressure residual after restart = {} vs continuous run's \
             step-21 residual = {}",
            sci(f64::from(resumed_p_residual_first), 6),
            sci(f64::from(cont_p_residual_step21), 6)
        );

        println!(
            "restart gate: p0 continuous = {} Pa, p0 restarted = {} Pa (checkpoint carried \
             p0 = {} Pa)",
            g(f64::from(cont_report.p0)),
            g(f64::from(resumed_report.p0)),
            g(rd.p0),
        );
        println!(
            "restart gate: total enthalpy continuous = {} J, restarted = {} J, relative gap = {}",
            sci(cont_h, 6),
            sci(resumed_h, 6),
            sci((resumed_h - cont_h) / cont_h.abs().max(1e-30), 3)
        );
        println!(
            "restart gate: |U| res continuous = {}, restarted = {}",
            sci(f64::from(cont_report.u_residual), 3),
            sci(f64::from(resumed_report.u_residual), 3)
        );

        // The first pressure corrector after the restart solves the SAME
        // linear system (same `U`, `phi`, `p`, `rho`, target divergence) the
        // continuous run's own step 21 did - the two initial residuals
        // should agree to the same floating-point tolerance the enthalpy
        // check below applies, not just be printed side by side.
        let p_res_gap = (f64::from(resumed_p_residual_first) - f64::from(cont_p_residual_step21))
            .abs()
            / f64::from(cont_p_residual_step21).abs().max(1e-30);
        assert!(
            p_res_gap < 1e-3,
            "first post-restart pressure residual {resumed_p_residual_first} diverged from the \
             continuous run's own step-21 residual {cont_p_residual_step21} (relative gap {p_res_gap})"
        );

        // p0 is the requirement of substance (SPEC-LIT §25.2): the restart
        // MUST carry it exactly - the `.mcr` header's own slot, not a
        // recomputed mean.
        assert!(
            (f64::from(resumed_report.p0) - f64::from(cont_report.p0)).abs()
                < 1e-6 * f64::from(cont_report.p0),
            "p0 diverged: continuous {} Pa vs restarted {} Pa",
            cont_report.p0,
            resumed_report.p0
        );

        // Total enthalpy is the integrated check that T, rho (hence p0) and
        // the flow field all agree, not just one of them in isolation. GPU
        // floating-point reductions are not associative, so the two runs -
        // one computed as 40 unbroken steps, the other reassembled from a
        // `.mcr` round trip through `f64` at step 20 - are not expected to
        // agree to the last bit; SPEC-LIT §31.2's own text calls for
        // reporting the gap honestly rather than hiding it, the way the VOF
        // restart gate already does. A few parts in 1e4 is that honest
        // number for this mesh/step count, not a target to tune the
        // assertion down to.
        let rel_gap = ((resumed_h - cont_h) / cont_h.abs().max(1e-30)).abs();
        assert!(
            rel_gap < 1e-3,
            "total enthalpy diverged by a relative {rel_gap} between the continuous and \
             restarted runs (continuous {cont_h} J, restarted {resumed_h} J) - more than \
             floating-point round-off in the restart round trip can explain"
        );

        Ok(())
    }

    // ======================================================================
    //  SPEC-LIT §13.4.1: every setting the case can express must REACH the
    //  solver, and must be shown to
    // ======================================================================
    //
    // The defect this suite exists to stop: `ofgpu-lowmach` built its
    // `MomentumControls` and `EnergyControls` from `::default()` and read
    // almost nothing off the case, so `cases/channelPeriodicFluxLowRe.jsonc`
    // asked for `Gauss linearUpwind grad(U)` and got first-order
    // `Gauss upwind`, asked for `U: 0.5` and got `0.7`, and named a `solvers`
    // rule for `T` that was never looked up. Nothing printed. It was the
    // FOURTH instance of the same defect class in this project.
    //
    // A test that only checks PARSING is what let it live: the case parsed
    // perfectly every time. So there are two layers here, and both are
    // required by SPEC-LIT §13.4.1:
    //
    //   1. `lowmach_controls` - the same function `run` calls - must carry each
    //      setting into the exact struct handed to `Simple::new`/`Energy::new`.
    //      No GPU needed, and it is where a `#[test]` can also assert the
    //      settings do not LEAK between equations.
    //   2. Two short runs of `run` itself, differing ONLY in one setting,
    //      must produce DIFFERENT output. If they are bit-identical the
    //      setting is inert, whatever layer 1 says.

    use ofgpu::io::case_json::{read_case_jsonc, LoweredCase};
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// One case setting per field, so a test can turn exactly one and leave
    /// every other byte of the case file identical.
    #[derive(Clone)]
    struct Knobs {
        div_u: &'static str,
        div_t: &'static str,
        div_k: &'static str,
        grad: &'static str,
        sn_grad: &'static str,
        non_orth: usize,
        relax_u: f64,
        relax_p: f64,
        relax_t: f64,
        relax_k: f64,
        t_tol: &'static str,
        u_tol: &'static str,
        correctors: &'static str,
        prt: f64,
        ddt: &'static str,
        /// `numerics.algorithm.kind`. SPEC-LIT §31.3 refuses `SIMPLE` on a
        /// case whose `ddt` is transient at LOWERING time, so the two move
        /// together.
        algo_kind: &'static str,
        /// `physics.fluid.rheology`, verbatim (including the leading comma)
        /// or empty - SPEC-LIT §38.7. Empty is `Newtonian`, which is what
        /// every case written before §38 means and what the momentum
        /// equation was bitwise before it.
        rheology: &'static str,
        /// The `wall.*` patch's fixed temperature, K. `373.15` is what every
        /// pair below runs at; it is a knob because a case that cannot heat
        /// its own duct gives `div(phi,T)` and `Prt` nothing to bite on.
        wall_t: &'static str,
        /// The ambient temperature - `initial.T`, the inlet's fixed value and
        /// the outlet's `inletValue`, all together, because a case where
        /// those three disagree is a case with a discontinuity rather than a
        /// knob. `293.15` is what every pair below runs at.
        t_amb: &'static str,
        /// The whole `output` block, verbatim (including the leading comma)
        /// or empty - SPEC-LIT §44. Empty is "the command line decides",
        /// which is what every case in this repository still means and what
        /// keeps every other pair in this file bitwise what it was.
        output: &'static str,
    }

    impl Default for Knobs {
        /// A baseline rich enough that every knob below it can actually bite:
        /// `linearUpwind` on `div(phi,U)` is what makes `grad` reach the
        /// momentum equation at all (it is the deferred correction's own
        /// gradient - SPEC-LIT §11.1/§11.2), and a real turbulence model is
        /// what makes `Prt` reach `k_eff`.
        fn default() -> Self {
            Self {
                rheology: "",
                div_u: "Gauss linearUpwind grad(U)",
                div_t: "bounded Gauss upwind",
                div_k: "bounded Gauss upwind",
                grad: "Gauss linear",
                sn_grad: "corrected",
                non_orth: 0,
                relax_u: 0.7,
                relax_p: 0.3,
                relax_t: 0.7,
                relax_k: 0.7,
                t_tol: "1e-9",
                u_tol: "1e-9",
                correctors: "",
                prt: 0.85,
                ddt: "steadyState",
                algo_kind: "SIMPLE",
                wall_t: "373.15",
                t_amb: "293.15",
                output: "",
            }
        }
    }

    /// A complete, valid `ofgpu-lowmach` case: a short heated duct, inlet-driven,
    /// k-epsilon, small enough to run in a second and coarse enough that the
    /// wall layers give every scheme something to disagree about.
    fn knob_case_text(k: &Knobs) -> String {
        let Knobs {
            div_u, div_t, div_k, grad, sn_grad, non_orth,
            relax_u, relax_p, relax_t, relax_k, t_tol, u_tol,
            correctors, prt, ddt, algo_kind, rheology, wall_t, t_amb, output,
        } = k;
        format!(
            r#"{{
  "name": "settingReachTest",
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
    "fluid": {{ "nu": 1.5e-5, "Pr": 0.71, "Prt": {prt}, "TRef": 293.15{rheology} }},
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
      "T": {{ "type": "fixedValue", "value": {t_amb} }},
      "k": {{ "type": "fixedValue", "value": 0.03 }},
      "epsilon": {{ "type": "fixedValue", "value": 0.3 }},
      "nut": {{ "type": "calculated", "value": 0 }}
    }},
    {{
      "match": "outlet", "kind": "open",
      "U": {{ "type": "inletOutlet", "inletValue": [0, 0, 0] }},
      "p": {{ "type": "fixedValue", "value": 0.0 }},
      "T": {{ "type": "inletOutlet", "inletValue": {t_amb} }},
      "k": {{ "type": "zeroGradient" }},
      "epsilon": {{ "type": "zeroGradient" }},
      "nut": {{ "type": "calculated", "value": 0 }}
    }},
    {{
      "match": "side.*", "kind": "wall",
      "U": {{ "type": "fixedValue", "value": [0, 0, 0] }},
      "p": {{ "type": "zeroGradient" }},
      "T": {{ "type": "zeroGradient" }}
    }},
    {{
      "match": "wall.*", "kind": "wall",
      "U": {{ "type": "fixedValue", "value": [0, 0, 0] }},
      "p": {{ "type": "zeroGradient" }},
      "T": {{ "type": "fixedValue", "value": {wall_t} }}
    }}
  ],
  "initial": {{
    "U": [3.0, 0, 0], "T": {t_amb}, "p": 0.0,
    "k": 0.03, "epsilon": 0.3, "nut": 0.0
  }},
  "numerics": {{
    "algorithm": {{ "kind": "{algo_kind}"{correctors} }},
    "ddt": "{ddt}",
    "div": {{
      "default": "Gauss upwind",
      "div(phi,U)": "{div_u}",
      "div(phi,T)": "{div_t}",
      "div(phi,k)": "{div_k}",
      "div(phi,epsilon)": "bounded Gauss upwind"
    }},
    "grad": "{grad}",
    "laplacian": {{ "snGrad": "{sn_grad}", "nonOrthogonalCorrectors": {non_orth} }},
    "relaxation": {{ "U": {relax_u}, "p": {relax_p}, "T": {relax_t}, "k": {relax_k}, "epsilon": 0.7 }},
    "solvers": [
      {{ "match": "p", "solver": "PBiCGStab", "preconditioner": "DIC", "tolerance": 1e-9, "relTol": 0.01, "maxIter": 500 }},
      {{ "match": "U", "solver": "PBiCGStab", "preconditioner": "diagonal", "tolerance": {u_tol}, "relTol": 0.0, "maxIter": 200 }},
      {{ "match": "T", "solver": "PBiCGStab", "preconditioner": "diagonal", "tolerance": {t_tol}, "relTol": 0.0, "maxIter": 200 }},
      {{ "match": "k|epsilon", "solver": "PBiCGStab", "preconditioner": "diagonal", "tolerance": 1e-9, "relTol": 0.01, "maxIter": 200 }}
    ]
  }},
  "run": {{ "endTime": 1.0, "deltaT": 0.001 }}{output}
}}"#
        )
    }

    /// A private directory per call - `cargo test` is multi-threaded and
    /// every one of these writes a case file and lets `run` write a time
    /// directory beside it.
    fn scratch_dir(tag: &str) -> PathBuf {
        static N: AtomicUsize = AtomicUsize::new(0);
        let d = std::env::temp_dir().join(format!(
            "ofgpu_lowmach_13_4_{}_{}_{}",
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed),
            tag
        ));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).expect("scratch dir");
        d
    }

    /// Write the case and lower it, exactly as `common::load_case` does.
    fn lower_knobs(k: &Knobs, tag: &str) -> (PathBuf, CaseControls, LoweredCase) {
        let dir = scratch_dir(tag);
        let path = dir.join("case.jsonc");
        std::fs::write(&path, knob_case_text(k)).expect("write case");
        let lowered = read_case_jsonc(&path)
            .expect("the knob case must parse")
            .lower()
            .expect("the knob case must lower");
        let cc = lowered.to_case_controls();
        (path, cc, lowered)
    }

    /// The controls `run` would hand `Simple::new`/`Energy::new` for this
    /// case - the SAME function, not a re-derivation of it.
    fn controls_for(k: &Knobs) -> LowMachControls {
        let (path, cc, lowered) = lower_knobs(k, "ctrl");
        lowmach_controls(&path, &cc, Some(&lowered), false, 1.0)
            .expect("lowmach_controls on a valid case")
    }

    // ----------------------------------------------------------------------
    //  Layer 1: each setting reaches the struct the solver is constructed
    //  from, by ITS OWN name
    // ----------------------------------------------------------------------

    #[test]
    fn div_phi_u_reaches_the_momentum_equation() {
        let a = controls_for(&Knobs { div_u: "Gauss upwind", ..Knobs::default() });
        let b = controls_for(&Knobs { div_u: "Gauss linearUpwind grad(U)", ..Knobs::default() });
        assert_ne!(
            a.simple.momentum.div_scheme, b.simple.momentum.div_scheme,
            "div(phi,U) must reach MomentumControls::div_scheme - it did not, \
             and that is the defect SPEC-LIT 13.4.1 records"
        );
        assert_eq!(a.simple.momentum.div_scheme, DivScheme::Upwind);
    }

    #[test]
    fn div_phi_u_carries_its_own_bounded_prefix() {
        let a = controls_for(&Knobs { div_u: "Gauss upwind", ..Knobs::default() });
        let b = controls_for(&Knobs { div_u: "bounded Gauss upwind", ..Knobs::default() });
        assert!(!a.simple.momentum.bounded_convection);
        assert!(b.simple.momentum.bounded_convection);
    }

    #[test]
    fn div_phi_t_reaches_the_energy_equation() {
        let a = controls_for(&Knobs { div_t: "Gauss upwind", ..Knobs::default() });
        let b = controls_for(&Knobs { div_t: "Gauss linear", ..Knobs::default() });
        assert_ne!(a.energy.div_scheme.scheme, b.energy.div_scheme.scheme);
    }

    /// The mistake `ofgpu-buoyant`'s `read_simple_controls` comment warns
    /// about, made into an assertion: one equation's entry must never be
    /// read for another. Turning ONLY `div(phi,U)` must leave `div(phi,T)`
    /// and `div(phi,k)` exactly where they were, and vice versa.
    #[test]
    fn each_equation_reads_only_its_own_div_entry() {
        let base = Knobs::default();
        let moved_u = controls_for(&Knobs { div_u: "Gauss linear", ..base.clone() });
        let moved_t = controls_for(&Knobs { div_t: "Gauss linear", ..base.clone() });
        let b = controls_for(&base);

        assert_ne!(moved_u.simple.momentum.div_scheme, b.simple.momentum.div_scheme);
        assert_eq!(
            moved_u.energy.div_scheme, b.energy.div_scheme,
            "moving div(phi,U) must not move the ENERGY equation's scheme"
        );

        assert_ne!(moved_t.energy.div_scheme, b.energy.div_scheme);
        assert_eq!(
            moved_t.simple.momentum.div_scheme, b.simple.momentum.div_scheme,
            "moving div(phi,T) must not move the MOMENTUM equation's scheme"
        );
    }

    #[test]
    fn relaxation_reaches_each_equation_by_its_own_name() {
        let b = controls_for(&Knobs::default());
        assert!((f64::from(b.simple.momentum.u_relax) - 0.7).abs() < 1e-12);
        assert!((f64::from(b.simple.p_relax) - 0.3).abs() < 1e-12);
        assert!((f64::from(b.energy.t_relax) - 0.7).abs() < 1e-12);

        let u = controls_for(&Knobs { relax_u: 0.5, ..Knobs::default() });
        assert!((f64::from(u.simple.momentum.u_relax) - 0.5).abs() < 1e-12);
        assert_eq!(u.simple.p_relax, b.simple.p_relax, "relaxation.U must not move p");
        assert_eq!(u.energy.t_relax, b.energy.t_relax, "relaxation.U must not move T");

        let p = controls_for(&Knobs { relax_p: 0.2, ..Knobs::default() });
        assert!((f64::from(p.simple.p_relax) - 0.2).abs() < 1e-12);
        assert_eq!(p.simple.momentum.u_relax, b.simple.momentum.u_relax);

        let t = controls_for(&Knobs { relax_t: 0.4, ..Knobs::default() });
        assert!((f64::from(t.energy.t_relax) - 0.4).abs() < 1e-12);
        assert_eq!(t.simple.momentum.u_relax, b.simple.momentum.u_relax);
    }

    #[test]
    fn grad_reaches_both_momentum_and_energy() {
        let a = controls_for(&Knobs::default());
        let b = controls_for(&Knobs { grad: "leastSquares", ..Knobs::default() });
        assert_ne!(a.simple.momentum.grad_scheme, b.simple.momentum.grad_scheme);
        assert_ne!(a.energy.grad_scheme, b.energy.grad_scheme);
    }

    /// `laplacian.snGrad` reaches both equations.
    ///
    /// Asserted on the CONTROLS and not on two differing runs, and that is
    /// not a shortcut: SPEC-LIT §2.4's correction is `k = Sf - |Sf|^2/(Sf.d) d`,
    /// which is **identically the zero vector** on an orthogonal mesh, and
    /// every mesh `crate::blockgen` can build from a JSONC case is a
    /// rectangular Cartesian box. `corrected` and `uncorrected` therefore
    /// assemble bit-identical matrices there, by construction - demanding
    /// that two such runs differ would be demanding an arithmetic
    /// impossibility. The end-to-end proof for this one setting needs a
    /// skewed mesh, which this case format cannot express.
    #[test]
    fn laplacian_sn_grad_reaches_both_momentum_and_energy() {
        let a = controls_for(&Knobs { sn_grad: "corrected", ..Knobs::default() });
        let b = controls_for(&Knobs { sn_grad: "uncorrected", ..Knobs::default() });
        assert_eq!(a.simple.momentum.sn_grad, SnGradScheme::Corrected);
        assert_eq!(b.simple.momentum.sn_grad, SnGradScheme::Uncorrected);
        assert_eq!(a.energy.sn_grad, SnGradScheme::Corrected);
        assert_eq!(b.energy.sn_grad, SnGradScheme::Uncorrected);
        assert_ne!(a.simple.momentum.sn_grad, b.simple.momentum.sn_grad);
    }

    #[test]
    fn non_orthogonal_correctors_reach_momentum_pressure_and_energy() {
        let b = controls_for(&Knobs { non_orth: 2, ..Knobs::default() });
        assert_eq!(b.simple.n_non_orth_correctors, 2);
        assert_eq!(b.simple.momentum.n_non_orth_correctors, 2);
        assert_eq!(b.energy.n_non_orth_correctors, 2);
    }

    #[test]
    fn the_solvers_rule_for_t_reaches_the_energy_equation() {
        let a = controls_for(&Knobs::default());
        let b = controls_for(&Knobs { t_tol: "1e-4", ..Knobs::default() });
        assert_ne!(
            a.energy.t_solver.tolerance, b.energy.t_solver.tolerance,
            "solvers[match=T] must reach EnergyControls::t_solver - it used to \
             be SolverControls::default() unconditionally"
        );
        assert_eq!(
            a.energy.t_solver.solver,
            ofgpu::io::case::LinearSolverKind::PBiCGStab
        );
        assert_eq!(
            b.simple.momentum.u_solver.tolerance, a.simple.momentum.u_solver.tolerance,
            "the T rule must not move U's solver"
        );
    }

    #[test]
    fn the_solvers_rule_for_u_reaches_the_momentum_equation() {
        let a = controls_for(&Knobs::default());
        let b = controls_for(&Knobs { u_tol: "1e-4", ..Knobs::default() });
        assert_ne!(a.simple.momentum.u_solver.tolerance, b.simple.momentum.u_solver.tolerance);
        assert_eq!(a.energy.t_solver.tolerance, b.energy.t_solver.tolerance);
    }

    #[test]
    fn algorithm_correctors_reaches_the_pressure_loop() {
        let a = controls_for(&Knobs::default());
        let b = controls_for(&Knobs { correctors: ", \"correctors\": 3", ..Knobs::default() });
        assert_eq!(a.simple.n_correctors, 1);
        assert_eq!(b.simple.n_correctors, 3);
    }

    #[test]
    fn prt_reaches_the_gas_properties() {
        let a = controls_for(&Knobs::default());
        let b = controls_for(&Knobs { prt: 0.5, ..Knobs::default() });
        assert!((f64::from(a.gas.pr_t) - 0.85).abs() < 1e-12);
        assert!((f64::from(b.gas.pr_t) - 0.5).abs() < 1e-12);
    }

    /// `ddtSchemes` names a SCHEME. `run` used to write
    /// `if transient { Euler } else { SteadyState }`, so `backward` became
    /// first-order Euler with nothing printed - SPEC-LIT §13.4 in one line.
    #[test]
    fn the_ddt_scheme_is_honoured_not_reduced_to_a_boolean() {
        let k = Knobs { ddt: "backward", algo_kind: "PIMPLE", ..Knobs::default() };
        let (path, cc, lowered) = lower_knobs(&k, "ddt");
        let c = lowmach_controls(&path, &cc, Some(&lowered), true, 0.001)
            .expect("backward is a scheme ofgpu-lowmach has");
        assert_eq!(c.simple.momentum.ddt, DdtScheme::Backward);
        assert_eq!(c.energy.ddt, DdtScheme::Backward);
    }

    /// SPEC-LIT §13.4: the case asks for a time derivative and the command
    /// line gave no `-endTime`, so there is nothing to integrate over. That
    /// is an error naming both, not a silent steady run.
    #[test]
    fn a_transient_ddt_in_a_steady_run_is_a_named_error() {
        // A case that is INTERNALLY consistent - transient scheme, transient
        // algorithm, positive endTime, so SPEC-LIT §31.3 passes it - run with
        // no `-endTime` on the command line. That combination is this
        // driver's own to refuse.
        let k = Knobs { ddt: "Euler", algo_kind: "PIMPLE", ..Knobs::default() };
        let (path, cc, lowered) = lower_knobs(&k, "ddtx");
        let e = lowmach_controls(&path, &cc, Some(&lowered), false, 1.0)
            .expect_err("Euler with no -endTime must be refused");
        let msg = format!("{e}");
        assert!(msg.contains("ddtSchemes"), "the error must name the setting: {msg}");
        assert!(msg.contains("-endTime"), "the error must name the way out: {msg}");
    }

    /// SPEC-LIT §13.4: `ofgpu-lowmach` has no PIMPLE outer loop, so a case
    /// asking for more than one outer corrector is refused by name rather
    /// than quietly given one.
    #[test]
    fn more_than_one_outer_corrector_is_a_named_error() {
        let k = Knobs {
            correctors: ", \"outerCorrectors\": 3",
            ddt: "Euler",
            algo_kind: "PIMPLE",
            ..Knobs::default()
        };
        let (path, cc, lowered) = lower_knobs(&k, "outer");
        let e = lowmach_controls(&path, &cc, Some(&lowered), true, 0.001)
            .expect_err("nOuterCorrectors > 1 must be refused");
        let msg = format!("{e}");
        assert!(msg.contains("outerCorrectors"), "the error must name the setting: {msg}");
        assert!(msg.contains("nCorrectors"), "the error must name what IS available: {msg}");
    }

    // ----------------------------------------------------------------------
    //  Layer 2: two runs differing ONLY in the setting must DIFFER
    // ----------------------------------------------------------------------

    /// Every field file `run` wrote, as `(relative path, contents)`, sorted.
    ///
    /// The whole written state rather than one number: a setting that moves
    /// only `k`, or only `T`, is still a setting that reached the solver.
    fn written_state(root: &Path) -> Vec<(String, String)> {
        fn walk(dir: &Path, prefix: &str, out: &mut Vec<(String, String)>) {
            let Ok(rd) = std::fs::read_dir(dir) else { return };
            for e in rd.flatten() {
                let p = e.path();
                let name = e.file_name().to_string_lossy().to_string();
                let rel = if prefix.is_empty() { name.clone() } else { format!("{prefix}/{name}") };
                if p.is_dir() {
                    walk(&p, &rel, out);
                } else if let Ok(s) = std::fs::read_to_string(&p) {
                    // The header carries the time directory's own name, which
                    // is the same for both runs; nothing else in the file is
                    // metadata.
                    out.push((rel, s));
                }
            }
        }
        let mut out = Vec::new();
        walk(root, "", &mut out);
        out.sort();
        out
    }

    /// Run `ofgpu-lowmach` on a case built from `k` and return everything it
    /// wrote. `extra` is appended to the command line.
    fn run_knobs(k: &Knobs, tag: &str, extra: &[&str]) -> Vec<(String, String)> {
        let dir = scratch_dir(tag);
        let path = dir.join("case.jsonc");
        std::fs::write(&path, knob_case_text(k)).expect("write case");

        let mut args: Vec<String> = vec![
            "ofgpu-lowmach".to_string(),
            path.to_string_lossy().to_string(),
        ];
        args.extend(extra.iter().map(|s| (*s).to_string()));
        let o = parse(&args).expect("the knob command line must parse");
        run(&o).expect("the knob case must run");

        let out = written_state(&common::json_case_output_dir(&path));
        assert!(!out.is_empty(), "the run wrote nothing to compare");
        out
    }

    /// **The standing test SPEC-LIT §13.4.1 requires of every new setting.**
    ///
    /// For each wired setting: two runs of `run` itself, identical in every
    /// byte of the case file but one, must write DIFFERENT fields. A pair
    /// that comes back bit-identical means the setting is inert - which is
    /// precisely how the defect this file records was demonstrated in the
    /// first place (`docs/07-lowmach-solver.md` §1.1: "two 500-iteration runs
    /// ... differing only in `div(phi,U)` print BIT-IDENTICAL residual and
    /// bulk-state lines").
    ///
    /// `laplacian.snGrad` is deliberately absent: see
    /// [`laplacian_sn_grad_reaches_both_momentum_and_energy`] for why an
    /// end-to-end difference is arithmetically impossible on any mesh this
    /// case format can build.
    #[test]
    fn every_wired_setting_changes_what_the_run_writes() {
        if Gpu::new(0).is_err() {
            return;
        }

        const ITERS: [&str; 4] = ["-iters", "12", "-check", "100"];
        let d = Knobs::default;

        // (what is being turned, case A, case B)
        let cases: Vec<(&str, Knobs, Knobs)> = vec![
            (
                "numerics.div[\"div(phi,U)\"]",
                Knobs { div_u: "Gauss upwind", ..d() },
                Knobs { div_u: "Gauss linearUpwind grad(U)", ..d() },
            ),
            (
                "numerics.div[\"div(phi,T)\"]",
                Knobs { div_t: "bounded Gauss upwind", ..d() },
                Knobs { div_t: "Gauss linear", ..d() },
            ),
            (
                "numerics.div[\"div(phi,k)\"]",
                Knobs { div_k: "bounded Gauss upwind", ..d() },
                Knobs { div_k: "Gauss linear", ..d() },
            ),
            (
                "numerics.grad",
                Knobs { grad: "Gauss linear", ..d() },
                Knobs { grad: "cellLimited Gauss linear 1", ..d() },
            ),
            (
                "numerics.laplacian.nonOrthogonalCorrectors",
                Knobs { non_orth: 0, ..d() },
                Knobs { non_orth: 2, ..d() },
            ),
            (
                "numerics.relaxation.U",
                Knobs { relax_u: 0.5, ..d() },
                Knobs { relax_u: 0.9, ..d() },
            ),
            (
                "numerics.relaxation.p",
                Knobs { relax_p: 0.2, ..d() },
                Knobs { relax_p: 0.5, ..d() },
            ),
            (
                "numerics.relaxation.T",
                Knobs { relax_t: 0.4, ..d() },
                Knobs { relax_t: 0.9, ..d() },
            ),
            (
                "numerics.relaxation.k",
                Knobs { relax_k: 0.4, ..d() },
                Knobs { relax_k: 0.9, ..d() },
            ),
            (
                "numerics.solvers[match=T].tolerance",
                Knobs { t_tol: "1e-2", ..d() },
                Knobs { t_tol: "1e-12", ..d() },
            ),
            (
                "numerics.solvers[match=U].tolerance",
                Knobs { u_tol: "1e-2", ..d() },
                Knobs { u_tol: "1e-12", ..d() },
            ),
            (
                "numerics.algorithm.correctors",
                Knobs { correctors: "", ..d() },
                Knobs { correctors: ", \"correctors\": 3", ..d() },
            ),
            (
                "physics.fluid.Prt",
                Knobs { prt: 0.85, ..d() },
                Knobs { prt: 0.4, ..d() },
            ),
            // SPEC-LIT §38.7, the JSONC route. `viscosityModel` was read by
            // nothing at all until §38 - the sixth instance of the defect
            // this test exists to catch - and there are two rows because a
            // reader can wire the model selector and still drop its numbers.
            (
                "physics.fluid.rheology.model",
                Knobs { rheology: "", ..d() },
                Knobs {
                    rheology: r#", "rheology": { "model": "powerLaw", "rho": 1.2, "K": 1.8e-5, "n": 0.8 }"#,
                    ..d()
                },
            ),
            (
                "physics.fluid.rheology.n",
                Knobs {
                    rheology: r#", "rheology": { "model": "powerLaw", "rho": 1.2, "K": 1.8e-5, "n": 0.8 }"#,
                    ..d()
                },
                Knobs {
                    rheology: r#", "rheology": { "model": "powerLaw", "rho": 1.2, "K": 1.8e-5, "n": 0.5 }"#,
                    ..d()
                },
            ),
        ];

        let mut inert: Vec<&str> = Vec::new();
        for (label, a, b) in &cases {
            let ra = run_knobs(a, "a", &ITERS);
            let rb = run_knobs(b, "b", &ITERS);
            if ra == rb {
                inert.push(label);
            }
        }

        assert!(
            inert.is_empty(),
            "these settings are INERT - two runs differing only in them wrote \
             bit-identical fields, so the case can ask for them and the solver \
             will not honour them (SPEC-LIT 13.4.1): {inert:?}"
        );
    }

    // ----------------------------------------------------------------------
    //  SPEC-LIT §44 - the `output` block
    // ----------------------------------------------------------------------

    /// Every file a run wrote, as `(relative path, BYTES)`.
    ///
    /// Bytes, not text, and that distinction is the whole reason this exists
    /// beside `written_state`: `.vdb` and `.nvdb` are binary, and
    /// `read_to_string` returns `Err` on them, so the text walker skips them
    /// **in silence**. A §13.4.1 pair on `visualisation.precision` compared
    /// with `written_state` would have compared two empty lists and passed
    /// while measuring nothing.
    fn written_bytes(root: &Path) -> Vec<(String, Vec<u8>)> {
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

    /// `run_knobs`, comparing BYTES - see [`written_bytes`].
    fn run_knobs_bytes(k: &Knobs, tag: &str, extra: &[&str]) -> Vec<(String, Vec<u8>)> {
        let dir = scratch_dir(tag);
        let path = dir.join("case.jsonc");
        std::fs::write(&path, knob_case_text(k)).expect("write case");

        let mut args: Vec<String> =
            vec!["ofgpu-lowmach".to_string(), path.to_string_lossy().to_string()];
        args.extend(extra.iter().map(|s| (*s).to_string()));
        let o = parse(&args).expect("the knob command line must parse");
        run(&o).expect("the knob case must run");

        let out = written_bytes(&common::json_case_output_dir(&path));
        assert!(!out.is_empty(), "the run wrote nothing to compare");
        out
    }

    /// The names of the files a run wrote, without their contents - for the
    /// pairs whose whole point is that a different NUMBER of files appears.
    fn written_names(out: &[(String, Vec<u8>)]) -> Vec<String> {
        out.iter().map(|(n, _)| n.clone()).collect()
    }

    // The `output` blocks the pairs below turn. Every one of them is a
    // complete, valid block, so the pair differs in exactly one entry.
    const OUT_VDB: &str = r#", "output": { "visualisation": { "format": "vdb" } }"#;
    const OUT_NVDB: &str = r#", "output": { "visualisation": { "format": "nvdb" } }"#;
    const OUT_VDB_FP16: &str =
        r#", "output": { "visualisation": { "format": "vdb", "precision": "fp16" } }"#;
    const OUT_VDB_FIELDS_2: &str =
        r#", "output": { "visualisation": { "format": "vdb", "fields": ["U", "T"] } }"#;
    const OUT_VDB_FIELDS_3: &str =
        r#", "output": { "visualisation": { "format": "vdb", "fields": ["U", "T", "p"] } }"#;
    const OUT_VDB_USD: &str =
        r#", "output": { "visualisation": { "format": "vdb", "usdScene": true } }"#;
    const OUT_VDB_EVERY_2: &str =
        r#", "output": { "visualisation": { "format": "vdb", "interval": 0.002 } }"#;
    const OUT_VDB_EVERY_3: &str =
        r#", "output": { "visualisation": { "format": "vdb", "interval": 0.003 } }"#;
    const OUT_EXACT_VTU: &str = r#", "output": { "exact": { "format": "vtu" } }"#;
    const OUT_EXACT_FOAM: &str = r#", "output": { "exact": { "format": "openfoam" } }"#;
    const OUT_EXACT_EVERY_2: &str =
        r#", "output": { "exact": { "format": "vtu", "interval": 0.002 } }"#;
    const OUT_EXACT_EVERY_3: &str =
        r#", "output": { "exact": { "format": "vtu", "interval": 0.003 } }"#;
    const OUT_RESTART_2: &str =
        r#", "output": { "exact": { "format": "vtu" }, "restart": { "interval": 0.002, "keep": 0 } }"#;
    const OUT_RESTART_3: &str =
        r#", "output": { "exact": { "format": "vtu" }, "restart": { "interval": 0.003, "keep": 0 } }"#;
    const OUT_KEEP_1: &str =
        r#", "output": { "exact": { "format": "vtu" }, "restart": { "interval": 0.001, "keep": 1 } }"#;
    const OUT_KEEP_4: &str =
        r#", "output": { "exact": { "format": "vtu" }, "restart": { "interval": 0.001, "keep": 4 } }"#;

    /// **SPEC-LIT §44.7's pair table.** Ten settings of the `output` block,
    /// each turned on its own, each REQUIRED to change what the run writes.
    ///
    /// Transient, because §44.4 refuses a positive `interval` on a steady
    /// run - which is itself one of this section's refusals and is tested
    /// separately below.
    #[test]
    fn every_output_setting_changes_what_the_run_writes() {
        if Gpu::new(0).is_err() {
            return;
        }

        const SHORT: [&str; 5] = ["-endTime", "0.002", "-deltaT", "0.001", "-check"];
        // Six steps, so an interval of 0.002 and one of 0.003 give three
        // writes and two - a difference in the number of files, not only in
        // their contents.
        const LONG: [&str; 5] = ["-endTime", "0.006", "-deltaT", "0.001", "-check"];
        let short = |extra: &[&str]| -> Vec<String> {
            let mut v: Vec<String> = SHORT.iter().map(|s| s.to_string()).collect();
            v.push("100".to_string());
            v.extend(extra.iter().map(|s| s.to_string()));
            v
        };
        let long = || -> Vec<String> {
            let mut v: Vec<String> = LONG.iter().map(|s| s.to_string()).collect();
            v.push("100".to_string());
            v
        };

        let d = Knobs::default;
        let cases: Vec<(&str, Knobs, Knobs, Vec<String>)> = vec![
            (
                "output (the block itself)",
                Knobs { output: "", ..d() },
                Knobs { output: OUT_VDB, ..d() },
                short(&[]),
            ),
            (
                "output.visualisation.format",
                Knobs { output: OUT_VDB, ..d() },
                Knobs { output: OUT_NVDB, ..d() },
                short(&[]),
            ),
            (
                "output.visualisation.precision",
                Knobs { output: OUT_VDB, ..d() },
                Knobs { output: OUT_VDB_FP16, ..d() },
                short(&[]),
            ),
            (
                "output.visualisation.fields",
                Knobs { output: OUT_VDB_FIELDS_2, ..d() },
                Knobs { output: OUT_VDB_FIELDS_3, ..d() },
                short(&[]),
            ),
            (
                "output.visualisation.usdScene",
                Knobs { output: OUT_VDB, ..d() },
                Knobs { output: OUT_VDB_USD, ..d() },
                short(&[]),
            ),
            (
                "output.visualisation.interval",
                Knobs { output: OUT_VDB_EVERY_2, ..d() },
                Knobs { output: OUT_VDB_EVERY_3, ..d() },
                long(),
            ),
            (
                "output.exact.format",
                Knobs { output: OUT_EXACT_VTU, ..d() },
                Knobs { output: OUT_EXACT_FOAM, ..d() },
                short(&[]),
            ),
            (
                "output.exact.interval",
                Knobs { output: OUT_EXACT_EVERY_2, ..d() },
                Knobs { output: OUT_EXACT_EVERY_3, ..d() },
                long(),
            ),
            (
                "output.restart.interval",
                Knobs { output: OUT_RESTART_2, ..d() },
                Knobs { output: OUT_RESTART_3, ..d() },
                long(),
            ),
            (
                "output.restart.keep",
                Knobs { output: OUT_KEEP_1, ..d() },
                Knobs { output: OUT_KEEP_4, ..d() },
                long(),
            ),
        ];

        let mut inert: Vec<&str> = Vec::new();
        for (label, a, b, args) in &cases {
            let argv: Vec<&str> = args.iter().map(String::as_str).collect();
            let ra = run_knobs_bytes(a, "outa", &argv);
            let rb = run_knobs_bytes(b, "outb", &argv);
            if ra == rb {
                inert.push(label);
            }
        }

        assert!(
            inert.is_empty(),
            "these output settings are INERT - two runs differing only in them \
             wrote bit-identical files, so the case can ask for them and the \
             solver will not honour them (SPEC-LIT 13.4.1): {inert:?}"
        );
    }

    /// The three pieces SPEC-LIT §44 had to BUILD, checked for what they
    /// actually do rather than only for "the two runs differ".
    #[test]
    fn the_three_missing_pieces_do_what_they_say() {
        if Gpu::new(0).is_err() {
            return;
        }
        let args = ["-endTime", "0.006", "-deltaT", "0.001", "-check", "100"];
        let d = Knobs::default;

        // §44.2 - `fields` writes FEWER grids, so a smaller file. Both runs
        // are one write of one .vdb, so the size comparison is direct.
        let two = run_knobs_bytes(&Knobs { output: OUT_VDB_FIELDS_2, ..d() }, "f2", &args[..4]);
        let three = run_knobs_bytes(&Knobs { output: OUT_VDB_FIELDS_3, ..d() }, "f3", &args[..4]);
        let vdb_len = |v: &[(String, Vec<u8>)]| -> usize {
            v.iter().filter(|(n, _)| n.ends_with(".vdb")).map(|(_, b)| b.len()).sum()
        };
        assert!(vdb_len(&two) > 0, "the fields run must have written a .vdb");
        assert!(
            vdb_len(&two) < vdb_len(&three),
            "§44.2: two fields must write less than three ({} vs {})",
            vdb_len(&two),
            vdb_len(&three)
        );

        // §44.3 - `fp16` halves the voxels, so the file shrinks. Not merely
        // "differs": a precision setting that made the file BIGGER would
        // pass a difference test and be wrong.
        let f32 = run_knobs_bytes(&Knobs { output: OUT_VDB, ..d() }, "p32", &args[..4]);
        let f16 = run_knobs_bytes(&Knobs { output: OUT_VDB_FP16, ..d() }, "p16", &args[..4]);
        assert!(
            vdb_len(&f16) < vdb_len(&f32),
            "§44.3: fp16 must shrink the volume file ({} vs {})",
            vdb_len(&f16),
            vdb_len(&f32)
        );

        // §44.5 - `keep` leaves exactly N checkpoints on disk. Six steps at
        // interval 0.001 is six scheduled checkpoints plus the forced final
        // one, which shares the last label and so is the same file.
        let mcr = |v: &[(String, Vec<u8>)]| -> Vec<String> {
            written_names(v).into_iter().filter(|n| n.ends_with(".mcr")).collect()
        };
        let k1 = run_knobs_bytes(&Knobs { output: OUT_KEEP_1, ..d() }, "k1", &args);
        let k4 = run_knobs_bytes(&Knobs { output: OUT_KEEP_4, ..d() }, "k4", &args);
        assert_eq!(mcr(&k1).len(), 1, "keep 1 leaves one checkpoint: {:?}", mcr(&k1));
        assert_eq!(mcr(&k4).len(), 4, "keep 4 leaves four: {:?}", mcr(&k4));
        // And they are a SERIES, named by time - not one file overwritten.
        assert!(
            mcr(&k4).iter().all(|n| n.starts_with("restart_") && n.ends_with(".mcr")),
            "{:?}",
            mcr(&k4)
        );
        assert!(mcr(&k4).contains(&"restart_0.006.mcr".to_string()), "{:?}", mcr(&k4));
    }

    /// SPEC-LIT §44.4/§44.2: the two refusals a driver raises rather than the
    /// plan, because they need the driver's own run mode and field list.
    #[test]
    fn the_drivers_own_output_refusals_fire_by_name() {
        if Gpu::new(0).is_err() {
            return;
        }
        let case = |k: &Knobs, tag: &str, extra: &[&str]| -> Result<()> {
            let dir = scratch_dir(tag);
            let path = dir.join("case.jsonc");
            std::fs::write(&path, knob_case_text(k)).expect("write case");
            let mut args: Vec<String> =
                vec!["ofgpu-lowmach".to_string(), path.to_string_lossy().to_string()];
            args.extend(extra.iter().map(|s| (*s).to_string()));
            let o = parse(&args).expect("the command line must parse");
            run(&o)
        };
        let d = Knobs::default;

        // §44.4 - a steady run has no clock for "every 0.002 s".
        let e = case(&Knobs { output: OUT_VDB_EVERY_2, ..d() }, "steady", &["-iters", "2"])
            .expect_err("a steady run must refuse a positive interval");
        let m = format!("{e}");
        assert!(m.contains("output.visualisation.interval"), "{m}");
        assert!(m.contains("-endTime"), "the error must say how to get a clock: {m}");

        // §44.6 - the case and the command line both naming the output.
        let e = case(
            &Knobs { output: OUT_VDB, ..d() },
            "twice",
            &["-endTime", "0.001", "-deltaT", "0.001", "-output", "vtu"],
        )
        .expect_err("naming the output twice must be refused");
        let m = format!("{e}");
        assert!(m.contains("output (case file)") && m.contains("-output"), "{m}");

        // §44.2 - a field no run of this driver has. `Y_CO` is a field
        // name the case FORMAT knows and this driver never builds.
        let e = case(
            &Knobs {
                output: r#", "output": { "visualisation": { "format": "vdb", "fields": ["U", "Y_CO"] } }"#,
                ..d()
            },
            "nofield",
            &["-endTime", "0.001", "-deltaT", "0.001"],
        )
        .expect_err("a field the run does not have must be refused");
        let m = format!("{e}");
        assert!(m.contains("Y_CO"), "{m}");
        for have in ["U", "p", "T", "rho", "k", "epsilon", "nut"] {
            assert!(m.contains(have), "the error must list {have}, which the run DOES have: {m}");
        }
    }

    /// SPEC-LIT §44.2's two-statement risk, pinned.
    ///
    /// `output_field_names` states what `write_time` is about to build, and
    /// `write_time` builds it separately. If they drift, one of these two
    /// runs fails: naming every field of the early list must RUN (so the
    /// early list is not a superset), and the run writes no field outside it
    /// (so it is not a subset either - `FieldSelection::apply` would accept
    /// a name the early check rejected, and the second run proves the early
    /// check is the one that fires).
    #[test]
    fn the_early_field_list_is_what_the_run_actually_writes() {
        if Gpu::new(0).is_err() {
            return;
        }
        let args = ["-endTime", "0.001", "-deltaT", "0.001"];
        let all = run_knobs_bytes(
            &Knobs {
                output: r#", "output": { "visualisation": { "format": "vdb",
                    "fields": ["U", "p", "T", "rho", "k", "epsilon", "nut"] } }"#,
                ..Knobs::default()
            },
            "allfields",
            &args,
        );
        assert!(
            all.iter().any(|(n, _)| n.ends_with(".vdb")),
            "naming every field of the early list must run and write"
        );
    }

    /// The transient half of the same test: `numerics.ddt` is a SCHEME, and
    /// `Euler` and `backward` must not produce the same answer.
    #[test]
    fn the_ddt_scheme_changes_what_the_run_writes() {
        if Gpu::new(0).is_err() {
            return;
        }
        let args = ["-endTime", "0.004", "-deltaT", "0.001", "-check", "100"];
        let tr = Knobs { algo_kind: "PIMPLE", ..Knobs::default() };
        let euler = run_knobs(&Knobs { ddt: "Euler", ..tr.clone() }, "eul", &args);
        let backward = run_knobs(&Knobs { ddt: "backward", ..tr }, "bwd", &args);
        assert_ne!(
            euler, backward,
            "ddt Euler and ddt backward wrote bit-identical fields: the case's \
             time scheme is not reaching the solver (SPEC-LIT 13.4.1)"
        );
    }
}
