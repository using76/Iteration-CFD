// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! `ofgpu-compressible` - the pressure-based all-speed compressible solver
//! of SPEC-LIT §122 to §127, driven by nothing but the case (CMP-15,
//! §125.8).
//!
//! ```text
//! ofgpu-compressible <case> [options]
//!
//!   <case>            an OpenFOAM case directory, or a .jsonc/.json case file
//!   -iters N          the number of steps (transient) or SIMPLE iterations
//!                     (steady); default endTime/deltaT of the case
//!   -reportEvery N    print the last outer iteration's line every N steps
//!                     (default 10; always on the first and the last step)
//!   -output LIST      comma list of foam,vtu,nvdb,vdb,usda (default: foam)
//!   -writeInterval W  write every W seconds of physical time - transient
//!                     runs only (SPEC-LIT §44.4)
//!   -writeEvery N     write every N iterations - steady runs only
//!   -noWrite          write nothing, not even the final state
//!   -permissive       SPEC-LIT 13.4's escape hatch
//! ```
//!
//! The run mode is the case's: `ddtSchemes default` (`numerics.ddt`)
//! `steadyState` or `localEuler` is a steady SIMPLE run (§125.3, §125.7),
//! `Euler` or `backward` a transient PIMPLE run at the case's `deltaT`.
//! Every step is [`ofgpu::compressible::Compressible::step`]; this file adds
//! nothing to the numerics. It reads the case, refuses by name what v1 does
//! not have (§122.4), and writes `p` (gauge, about `pOperating`), `U`, `T`,
//! `rho` and `Mach` through the one output pipeline of SPEC-LIT §44.
//!
//! Written from:
//!   ofgpu SPEC-LIT.md §122.4 - v1's scope, and that everything else is
//!     refused by name here
//!   ofgpu SPEC-LIT.md §122.2, §122.6 - the gauge pressure and the two
//!     routes a case states its gas by
//!   ofgpu SPEC-LIT.md §125.3, §125.6, §125.7 - the outer loop this runs
//!   ofgpu SPEC-LIT.md §126, §126.7 - which boundary conditions carry no
//!     compressible algebra; §126.8-§126.10 - the ones CMP-16 to CMP-18
//!     rewrite each outer iteration; §126.11 - BC-20's free-stream family
//!     and the `rho` file it accepts
//!   ofgpu SPEC-LIT.md §13.4, §13.4.1, §44 - refusals, the standing pair
//!     test, the output block
//!
//! Provenance: ORIGINAL - the driver is this tree's own wiring of the
//! module's solver to the case readers and writers; it carries no numerics.
//! No GPL-licensed source was consulted.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use ofgpu::compressible::pressure::Retarded;
use ofgpu::compressible::thermo::{PerfectGas, Viscosity};
use ofgpu::compressible::{
    Compressible, CompressibleControls, CompressibleState, OuterReport,
    ResidualControl as CmpResidualControl,
};
use ofgpu::field_setup::{
    harvest_scalar_field, harvest_vector_field, setup_scalar_field, setup_vector_field,
};
use ofgpu::fv::DivScheme;
use ofgpu::io::case::{find_start_time, format_time_name, CaseControls};
use ofgpu::io::case_json::LoweredCase;
use ofgpu::io::fields::{
    read_scalar_field, read_vector_field, PatchFieldSpec, RawScalarField, RawVectorField,
};
use ofgpu::io::voxelmap::LatticeSpec;
use ofgpu::io::{DivEntry, FoamDict};
use ofgpu::models::registry::{select_turbulence_model, RasModel};
use ofgpu::momentum::BuoyancyForm;
use ofgpu::timescheme::DdtScheme;
use ofgpu::{Error, Gpu, GpuMesh, GpuScalarField, GpuVectorField, HostMesh, Result, Scalar};

#[path = "common/mod.rs"]
mod common;

use common::{
    device_banner, g, load_case, next_arg, output_root, parse_output_formats, sci, CaseNumerics,
    OutputFormat,
};

/// The binary's name, in every refusal it makes.
const DRIVER: &str = "ofgpu-compressible";

/// A refusal in this driver's format: the setting first, then why.
fn refuse(setting: &str, why: &str) -> Error {
    Error::Refused(format!("{DRIVER}: {setting}: {why}"))
}

// ==========================================================================
//  Command line
// ==========================================================================

struct Options {
    case_path: PathBuf,
    /// `-iters N`; `None` takes the case's endTime/deltaT.
    iters: Option<u64>,
    report_every: u64,
    output: Vec<OutputFormat>,
    /// `-writeInterval W`, seconds; non-positive means not given.
    write_interval: f64,
    /// `-writeEvery N`, iterations.
    write_every: Option<u64>,
    do_write: bool,
    /// Which output flags the command line NAMED (SPEC-LIT §44.6).
    output_flags: Vec<&'static str>,
}

/// One literal with `{}` for the shared `-permissive` line only, so the
/// studio's registry sync test can read the flag set out of it.
fn usage() {
    eprintln!(
        "usage: ofgpu-compressible <case> [-iters N] [-reportEvery N] [-output LIST]\n       \
         [-writeInterval W] [-writeEvery N] [-noWrite] [-permissive]\n{}",
        ofgpu::io::contract::PERMISSIVE_USAGE
    );
}

/// A positive count from the command line.
fn parse_count(flag: &str, s: &str) -> Result<u64> {
    match s.trim().parse::<u64>() {
        Ok(n) if n > 0 => Ok(n),
        _ => Err(Error::Config(format!("{flag} needs a positive whole number, got \"{s}\""))),
    }
}

/// A finite number from the command line.
fn parse_number(flag: &str, s: &str) -> Result<f64> {
    match s.trim().parse::<f64>() {
        Ok(v) if v.is_finite() => Ok(v),
        _ => Err(Error::Config(format!("{flag} needs a finite number, got \"{s}\""))),
    }
}

fn parse(args: &[String]) -> Result<Options> {
    if args.len() < 2 {
        usage();
        return Err(Error::Config("no case given".to_string()));
    }
    let mut o = Options {
        case_path: PathBuf::from(&args[1]),
        iters: None,
        report_every: 10,
        output: vec![OutputFormat::Foam],
        write_interval: -1.0,
        write_every: None,
        do_write: true,
        output_flags: Vec::new(),
    };
    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "-iters" => o.iters = Some(parse_count("-iters", &next_arg(args, &mut i)?)?),
            "-reportEvery" => {
                o.report_every = parse_count("-reportEvery", &next_arg(args, &mut i)?)?
            }
            "-output" => {
                o.output = parse_output_formats(&next_arg(args, &mut i)?)?;
                o.output_flags.push("-output");
            }
            "-writeInterval" => {
                o.write_interval = parse_number("-writeInterval", &next_arg(args, &mut i)?)?;
                o.output_flags.push("-writeInterval");
            }
            "-writeEvery" => {
                o.write_every = Some(parse_count("-writeEvery", &next_arg(args, &mut i)?)?);
                o.output_flags.push("-writeEvery");
            }
            "-noWrite" => o.do_write = false,
            "-permissive" => ofgpu::io::contract::set_permissive(true),
            // SPEC-LIT §122.4: v1 is single-GPU (docs/17 COMP decision 14).
            flag @ ("-parallel" | "-np") => {
                return Err(refuse(
                    flag,
                    "decomposition is refused: ofgpu-compressible runs one GPU \
                     (SPEC-LIT §122.4). Run the undecomposed case",
                ))
            }
            other => {
                usage();
                return Err(Error::Config(format!("unknown option {other}")));
            }
        }
        i += 1;
    }
    Ok(o)
}

// ==========================================================================
//  What v1 does not have, refused by name (SPEC-LIT §122.4)
// ==========================================================================

/// Decomposition, motion, sources, turbulence other than laminar, a
/// Boussinesq body force and a non-Newtonian rheology - each refused by its
/// own setting before a field is set up.
fn refuse_out_of_scope(case_path: &Path, cc: &CaseControls, lowered: Option<&LoweredCase>) -> Result<()> {
    // ---- decomposition ---------------------------------------------------
    for var in ["PMI_RANK", "PMI_SIZE"] {
        if std::env::var_os(var).is_some() {
            return Err(refuse(
                var,
                "an mpiexec environment: decomposition is refused, ofgpu-compressible \
                 runs one GPU (SPEC-LIT §122.4)",
            ));
        }
    }
    if lowered.is_none() {
        if let Ok(rd) = std::fs::read_dir(case_path) {
            for e in rd.flatten() {
                let name = e.file_name().to_string_lossy().to_string();
                if name.starts_with("processor") && e.path().is_dir() {
                    return Err(refuse(
                        &name,
                        "a decomposed case: decomposition is refused, ofgpu-compressible \
                         runs one GPU on the undecomposed case (SPEC-LIT §122.4)",
                    ));
                }
            }
        }
    }

    // ---- motion ------------------------------------------------------------
    match lowered {
        Some(l) => {
            if l.motion.is_some() {
                return Err(refuse(
                    "motion",
                    "the case's motion block: mesh motion is refused, v1 runs a fixed mesh \
                     (SPEC-LIT §122.4); ofgpu-lowmach runs the motion block",
                ));
            }
        }
        None => {
            let p = case_path.join("constant").join("dynamicMeshDict");
            if p.exists() {
                let d = FoamDict::read(&p)?;
                let kind = d.get("dynamicFvMesh").unwrap_or("").trim().to_string();
                if kind != "staticFvMesh" {
                    return Err(refuse(
                        "constant/dynamicMeshDict",
                        &format!(
                            "dynamicFvMesh {}: mesh motion is refused, v1 runs a fixed mesh \
                             (SPEC-LIT §122.4); only staticFvMesh is accepted",
                            if kind.is_empty() { "(not stated)" } else { kind.as_str() }
                        ),
                    ));
                }
            }
        }
    }

    // ---- sources -----------------------------------------------------------
    match lowered {
        Some(l) => {
            if !l.sources.is_empty() {
                return Err(refuse(
                    "sources",
                    &format!(
                        "{} source(s): v1 has no sources (SPEC-LIT §122.4)",
                        l.sources.len()
                    ),
                ));
            }
        }
        None => {
            for rel in [
                "constant/fvOptions",
                "system/fvOptions",
                "constant/fvModels",
                "system/fvModels",
                "constant/fvConstraints",
                "system/fvConstraints",
            ] {
                if case_path.join(rel).exists() {
                    return Err(refuse(
                        rel,
                        "sources: v1 has no sources (SPEC-LIT §122.4); remove the file",
                    ));
                }
            }
        }
    }

    // ---- turbulence ----------------------------------------------------------
    //
    // A JSONC case with no `physics.turbulence` states no model, and that is
    // laminar; every other statement goes through the one selector both
    // formats share.
    let named = match lowered {
        Some(l) if l.turbulence_model.is_none() && l.turbulence_kind.is_none() => None,
        _ => Some(select_turbulence_model(cc)?),
    };
    if let Some(sel) = named {
        let is_laminar = sel.model == RasModel::Laminar && sel.les.is_none() && sel.des.is_none();
        if !is_laminar {
            let setting = if lowered.is_some() {
                "physics.turbulence"
            } else {
                "constant/momentumTransport"
            };
            let name = sel.describe();
            let why = if sel.model == RasModel::SpalartAllmaras && sel.les.is_none() && sel.des.is_none() {
                format!(
                    "{name}: Spalart-Allmaras runs in ofgpu-compressible in its conservative \
                     form, which CMP-25 builds (SPEC-LIT §127.1); until then this driver is \
                     laminar only and running SA as laminar would be a substitution \
                     (SPEC-LIT §13.4). Alternative: laminar"
                )
            } else {
                format!(
                    "{name}: every turbulence model but Spalart-Allmaras is refused \
                     (SPEC-LIT §127.1, docs/17 COMP decision 9). Alternative: laminar"
                )
            };
            return Err(refuse(setting, &why));
        }
    }

    // ---- the body force and the rheology ----------------------------------
    if let BuoyancyForm::Boussinesq { beta } = cc.buoyancy.form {
        return Err(refuse(
            "physics.buoyancy",
            &format!(
                "boussinesq (beta {}): this solver carries the full density rho = psi \
                 (p + p_op) in every term (SPEC-LIT §122.1, §123.1), so a linearised body \
                 force would be a second density law; gravity acts as rho g",
                g(f64::from(beta))
            ),
        ));
    }
    if !cc.rheology.is_newtonian() {
        return Err(refuse(
            "rheology",
            "a non-Newtonian model: the viscosity is the gas's, constant or Sutherland \
             (SPEC-LIT §122.1)",
        ));
    }

    // SPEC-LIT 13.4: run.adjustTimeStep and run.maxCo.
    common::refuse_unimplemented_blocks(lowered)
}

/// The boundary types that carry no compressible algebra (SPEC-LIT §126.7)
/// and need no per-iteration rewrite: the §4 triple as set up is the whole
/// condition.
const STATIC_TYPES: &[&str] = &[
    "fixedValue",
    "zeroGradient",
    "fixedGradient",
    "mixed",
    "noSlip",
    "slip",
    "symmetry",
    "symmetryPlane",
    "empty",
];

/// SPEC-LIT §126.8's inlets, §126.9's outlet, §126.10's far-field and
/// §126.11's free-stream family, for the refusal's list of what is
/// accepted. `compressible::bc::lower` turns each into a placeholder and a
/// rule before this check sees the fields.
const CMP16_KINDS: &str =
    "totalPressure (with gamma), totalTemperature, flowRateInletVelocity (with massFlowRate), \
     waveTransmissive (on p or T), characteristicFarfield or farfield (on U, p and T), \
     inletOutletTotalTemperature (on T), supersonicFreestream (on U), freestreamPressure (on p)";

/// The case-file name of a §126.8 inlet, for the run's report.
fn rule_name(k: ofgpu::compressible::bc::RuleKind) -> &'static str {
    use ofgpu::compressible::bc::RuleKind;
    match k {
        RuleKind::TotalPressure => "p totalPressure",
        RuleKind::TotalTemperature => "T totalTemperature",
        RuleKind::MassFlowInlet => "U flowRateInletVelocity (massFlowRate)",
    }
}

/// Every patch of `field` must be one of [`STATIC_TYPES`]: a condition the
/// incompressible loop rewrites each iteration would stand still here.
fn check_static_bcs(field: &str, boundary: &BTreeMap<String, PatchFieldSpec>) -> Result<()> {
    for (patch, spec) in boundary {
        let t = spec.type_name.as_str();
        if STATIC_TYPES.contains(&t) {
            continue;
        }
        let why = if t == "advective" {
            format!(
                "advective is the generic outlet of SPEC-LIT §120.4, whose wave speed is the \
                 flux; a gas outlet is waveTransmissive, which adds the speed of sound on p \
                 (SPEC-LIT §126.9). Accepted: {}, {CMP16_KINDS}",
                STATIC_TYPES.join(", ")
            )
        } else {
            format!(
                "{t} is not a static condition, and ofgpu-compressible rewrites no boundary \
                 between iterations but the compressible conditions of SPEC-LIT §126.8-§126.10 \
                 (§126, §126.7). Accepted: {}, {CMP16_KINDS}",
                STATIC_TYPES.join(", ")
            )
        };
        return Err(refuse(&format!("{field} patch {patch}"), &why));
    }
    Ok(())
}

// ==========================================================================
//  The case's settings, read into the solver's controls
// ==========================================================================

/// Everything [`run`] hands the solver, and the run's length.
#[derive(Debug, Clone)]
struct DriverControls {
    ctrl: CompressibleControls,
    transient: bool,
    n_steps: u64,
}

/// A convection entry with the `bounded` prefix is refused by name: the
/// equations here are in conservative form on the mass flux, and §3.1's
/// correction would remove `V_P (div mdot)_P`, which is `-ddt(rho)` in a
/// compressible flow, not an error (SPEC-LIT §123.1, §125.1).
fn unbounded(key: &str, e: DivEntry, lowered: bool) -> Result<DivScheme> {
    if e.bounded {
        let setting = if lowered { format!("numerics.div.{key}") } else { format!("divSchemes/{key}") };
        return ofgpu::io::contract::unsupported_note(
            &setting,
            "bounded",
            &["Gauss <scheme> (no bounded prefix)"],
            "the compressible equations are conservative on the mass flux, and div(mdot) \
             is -ddt(rho), not a convergence error (SPEC-LIT §123.1, §125.1)",
            "the scheme without bounded",
            e.scheme,
        );
    }
    Ok(e.scheme)
}

/// The retarded face values of (124.3): `div(phid,p)` upwind (the default
/// when the case names none) or a §7 limiter.
fn retarded(num: &CaseNumerics, lowered: bool) -> Result<Retarded> {
    let Some(e) = num.div_named("div(phid,p)")? else { return Ok(Retarded::Upwind) };
    let setting = if lowered { "numerics.div.div(phid,p)" } else { "divSchemes/div(phid,p)" };
    match (e.bounded, e.scheme) {
        (false, DivScheme::Upwind) => Ok(Retarded::Upwind),
        (false, DivScheme::Limited(l)) => Ok(Retarded::Limited(l)),
        _ => Err(refuse(
            setting,
            &format!(
                "{:?}: the retarded density of (124.3) is the upwind cell's or a §7-limited \
                 face value (SPEC-LIT §124.1). Alternative: Gauss upwind, or a limited scheme \
                 such as Gauss vanLeer",
                e.scheme
            ),
        )),
    }
}

/// SPEC-LIT §13.4.1: one function, called by [`run`] and by the pair test,
/// that reads every setting the solver honours from either case format.
fn driver_controls(
    case_path: &Path,
    cc: &CaseControls,
    lowered: Option<&LoweredCase>,
    gas: PerfectGas,
    iters: Option<u64>,
) -> Result<DriverControls> {
    let num = CaseNumerics::read(case_path, cc, lowered)?;
    let json = lowered.is_some();
    let d = CompressibleControls::default();

    let ddt = cc.turb.ddt;
    let transient = !ddt.is_steady();
    let delta_t: Scalar = if transient { cc.turb.delta_t } else { 1.0 };
    let n_steps = match iters {
        Some(n) => n,
        None => {
            let n = cc.turb.n_outer_iterations;
            if n < 1 {
                return Err(refuse(
                    "endTime/deltaT",
                    &format!("{n} steps: the run takes at least one step. Alternative: -iters N"),
                ));
            }
            n as u64
        }
    };
    let g_vec = cc.buoyancy.g;

    let mut ctrl = d;
    ctrl.gas = gas;

    ctrl.momentum.u_solver = num.solver("U", d.momentum.u_solver)?;
    ctrl.momentum.u_relax = num.relax("U", d.momentum.u_relax)?;
    ctrl.momentum.div_scheme = unbounded("div(phi,U)", num.div("div(phi,U)")?, json)?;
    ctrl.momentum.grad_scheme = num.grad("grad(U)")?;
    ctrl.momentum.sn_grad = num.sn_grad("default")?;
    ctrl.momentum.ddt = ddt;
    ctrl.momentum.delta_t = delta_t;
    ctrl.momentum.g = g_vec;

    ctrl.pressure.p_solver = num.solver("p", d.pressure.p_solver)?;
    ctrl.pressure.retarded = retarded(&num, json)?;
    ctrl.pressure.sn_grad = num.sn_grad("default")?;
    ctrl.pressure.n_non_orth_corr = num.n_non_orth_correctors();

    ctrl.energy.t_solver = num.solver("T", d.energy.t_solver)?;
    ctrl.energy.t_relax = num.relax("T", d.energy.t_relax)?;
    ctrl.energy.div_scheme = unbounded("div(phi,T)", num.div("div(phi,T)")?, json)?;
    ctrl.energy.grad_scheme = num.grad("grad(T)")?;
    ctrl.energy.sn_grad = num.sn_grad("default")?;
    ctrl.energy.ddt = ddt;
    ctrl.energy.delta_t = delta_t;
    ctrl.energy.g = g_vec;

    ctrl.n_outer = cc.algorithm.n_outer_correctors;
    ctrl.n_corr = cc.algorithm.n_correctors;
    ctrl.p_relax = num.relax_field("p", d.p_relax)?;
    ctrl.lts.cfl = cc.turb.lts.co_max;
    // The energy equation's bar may be spelled for its unknown (`T`) or for
    // the energy form a case names (`h`, `e`): one equation, one residual.
    ctrl.residual_control = if !transient && !cc.residual_control.is_empty() {
        let rc = &cc.residual_control;
        Some(CmpResidualControl {
            u: rc.tolerance("U").unwrap_or(0.0),
            p: rc.tolerance("p").unwrap_or(0.0),
            t: rc.tolerance("T").or_else(|| rc.tolerance("h")).or_else(|| rc.tolerance("e")).unwrap_or(0.0),
        })
    } else {
        None
    };
    ctrl.log = false;

    Ok(DriverControls { ctrl, transient, n_steps })
}

/// The controls banner: every setting the run took, in one place.
fn print_controls(dc: &DriverControls) {
    let c = &dc.ctrl;
    let gas = &c.gas;
    let visc = match gas.viscosity {
        Viscosity::Sutherland { a_s, t_s } => format!("sutherland As {} Ts {}", g(a_s), g(t_s)),
        Viscosity::Constant { mu } => format!("const mu {}", g(mu)),
    };
    println!(
        "gas (SPEC-LIT §122.1): gamma {} W {} kg/mol R_s {} c_p {} Pr {} Pr_t {} {} | p_op {} Pa \
         (p is the gauge pressure about it, SPEC-LIT §122.2)",
        g(gas.gamma),
        g(gas.w),
        g(gas.r_s()),
        g(gas.cp()),
        g(gas.pr),
        g(gas.pr_t),
        visc,
        g(gas.p_op)
    );
    println!(
        "run: {} {} {}, ddt {}{}",
        if dc.transient { "transient PIMPLE," } else { "steady SIMPLE," },
        dc.n_steps,
        if dc.transient { "steps" } else { "iterations" },
        c.momentum.ddt.describe(),
        if dc.transient { format!(", deltaT {}", g(f64::from(c.momentum.delta_t))) } else { String::new() }
    );
    println!(
        "  nOuterCorrectors {} nCorrectors {} nNonOrthogonalCorrectors {} | relax U {} T {} p {}",
        c.n_outer,
        c.n_corr,
        c.pressure.n_non_orth_corr,
        g(f64::from(c.momentum.u_relax)),
        g(f64::from(c.energy.t_relax)),
        g(f64::from(c.p_relax))
    );
    println!(
        "  div(phi,U) {:?} div(phi,T) {:?} retarded {:?} | g ({}, {}, {}) | viscousWork {}{}",
        c.momentum.div_scheme,
        c.energy.div_scheme,
        c.pressure.retarded,
        g(f64::from(c.momentum.g.x)),
        g(f64::from(c.momentum.g.y)),
        g(f64::from(c.momentum.g.z)),
        c.energy.viscous_work,
        if c.momentum.ddt == DdtScheme::LocalEuler {
            format!(" | localEuler CFL {}", g(f64::from(c.lts.cfl)))
        } else {
            String::new()
        }
    );
    if let Some(rc) = c.residual_control {
        println!(
            "  residualControl U {} p {} T {}",
            sci(f64::from(rc.u), 2),
            sci(f64::from(rc.p), 2),
            sci(f64::from(rc.t), 2)
        );
        // §125.6: the run stops only when EVERY initial residual is under
        // its bar, and a bar of 0 is never met.
        if rc.u == 0.0 || rc.p == 0.0 || rc.t == 0.0 {
            println!(
                "  NOTE: residualControl names no bar for U, p or T (T may be spelled h or e), \
                 so the run takes all {} iterations",
                dc.n_steps
            );
        }
    }
}

// ==========================================================================
//  Fields
// ==========================================================================

struct InitialFields {
    u: RawVectorField,
    p: RawScalarField,
    t: RawScalarField,
}

/// `U`, `p` (gauge, SPEC-LIT §122.2) and `T` from either case format, and
/// the gas (SPEC-LIT §122.6).
fn load_fields_and_gas(
    case_path: &Path,
    hm: &HostMesh,
    lowered: Option<&LoweredCase>,
) -> Result<(InitialFields, PerfectGas)> {
    match lowered {
        Some(l) => {
            let gas = l.compressible.ok_or_else(|| {
                refuse(
                    "physics.compressible",
                    "the case states no gas: ofgpu-compressible needs physics.compressible \
                     { gamma, W, viscosity, pOperating } (SPEC-LIT §122.6)",
                )
            })?;
            let t = l.t_field.as_ref().ok_or_else(|| {
                refuse("initial.T", "the gas law has no state without T (SPEC-LIT §122.6)")
            })?;
            let f = InitialFields {
                u: l.u_field.to_raw(hm.n_cells),
                p: l.p_field.to_raw(hm.n_cells),
                t: t.to_raw(hm.n_cells),
            };
            Ok((f, gas))
        }
        None => {
            if !case_path.join("constant").join("thermophysicalProperties").exists() {
                return Err(refuse(
                    "constant/thermophysicalProperties",
                    "the case states no gas: ofgpu-compressible reads the perfect-gas subset \
                     of SPEC-LIT §122.6 from it",
                ));
            }
            let gas = ofgpu::io::thermo_dict::read_thermophysical_properties(case_path)?;
            let dir = case_path.join(find_start_time(case_path)?);
            for name in ["U", "p", "T"] {
                if !dir.join(name).exists() {
                    return Err(Error::Config(format!(
                        "{} has no {name} field; {DRIVER} solves U, p and T",
                        dir.display()
                    )));
                }
            }
            // SPEC-LIT §126.11: a `rho` file is read only to check that its
            // patches state the gas law the solver forms on every face.
            if dir.join("rho").exists() {
                let rho = read_scalar_field(&dir.join("rho"), hm.n_cells)?;
                ofgpu::compressible::bc::check_rho(&rho).map_err(|e| refuse("boundaryField", &e.to_string()))?;
            }
            let f = InitialFields {
                u: read_vector_field(&dir.join("U"), hm.n_cells)?,
                p: read_scalar_field(&dir.join("p"), hm.n_cells)?,
                t: read_scalar_field(&dir.join("T"), hm.n_cells)?,
            };
            Ok((f, gas))
        }
    }
}

/// The patch types of a field the solver forms rather than solves (`rho`,
/// `Mach`): a constraint type stays, everything else is `calculated`.
fn calculated_like(t: &RawScalarField, name: &str) -> RawScalarField {
    let mut out = t.types_only();
    out.name = name.to_string();
    for spec in out.boundary.values_mut() {
        let keep = matches!(
            spec.type_name.as_str(),
            "empty" | "symmetry" | "symmetryPlane" | "wedge" | "cyclic" | "cyclicAMI"
        );
        if !keep {
            spec.type_name = "calculated".to_string();
        }
    }
    out
}

/// What the run writes, by name and in order (SPEC-LIT §44.2's early
/// check reads the same list).
const OUTPUT_FIELDS: [&str; 5] = ["U", "p", "T", "rho", "Mach"];

/// Write the five fields at time label `t` (seconds, or the iteration count
/// of a steady run), through the one pipeline.
#[allow(clippy::too_many_arguments)]
fn write_time(
    gpu: &Gpu,
    s: &CompressibleState,
    gas: &PerfectGas,
    hm: &HostMesh,
    fields: &InitialFields,
    t: f64,
    pipeline: &mut ofgpu::io::OutputPipeline,
    force: bool,
    volume: Option<&ofgpu::io::VolumeTarget>,
) -> Result<bool> {
    if !force && !pipeline.any_due(t) {
        return Ok(false);
    }
    let name = format_time_name(t as Scalar);

    let mut out_u = fields.u.types_only();
    let mut out_p = fields.p.types_only();
    let mut out_t = fields.t.types_only();
    harvest_vector_field(gpu, &mut out_u, &s.u, hm)?;
    harvest_scalar_field(gpu, &mut out_p, &s.p, hm)?;
    harvest_scalar_field(gpu, &mut out_t, &s.t, hm)?;
    out_u.dimensions = "[0 1 -1 0 0 0 0]".to_string();
    out_p.dimensions = "[1 -1 -2 0 0 0 0]".to_string();
    out_t.dimensions = "[0 0 0 1 0 0 0]".to_string();

    let mut out_rho = calculated_like(&fields.t, "rho");
    harvest_scalar_field(gpu, &mut out_rho, &s.rho, hm)?;
    out_rho.dimensions = "[1 -3 0 0 0 0 0]".to_string();

    // (122.12) per cell and per boundary face, from the solver's u and T:
    // the patch entries are T's (expanded per patch by the harvest), and
    // each patch's values are overwritten from its own slice of the
    // boundary arrays, as the harvest itself slices them.
    let k = gas.constants();
    let mut out_m = calculated_like(&fields.t, "Mach");
    harvest_scalar_field(gpu, &mut out_m, &s.t, hm)?;
    out_m.dimensions = "[0 0 0 0 0 0 0]".to_string();
    out_m.internal = out_u
        .internal
        .iter()
        .zip(&out_t.internal)
        .map(|(u, tt)| k.mach(*u, *tt))
        .collect();
    let (ub, tb) = (gpu.download(&s.u.bf)?, gpu.download(&s.t.bf)?);
    let face_mach: Vec<Scalar> = ub.iter().zip(&tb).map(|(u, tt)| k.mach(*u, *tt)).collect();
    for patch in &hm.patches {
        if let Some(spec) = out_m.boundary.get_mut(&patch.name) {
            spec.value = face_mach[patch.start..patch.start + patch.size].to_vec();
        }
    }

    let foam = vec![
        ofgpu::io::FoamField::vector("U", &out_u),
        ofgpu::io::FoamField::scalar("p", &out_p),
        ofgpu::io::FoamField::scalar("T", &out_t),
        ofgpu::io::FoamField::scalar("rho", &out_rho),
        ofgpu::io::FoamField::scalar("Mach", &out_m),
    ];
    let vis = vec![
        ofgpu::io::OutputField::vector("U", &out_u.internal),
        ofgpu::io::OutputField::scalar("p", &out_p.internal),
        ofgpu::io::OutputField::scalar("T", &out_t.internal),
        ofgpu::io::OutputField::scalar("rho", &out_rho.internal),
        ofgpu::io::OutputField::scalar("Mach", &out_m.internal),
    ];
    let cart = ofgpu::pressure::cartesian::detect(hm).ok().map(|c| ofgpu::io::cartesian_info(hm, &c));
    let ctx = ofgpu::io::WriteCtx {
        time: t as Scalar,
        step: 0,
        name: &name,
        mesh: hm,
        cart: cart.as_ref(),
        vox: volume.and_then(|v| v.vox()),
        fields: &vis,
        foam: &foam,
    };
    pipeline.write(&ctx, t, force)
}

// ==========================================================================
//  The run
// ==========================================================================

/// What a finished run reports, for `main` and the tests.
#[derive(Debug, Clone)]
struct RunEnd {
    steps: u64,
    converged: bool,
    last: Option<OuterReport>,
    seconds: f64,
}

fn run(o: &Options) -> Result<RunEnd> {
    let t0 = Instant::now();
    let gpu = Gpu::new(0)?;
    println!("{}", device_banner(&gpu, "compressible")?);

    let (hm, cc, lowered) = load_case(&o.case_path)?;
    refuse_out_of_scope(&o.case_path, &cc, lowered.as_ref())?;
    hm.print_report();

    let (fields, gas) = load_fields_and_gas(&o.case_path, &hm, lowered.as_ref())?;
    // SPEC-LIT §126.8: the compressible inlets become placeholders and
    // rules; `fields` keeps the types as read, for the written fields.
    let inlets = ofgpu::compressible::bc::lower(&fields.u, &fields.p, &fields.t, &hm, &gas)
        .map_err(|e| refuse("boundaryField", &e.to_string()))?;
    check_static_bcs("U", &inlets.u.boundary)?;
    check_static_bcs("p", &inlets.p.boundary)?;
    check_static_bcs("T", &inlets.t.boundary)?;
    for r in &inlets.rules {
        println!("inlet: {} on patch {} ({} faces), SPEC-LIT §126.8", rule_name(r.kind), r.patch, r.size);
    }
    for r in &inlets.freestreams {
        use ofgpu::compressible::bc::FreestreamKind;
        match r.kind {
            FreestreamKind::SupersonicFreestream { u, p, t } => println!(
                "free stream: U supersonicFreestream on patch {} ({} faces), UInf ({}, {}, {}) pInf {} TInf {}, SPEC-LIT §126.11",
                r.patch, r.size, g(u[0]), g(u[1]), g(u[2]), g(p), g(t)
            ),
            FreestreamKind::FreestreamPressure { value, supersonic } => println!(
                "free stream: p freestreamPressure on patch {} ({} faces), freestreamValue {} supersonic {supersonic}, SPEC-LIT §126.11",
                r.patch, r.size, g(value)
            ),
        }
    }
    for r in &inlets.farfields {
        println!(
            "far-field: characteristicFarfield on patch {} ({} faces), free stream U ({}, {}, {}) p {} T {}, SPEC-LIT §126.10",
            r.patch, r.size, g(r.u[0]), g(r.u[1]), g(r.u[2]), g(r.p), g(r.t)
        );
    }
    for r in &inlets.outlets {
        use ofgpu::compressible::bc::{OutletField, OutletKind};
        let field = if r.field == OutletField::P { "p" } else { "T" };
        match r.kind {
            OutletKind::SupersonicSwitch => println!(
                "outlet: p fixedValue on patch {} ({} faces) switches to extrapolation where u_n/c >= 1, SPEC-LIT §126.9",
                r.patch, r.size
            ),
            OutletKind::WaveTransmissive if r.l_inf > 0.0 => println!(
                "outlet: {field} waveTransmissive on patch {} ({} faces), lInf {} sigma {} fieldInf {}, SPEC-LIT §126.9",
                r.patch, r.size, r.l_inf, r.sigma, r.field_inf
            ),
            OutletKind::WaveTransmissive => println!(
                "outlet: {field} waveTransmissive on patch {} ({} faces), no lInf (K = 0), SPEC-LIT §126.9",
                r.patch, r.size
            ),
        }
    }

    let dc = driver_controls(&o.case_path, &cc, lowered.as_ref(), gas, o.iters)?;
    print_controls(&dc);

    // ---- the output pipeline (SPEC-LIT §44), resolved before any kernel --
    let mut output_plan = common::output_plan(lowered.as_ref())?;
    if let Some(plan) = &mut output_plan {
        if !common::refuse_output_named_twice(plan, &o.output_flags)? {
            output_plan = None;
        }
    }
    if dc.transient && o.write_every.is_some() {
        return Err(refuse(
            "-writeEvery",
            "a transient run schedules its writes in seconds: use -writeInterval (SPEC-LIT §44.4)",
        ));
    }
    if !dc.transient && o.write_interval > 0.0 {
        return Err(refuse(
            "-writeInterval",
            "a steady run has no clock: use -writeEvery N iterations (SPEC-LIT §44.4)",
        ));
    }
    let wants_volume = ofgpu::io::has_volume_format(&o.output)
        || output_plan
            .as_ref()
            .and_then(|p| p.vis.as_ref())
            .is_some_and(|v| ofgpu::io::has_volume_format(&v.formats));
    let raw_mesh = if wants_volume && ofgpu::pressure::cartesian::detect(&hm).is_err() {
        Some(common::case_raw_mesh(&o.case_path, lowered.as_ref())?)
    } else {
        None
    };
    if let Some(plan) = &mut output_plan {
        if !dc.transient {
            plan.refuse_interval_when_steady(
                DRIVER,
                "a steady run writes its final state once; give the case a transient ddt \
                 for a time series",
            )?;
        }
        plan.refuse_restart(DRIVER, "The final state is written as fields; restart from them.")?;
        let no_lattice = match (plan.vis.as_ref(), raw_mesh.as_ref()) {
            (Some(v), Some(raw)) if ofgpu::io::has_volume_format(&v.formats) => v.lattice_refusal(raw, &hm),
            _ => None,
        };
        plan.refuse_visualisation_without_a_lattice(no_lattice.as_deref())?;
        if plan.is_empty() {
            output_plan = None;
        }
    }
    let cli_output: Vec<OutputFormat> = match &output_plan {
        Some(_) => o.output.clone(),
        None => {
            let no_lattice = match raw_mesh.as_ref() {
                Some(raw) if ofgpu::io::has_volume_format(&o.output) => {
                    ofgpu::io::voxelmap::lattice_shape(raw, &hm, &LatticeSpec::default())
                        .err()
                        .map(|e| e.to_string())
                }
                _ => None,
            };
            ofgpu::io::output_plan::drop_volume_formats_without_a_lattice(&o.output, no_lattice.as_deref())?
        }
    };
    let (volume_formats, volume_spec) = match output_plan.as_ref().and_then(|p| p.vis.as_ref()) {
        Some(v) => (v.formats.clone(), v.lattice.clone()),
        None if output_plan.is_none() => (cli_output.clone(), LatticeSpec::default()),
        None => (Vec::new(), LatticeSpec::default()),
    };
    let volume: Option<ofgpu::io::VolumeTarget> =
        if o.do_write && ofgpu::io::has_volume_format(&volume_formats) {
            let cart = ofgpu::pressure::cartesian::detect(&hm).ok().map(|c| ofgpu::io::cartesian_info(&hm, &c));
            let t = ofgpu::io::volume_target(raw_mesh.as_ref(), &hm, cart, &volume_spec)?;
            for line in t.report_lines() {
                println!("{line}");
            }
            Some(t)
        } else {
            None
        };
    drop(raw_mesh);
    let out_root = output_root(&o.case_path);
    let mut pipeline = match &output_plan {
        Some(plan) => {
            plan.check_fields(&OUTPUT_FIELDS)?;
            ofgpu::io::OutputPipeline::from_plan(plan, &out_root, "compressible", "restart")?
        }
        None => ofgpu::io::OutputPipeline::from_command_line(
            &out_root,
            "compressible",
            &cli_output,
            if dc.transient { o.write_interval } else { o.write_every.map_or(0.0, |n| n as f64) },
        )?,
    };
    if !dc.transient {
        pipeline.set_clock_unit("iterations");
    }
    if o.do_write {
        println!("{}", pipeline.describe());
    } else {
        println!("output: nothing (-noWrite)");
    }

    // ---- the solver --------------------------------------------------------
    let mesh = GpuMesh::upload(&gpu, &hm)?;
    let mut u = GpuVectorField::zeros(&gpu, &mesh, "U")?;
    let mut p = GpuScalarField::zeros(&gpu, &mesh, "p")?;
    let mut t = GpuScalarField::zeros(&gpu, &mesh, "T")?;
    setup_vector_field(&gpu, &mut u, &inlets.u, &hm)?;
    setup_scalar_field(&gpu, &mut p, &inlets.p, &hm)?;
    setup_scalar_field(&gpu, &mut t, &inlets.t, &hm)?;
    let mut s = CompressibleState::new(&gpu, &mesh, u, p, t)?;
    let mut solver = Compressible::new(&gpu, &mesh, dc.ctrl)?;
    solver.set_boundary_rules(&gpu, inlets.rules)?;
    solver.set_outlet_rules(&gpu, inlets.outlets)?;
    solver.set_farfield_rules(&gpu, inlets.farfields)?;
    solver.set_freestream_rules(&gpu, inlets.freestreams)?;
    solver.initialise(&gpu, &mut s)?;

    let dt = f64::from(dc.ctrl.momentum.delta_t);
    let mut end = RunEnd { steps: 0, converged: false, last: None, seconds: 0.0 };
    for step in 1..=dc.n_steps {
        let r = solver.step(&gpu, &mut s)?;
        end.steps = step;
        end.converged = r.converged;
        end.last = r.outers.last().cloned();
        let last_step = step == dc.n_steps || r.converged;
        if step == 1 || last_step || step % o.report_every == 0 {
            if let Some(l) = &end.last {
                println!("{}", l.line());
            }
        }
        let clock = if dc.transient { step as f64 * dt } else { step as f64 };
        if o.do_write && !last_step && write_time(&gpu, &s, &gas, &hm, &fields, clock, &mut pipeline, false, volume.as_ref())? {
            println!("    written at {}", g(clock));
        }
        if last_step {
            if o.do_write {
                write_time(&gpu, &s, &gas, &hm, &fields, clock, &mut pipeline, true, volume.as_ref())?;
                println!("    final state written at {} to {}", g(clock), out_root.display());
            }
            break;
        }
    }
    end.seconds = t0.elapsed().as_secs_f64();
    println!(
        "{DRIVER}: {} {}{} in {:.2} s",
        end.steps,
        if dc.transient { "steps" } else { "iterations" },
        if end.converged { ", residualControl met" } else { "" },
        end.seconds
    );
    Ok(end)
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    let outcome = parse(&args).and_then(|o| run(&o));
    match outcome {
        Ok(_) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

// ==========================================================================
//  Tests - SPEC-LIT §125.8 (CMP-15)
// ==========================================================================

#[cfg(test)]
mod compressible {
    use super::*;
    use common::knobs::{apply, assert_none_inert, scratch_dir, written_state, Knob, NO_PRE};

    /// The 50-cell tube of SPEC-LIT §125.8: 1 m long, one cell across,
    /// air at 300 K moving at 50 m/s, a 320 K inlet, a fixed-`p` outlet and
    /// symmetry sides; transient Euler, three steps of 1e-5 s. Every entry a
    /// knob below turns is spelled once, so its `from` text is unique.
    const TUBE: &str = r#"{
  "name": "tube",
  "mesh": {
    "kind": "cartesian",
    "bounds": { "min": [0, 0, 0], "max": [1, 0.02, 0.02] },
    "cells": [50, 1, 1],
    "boundaries": { "xmin": "inlet", "xmax": "outlet", "ymin": "ya", "ymax": "yb", "zmin": "za", "zmax": "zb" }
  },
  "physics": {
    "gravity": [0, 0, 0],
    "fluid": { "nu": 1e-5, "Pr": 0.71, "Prt": 0.85, "TRef": 300 },
    "buoyancy": "densityRatio",
    "compressible": {
      "gamma": 1.4, "W": 0.0289647, "Pr": 0.71,
      "viscosity": { "model": "sutherland", "As": 1.458e-6, "Ts": 110.4 },
      "pOperating": 1e5
    }
  },
  "patches": [
    { "match": "inlet", "kind": "inlet", "U": { "type": "fixedValue", "value": [50, 0, 0] }, "T": { "type": "fixedValue", "value": 320 } },
    { "match": "outlet", "kind": "open", "U": { "type": "zeroGradient" }, "p": { "type": "fixedValue", "value": 0 }, "T": { "type": "zeroGradient" } },
    { "match": "(ya|yb|za|zb)", "kind": "symmetry" }
  ],
  "initial": { "U": [50, 0, 0], "p": 0.0, "T": 300.0 },
  "numerics": {
    "algorithm": { "kind": "PIMPLE", "correctors": 2, "outerCorrectors": 1 },
    "ddt": "Euler",
    "div": { "default": "Gauss upwind" },
    "grad": "Gauss linear",
    "laplacian": { "snGrad": "corrected", "nonOrthogonalCorrectors": 0 },
    "relaxation": { "U": 0.7, "T": 0.9, "p": 0.3 },
    "solvers": [
      { "match": "p", "solver": "PBiCGStab", "preconditioner": "DILU", "tolerance": 1e-12 },
      { "match": "U", "solver": "PBiCGStab", "preconditioner": "DILU", "tolerance": 1e-12 },
      { "match": "T", "solver": "PBiCGStab", "preconditioner": "DILU", "tolerance": 1e-12 }
    ]
  },
  "run": { "endTime": 0.00003, "deltaT": 0.00001 }
}
"#;

    /// The tube's `physics.compressible` block with the comma before it.
    const GAS_BLOCK: &str = ",\n    \"compressible\": {\n      \"gamma\": 1.4, \"W\": 0.0289647, \"Pr\": 0.71,\n      \"viscosity\": { \"model\": \"sutherland\", \"As\": 1.458e-6, \"Ts\": 110.4 },\n      \"pOperating\": 1e5\n    }";

    fn argv(v: &[&str]) -> Vec<String> {
        std::iter::once(DRIVER).chain(v.iter().copied()).map(String::from).collect()
    }

    /// A fresh tube case in its own scratch directory: `(dir, case file)`.
    fn tube(tag: &str) -> (PathBuf, PathBuf) {
        let dir = scratch_dir(tag);
        let case = dir.join("case.jsonc");
        std::fs::write(&case, TUBE).expect("write the tube case");
        (dir, case)
    }

    /// One edit of a case file that must find its text.
    fn edit(path: &Path, from: &str, to: &str) {
        let s = std::fs::read_to_string(path).expect("read the case");
        assert!(s.contains(from), "{from:?} is not in {}", path.display());
        std::fs::write(path, s.replacen(from, to, 1)).expect("write the case");
    }

    /// Run the tube with `k` applied on `side`, and return what it wrote.
    fn run_knob(k: &Knob, side: bool, tag: &str) -> Vec<(String, String)> {
        let (dir, case) = tube(tag);
        apply(&dir, k, side);
        let o = parse(&argv(&[case.to_string_lossy().as_ref(), "-reportEvery", "100"]))
            .expect("the knob command line must parse");
        run(&o).unwrap_or_else(|e| panic!("{}: the knob case must run: {e}", k.label));
        let out = written_state(&dir.join("case_jsonc"));
        assert!(!out.is_empty(), "{}: the run wrote nothing to compare", k.label);
        out
    }

    /// **SPEC-LIT 13.4.1's standing test for this driver**: every setting
    /// [`driver_controls`] reads changes what the run writes (class D, a
    /// bit difference). `nonOrthogonalCorrectors` is absent for 13.4.1's
    /// one admissible reason: the tube is orthogonal, so the explicit
    /// correction it repeats is identically zero; it is asserted on the
    /// controls in `the_orthogonal_settings_reach_the_controls`.
    #[test]
    fn every_knob_changes_the_run() {
        if Gpu::new(0).is_err() {
            return;
        }
        const F: &str = "case.jsonc";
        let two_outer: (&str, &str, &str) = (F, r#""outerCorrectors": 1"#, r#""outerCorrectors": 2"#);
        let limited: (&str, &str, &str) = (F, r#""default": "Gauss upwind""#, r#""default": "Gauss vanLeer""#);
        let solver = |f: &'static str, t: &'static str| -> (&'static str, &'static str) { (f, t) };
        let (p_from, p_to) = solver(
            r#"{ "match": "p", "solver": "PBiCGStab", "preconditioner": "DILU", "tolerance": 1e-12 }"#,
            r#"{ "match": "p", "solver": "PBiCGStab", "preconditioner": "DILU", "tolerance": 1e-3 }"#,
        );
        let (u_from, u_to) = solver(
            r#"{ "match": "U", "solver": "PBiCGStab", "preconditioner": "DILU", "tolerance": 1e-12 }"#,
            r#"{ "match": "U", "solver": "PBiCGStab", "preconditioner": "DILU", "tolerance": 1e-3 }"#,
        );
        let (t_from, t_to) = solver(
            r#"{ "match": "T", "solver": "PBiCGStab", "preconditioner": "DILU", "tolerance": 1e-12 }"#,
            r#"{ "match": "T", "solver": "PBiCGStab", "preconditioner": "DILU", "tolerance": 1e-3 }"#,
        );
        let knob = |label: &'static str, from: &'static str, to: &'static str, pre: (&'static str, &'static str, &'static str)| Knob {
            label,
            file: F,
            from,
            to,
            pre,
        };
        let knobs = vec![
            knob("numerics.div.div(phi,U)", r#""default": "Gauss upwind""#,
                r#""default": "Gauss upwind", "div(phi,U)": "Gauss vanLeer""#, NO_PRE),
            knob("numerics.div.div(phi,T)", r#""default": "Gauss upwind""#,
                r#""default": "Gauss upwind", "div(phi,T)": "Gauss vanLeer""#, NO_PRE),
            knob("numerics.div.div(phid,p)", r#""default": "Gauss upwind""#,
                r#""default": "Gauss upwind", "div(phid,p)": "Gauss vanLeer""#, NO_PRE),
            knob("numerics.grad", r#""grad": "Gauss linear""#, r#""grad": "cellLimited Gauss linear 1""#, limited),
            knob("numerics.relaxation.U", r#""U": 0.7"#, r#""U": 0.4"#, two_outer),
            knob("numerics.relaxation.T", r#""T": 0.9"#, r#""T": 0.5"#, two_outer),
            knob("numerics.relaxation.p", r#""p": 0.3 }"#, r#""p": 0.6 }"#, two_outer),
            knob("numerics.algorithm.correctors", r#""correctors": 2"#, r#""correctors": 3"#, NO_PRE),
            knob("numerics.algorithm.outerCorrectors", r#""outerCorrectors": 1"#, r#""outerCorrectors": 3"#, NO_PRE),
            knob("numerics.solvers p", p_from, p_to, NO_PRE),
            knob("numerics.solvers U", u_from, u_to, NO_PRE),
            knob("numerics.solvers T", t_from, t_to, NO_PRE),
            knob("numerics.ddt", r#""ddt": "Euler""#, r#""ddt": "backward""#, NO_PRE),
            knob("run.deltaT", r#""deltaT": 0.00001"#, r#""deltaT": 0.000015"#, NO_PRE),
            knob("physics.compressible.gamma", r#""gamma": 1.4"#, r#""gamma": 1.3"#, NO_PRE),
            knob("physics.compressible.W", r#""W": 0.0289647"#, r#""W": 0.032"#, NO_PRE),
            knob("physics.compressible.Pr", r#""W": 0.0289647, "Pr": 0.71"#, r#""W": 0.0289647, "Pr": 0.5"#, NO_PRE),
            knob("physics.compressible.viscosity", r#""As": 1.458e-6"#, r#""As": 3e-6"#, NO_PRE),
            knob("physics.compressible.pOperating", r#""pOperating": 1e5"#, r#""pOperating": 2e5"#, NO_PRE),
            knob("physics.gravity", r#""gravity": [0, 0, 0]"#, r#""gravity": [-9.81, 0, 0]"#, NO_PRE),
        ];
        let mut inert = Vec::new();
        for (i, k) in knobs.iter().enumerate() {
            let a = run_knob(k, false, &format!("cmp15_knob{i}a"));
            let b = run_knob(k, true, &format!("cmp15_knob{i}b"));
            if a == b {
                inert.push(k.label);
            }
        }
        println!("{} knobs, inert: {inert:?}", knobs.len());
        assert_none_inert(&inert);
    }

    /// The settings the tube cannot show in its output reach the controls.
    #[test]
    fn the_orthogonal_settings_reach_the_controls() {
        let (_dir, case) = tube("cmp15_ctrl");
        edit(&case, r#""nonOrthogonalCorrectors": 0"#, r#""nonOrthogonalCorrectors": 2"#);
        let (hm, cc, l) = load_case(&case).expect("the tube loads");
        let (_, gas) = load_fields_and_gas(&case, &hm, l.as_ref()).expect("the gas");
        let dc = driver_controls(&case, &cc, l.as_ref(), gas, None).expect("the controls");
        assert_eq!(dc.ctrl.pressure.n_non_orth_corr, 2);
        assert!(dc.transient);
        assert_eq!(dc.n_steps, 3);
        let dc = driver_controls(&case, &cc, l.as_ref(), gas, Some(7)).expect("the controls");
        assert_eq!(dc.n_steps, 7, "-iters overrides the case's step count");

        // The OpenFOAM-only settings of a steady run: residualControl (the
        // energy bar spelled `h`) and localEuler's maxCo.
        let dir = openfoam_cavity("cmp15_ctrl_of");
        edit(&dir.join("system/fvSchemes"), "default         Euler;", "default         localEuler;");
        // localEuler pairs with PIMPLE in the reader (a pseudo-transient
        // march), and §125.7 makes it a steady run here.
        edit(&dir.join("system/fvSolution"), "PIMPLE\n{\n    nCorrectors 2;",
            "PIMPLE\n{\n    nCorrectors 2;\n    residualControl { p 1e-3; U 1e-4; h 1e-5; }");
        edit(&dir.join("system/controlDict"), "deltaT          0.001;", "deltaT          0.001;\nmaxCo           5;");
        // A steady run counts endTime in iterations.
        edit(&dir.join("system/controlDict"), "endTime         0.005;", "endTime         50;");
        let (hm, cc, l) = load_case(&dir).expect("the cavity loads");
        let (_, gas) = load_fields_and_gas(&dir, &hm, l.as_ref()).expect("the gas");
        let dc = driver_controls(&dir, &cc, l.as_ref(), gas, None).expect("the controls");
        assert!(!dc.transient);
        assert_eq!(dc.ctrl.momentum.ddt, DdtScheme::LocalEuler);
        assert_eq!(f64::from(dc.ctrl.lts.cfl), 5.0);
        let rc = dc.ctrl.residual_control.expect("residualControl reaches the controls");
        assert_eq!(
            (f64::from(rc.p), f64::from(rc.u), f64::from(rc.t)),
            (f64::from(1e-3 as Scalar), f64::from(1e-4 as Scalar), f64::from(1e-5 as Scalar))
        );
    }

    /// Run `extra` on `case` and return the refusal's text.
    fn refusal(case: &Path, extra: &[&str]) -> String {
        let mut v = vec![case.to_string_lossy().to_string()];
        v.extend(extra.iter().map(|s| s.to_string()));
        let refs: Vec<&str> = v.iter().map(String::as_str).collect();
        match parse(&argv(&refs)).and_then(|o| run(&o)) {
            Ok(_) => panic!("{} {extra:?} ran; it must be refused", case.display()),
            Err(e) => e.to_string(),
        }
    }

    /// A blockgen cavity turned into a compressible OpenFOAM case: laminar,
    /// air, transient Euler at 1e-3 s for five steps.
    fn openfoam_cavity(tag: &str) -> PathBuf {
        let dir = scratch_dir(tag).join("cavity");
        ofgpu::blockgen::write_case(&dir, ofgpu::blockgen::CaseKind::Cavity, 8, 8, 1).expect("write the cavity");
        edit(&dir.join("constant/momentumTransport"), "simulationType  RAS;", "simulationType  laminar;");
        edit(&dir.join("system/fvSchemes"), "default         steadyState;", "default         Euler;");
        edit(&dir.join("system/fvSchemes"), "div(phi,T)       bounded Gauss upwind;", "div(phi,T)       Gauss upwind;");
        edit(&dir.join("system/fvSolution"), "SIMPLE\n{", "PIMPLE\n{\n    nCorrectors 2;");
        // The all-speed pressure matrix is asymmetric: DIC is refused by name.
        edit(&dir.join("system/fvSolution"), "preconditioner  DIC;\n        tolerance       1e-08;",
            "preconditioner  DILU;\n        tolerance       1e-08;");
        edit(&dir.join("system/controlDict"), "endTime         1;", "endTime         0.005;");
        edit(&dir.join("system/controlDict"), "deltaT          1;", "deltaT          0.001;");
        std::fs::write(
            dir.join("constant/thermophysicalProperties"),
            "FoamFile { format ascii; class dictionary; object thermophysicalProperties; }
thermoType
{
    type            hePsiThermo;
    mixture         pureMixture;
    transport       sutherland;
    thermo          hConst;
    equationOfState perfectGas;
    specie          specie;
    energy          sensibleInternalEnergy;
}
mixture
{
    specie { molWeight 28.9647; }
    thermodynamics { Cp 1004.692579691832; Hf 0; }
    transport { As 1.458e-06; Ts 110.4; }
}
pOperating [1 -1 -2 0 0 0 0] 1e5;
",
        )
        .expect("write thermophysicalProperties");
        dir
    }

    /// Rewrite one patch entry of the cavity's `0/<field>` (64 cells).
    fn set_patch(dir: &Path, field: &str, patch: &str, type_name: &str, kv: &[(&str, &str)]) {
        let path = dir.join("0").join(field);
        let mut f = read_scalar_field(&path, 64).expect("read the cavity field");
        let spec = f.boundary.get_mut(patch).expect("the cavity has the patch");
        spec.type_name = type_name.to_string();
        for (k, v) in kv {
            spec.extra.insert((*k).to_string(), (*v).to_string());
        }
        ofgpu::io::fields::write_scalar_field(&path, &f, "0").expect("write the cavity field");
    }

    /// SPEC-LIT §126.8 on the OpenFOAM route: `totalPressure` with `gamma`
    /// and `totalTemperature` read from `0/p` and `0/T` become two rules,
    /// and the cavity runs with them (the lid's faces carry no normal flow,
    /// so both are the zero-gradient branch).
    #[test]
    fn an_openfoam_case_with_the_compressible_inlets_runs() {
        if Gpu::new(0).is_err() {
            return;
        }
        let dir = openfoam_cavity("cmp16_of_inlets");
        set_patch(&dir, "p", "movingWall", "totalPressure", &[("p0", "uniform 0"), ("gamma", "1.4")]);
        set_patch(&dir, "T", "movingWall", "totalTemperature", &[("T0", "uniform 300"), ("gamma", "1.4")]);
        let o = parse(&argv(&[dir.to_string_lossy().as_ref()])).unwrap();
        let end = run(&o).expect("the cavity with the inlets runs");
        assert_eq!(end.steps, 5);
        assert_eq!(end.last.expect("an outer report").bounds.total(), 0);
        let p = std::fs::read_to_string(dir.join("0.005").join("p")).expect("p is written");
        assert!(p.contains("totalPressure"), "the written p keeps the type as read");
    }

    /// SPEC-LIT §126.9: the OpenFOAM route reads `waveTransmissive` on `p`
    /// and runs.
    #[test]
    fn an_openfoam_case_with_a_wave_transmissive_outlet_runs() {
        if Gpu::new(0).is_err() {
            return;
        }
        let dir = openfoam_cavity("cmp17_of_wave");
        set_patch(
            &dir,
            "p",
            "movingWall",
            "waveTransmissive",
            &[("lInf", "0.1"), ("sigma", "0.25"), ("fieldInf", "0"), ("gamma", "1.4")],
        );
        let o = parse(&argv(&[dir.to_string_lossy().as_ref()])).unwrap();
        let end = run(&o).expect("the cavity with a waveTransmissive p runs");
        assert_eq!(end.steps, 5);
        assert_eq!(end.last.expect("an outer report").bounds.total(), 0);
        let p = std::fs::read_to_string(dir.join("0.005").join("p")).expect("p is written");
        assert!(p.contains("waveTransmissive"), "the written p keeps the type as read");
    }

    /// SPEC-LIT §122.4: each thing v1 does not have is refused, and the
    /// message names the setting.
    #[test]
    fn refusals_name_the_setting() {
        if Gpu::new(0).is_err() {
            return;
        }
        let _g = ofgpu::io::contract::permissive_test_guard();
        ofgpu::io::contract::set_permissive(false);
        type Row<'a> = (&'a str, &'a str, &'a str, &'a [&'a str], &'a [&'a str]);
        let jsonc: Vec<Row> = vec![
            ("SA", r#""initial":"#,
                r#""turbulence": {"kind":"RAS","model":"SpalartAllmaras","wallFunctions":{"kappa":0.41,"E":9.8}}, "initial":"#,
                &[], &["physics.turbulence", "SpalartAllmaras", "CMP-25"]),
            ("SST", r#""initial":"#,
                r#""turbulence": {"kind":"RAS","model":"kOmegaSST","wallFunctions":{"kappa":0.41,"E":9.8}}, "initial":"#,
                &[], &["physics.turbulence", "kOmegaSST"]),
            ("sources", r#""run":"#,
                r#""sources": [{ "type": "momentumSource", "field": "U", "bodyForce": [3.9, 0.0, -1.5] }], "run":"#,
                &[], &["sources"]),
            ("motion", r#""run":"#,
                r#""motion": { "patches": [ { "patch": "outlet", "rule": "fixed" } ] }, "run":"#,
                &[], &["motion"]),
            ("boussinesq", r#""buoyancy": "densityRatio""#, r#""buoyancy": "boussinesq""#,
                &[], &["physics.buoyancy", "boussinesq"]),
            ("phid", r#""default": "Gauss upwind""#, r#""default": "Gauss upwind", "div(phid,p)": "Gauss linear""#,
                &[], &["div(phid,p)"]),
            ("bounded", r#""default": "Gauss upwind""#, r#""default": "bounded Gauss upwind""#,
                &[], &["div(phi,U)", "bounded"]),
            ("inletOutlet", r#""U": { "type": "zeroGradient" }"#,
                r#""U": { "type": "inletOutlet", "inletValue": [0, 0, 0] }"#,
                &[], &["U patch outlet", "inletOutlet"]),
            ("nogas", GAS_BLOCK, "", &[], &["physics.compressible"]),
            ("np", "", "", &["-np", "2"], &["-np", "decomposition"]),
            ("writeEvery", "", "", &["-writeEvery", "2"], &["-writeEvery"]),
        ];
        for (tag, from, to, extra, wanted) in jsonc {
            let (_dir, case) = tube(&format!("cmp15_ref_{tag}"));
            if !from.is_empty() {
                edit(&case, from, to);
            }
            if tag == "boussinesq" {
                edit(&case, r#""TRef": 300"#, r#""TRef": 300, "beta": 3.3e-3"#);
            }
            let e = refusal(&case, extra);
            for w in wanted {
                assert!(e.contains(w), "{tag}: {w:?} not in {e}");
            }
        }

        // The OpenFOAM route.
        type Mutate = Box<dyn Fn(&Path)>;
        let of: Vec<(&str, Mutate, &[&str])> = vec![
            ("processor0", Box::new(|d: &Path| std::fs::create_dir_all(d.join("processor0")).unwrap()),
                &["processor0", "decomposition"]),
            ("dynamicMeshDict", Box::new(|d: &Path| std::fs::write(d.join("constant/dynamicMeshDict"),
                "FoamFile { format ascii; class dictionary; object dynamicMeshDict; }\ndynamicFvMesh dynamicMotionSolverFvMesh;\n").unwrap()),
                &["constant/dynamicMeshDict", "dynamicMotionSolverFvMesh"]),
            ("fvOptions", Box::new(|d: &Path| std::fs::write(d.join("system/fvOptions"), "FoamFile { }\n").unwrap()),
                &["system/fvOptions", "sources"]),
            ("kEpsilon", Box::new(|d: &Path| edit(&d.join("constant/momentumTransport"),
                "simulationType  laminar;", "simulationType  RAS;")),
                &["constant/momentumTransport"]),
            ("nothermo", Box::new(|d: &Path| std::fs::remove_file(d.join("constant/thermophysicalProperties")).unwrap()),
                &["constant/thermophysicalProperties"]),
            // SPEC-LIT §126.8: the kinematic totalPressure is not a gas's.
            ("totalPressure", Box::new(|d: &Path| set_patch(d, "p", "movingWall", "totalPressure", &[("p0", "uniform 0")])),
                &["p patch movingWall", "totalPressure", "gamma"]),
        ];
        for (tag, mutate, wanted) in of {
            let dir = openfoam_cavity(&format!("cmp15_of_{tag}"));
            mutate(&dir);
            let e = refusal(&dir, &[]);
            for w in wanted {
                assert!(e.contains(w), "{tag}: {w:?} not in {e}");
            }
        }
    }

    /// The five fields of a written time, read back as text.
    fn five_fields(dir: &Path) -> Vec<String> {
        OUTPUT_FIELDS
            .iter()
            .map(|f| {
                std::fs::read_to_string(dir.join(f))
                    .unwrap_or_else(|e| panic!("{} has no {f}: {e}", dir.display()))
            })
            .collect()
    }

    /// SPEC-LIT §125.8's smoke: the 50-cell tube, 30 steps, in under 30 s
    /// (class P), with no bounds violation and the five fields written.
    #[test]
    fn a_fifty_cell_tube_runs_in_under_thirty_seconds() {
        if Gpu::new(0).is_err() {
            return;
        }
        let (dir, case) = tube("cmp15_smoke");
        edit(&case, r#""endTime": 0.00003"#, r#""endTime": 0.0003"#);
        let o = parse(&argv(&[case.to_string_lossy().as_ref()])).unwrap();
        let end = run(&o).expect("the tube runs");
        println!("smoke: {} steps in {:.3} s", end.steps, end.seconds);
        assert_eq!(end.steps, 30);
        assert!(end.seconds < 30.0, "the 50-cell tube took {} s", end.seconds);
        let last = end.last.expect("an outer report");
        assert_eq!(last.bounds.total(), 0);
        assert!(last.mach_max > 0.14 && last.mach_max < 0.15, "M_max {}", last.mach_max);
        let files = five_fields(&dir.join("case_jsonc").join("0.0003"));
        assert!(files[4].contains("object      Mach;"), "{}", files[4]);
    }

    /// SPEC-LIT §126.10 through the JSONC route: the tube with a `farfield`
    /// at both ends (free stream 50 m/s, 300 K) runs, and the written `p`
    /// keeps the type as lowered, `characteristicFarfield`.
    #[test]
    fn a_jsonc_farfield_tube_runs() {
        if Gpu::new(0).is_err() {
            return;
        }
        let (dir, case) = tube("cmp18_farfield");
        let ff = r#"{ "match": "(inlet|outlet)", "kind": "farfield", "freestream": { "U": [50, 0, 0], "p": 0, "T": 300 } },"#;
        edit(
            &case,
            r#"{ "match": "inlet", "kind": "inlet", "U": { "type": "fixedValue", "value": [50, 0, 0] }, "T": { "type": "fixedValue", "value": 320 } },"#,
            ff,
        );
        edit(
            &case,
            r#"    { "match": "outlet", "kind": "open", "U": { "type": "zeroGradient" }, "p": { "type": "fixedValue", "value": 0 }, "T": { "type": "zeroGradient" } },
"#,
            "",
        );
        edit(&case, r#""T": 300.0 }"#, r#""T": 305.0 }"#);
        let o = parse(&argv(&[case.to_string_lossy().as_ref()])).unwrap();
        let end = run(&o).expect("the far-field tube runs");
        assert_eq!(end.steps, 3);
        assert_eq!(end.last.expect("an outer report").bounds.total(), 0);
        let root = dir.join("case_jsonc");
        let last = std::fs::read_dir(&root)
            .expect("the case is written")
            .filter_map(|e| e.ok()?.file_name().into_string().ok())
            .filter(|n| n.parse::<f64>().is_ok_and(|t| t > 0.0))
            .max_by(|a, b| a.parse::<f64>().unwrap().total_cmp(&b.parse::<f64>().unwrap()))
            .expect("a written time");
        let files = five_fields(&root.join(last));
        assert!(files[1].contains("characteristicFarfield"), "{}", files[1]);
    }

    /// The OpenFOAM route: the cavity runs and writes the five fields.
    #[test]
    fn an_openfoam_case_runs_and_writes_the_five_fields() {
        if Gpu::new(0).is_err() {
            return;
        }
        let dir = openfoam_cavity("cmp15_of_run");
        let o = parse(&argv(&[dir.to_string_lossy().as_ref()])).unwrap();
        let end = run(&o).expect("the cavity runs");
        assert_eq!(end.steps, 5);
        assert_eq!(end.last.expect("an outer report").bounds.total(), 0);
        five_fields(&dir.join("0.005"));
    }
}
