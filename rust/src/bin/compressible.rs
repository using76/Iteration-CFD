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
//! SPEC-LIT §158.8 (DST-09): a JSONC case's `vent` block runs the one-domain
//! vent here and nowhere else. The run starts on the mesh split at the
//! membrane, reads the membrane's pressure difference after every step, and
//! on opening hands the state over to a solver on the case's own mesh; a
//! steady run with a vent is refused, and every other driver refuses the
//! case by name (`LoweredCase::refuse_vent`).
//!
//! SPEC-LIT §158.6 (DST-10): a JSONC case's dust model runs here too, on
//! the mixture gas law (152.8)-(152.9) about the case's `physics.compressible`
//! air. The coupling - the c~ equation (155.3) on the solver's own mass
//! flux, the heat release (158.4), R_mix(c~), the local ρ_u(p) of (155.10),
//! the ρ-weighted SST of §156 as the one carried turbulence model, and the
//! flame-Courant limit (155.9) - is
//! [`ofgpu::dust::source::CompressibleDust`], attached before
//! `initialise`; a vent beside it, a non-Euler ddt and any §126 rule are
//! refused by name. Without a dust block nothing in the run changes, bit
//! for bit.
//!
//! Written from:
//!   ofgpu SPEC-LIT.md §158.1, §158.8 - the membrane and the vent as built
//!   ofgpu SPEC-LIT.md §158.4, §158.6 - the mixture heat release, and the
//!     dust model this driver runs: its refusals, its c~ controls, its
//!     (155.9) checks and its four written fields
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
use ofgpu::dust::source::{CompressibleDust, DustSetup};
use ofgpu::dust::vent;
use ofgpu::field_setup::{
    harvest_scalar_field, harvest_vector_field, setup_scalar_field, wall_coeffs_from_case,
    NutRoughness, WallFaces,
};
use ofgpu::fv::DivScheme;
use ofgpu::io::case::{find_start_time, format_time_name, CaseControls};
use ofgpu::io::case_json::LoweredCase;
use ofgpu::io::fields::{
    read_scalar_field, read_vector_field, PatchFieldSpec, RawScalarField, RawVectorField,
};
use ofgpu::io::voxelmap::LatticeSpec;
use ofgpu::io::{DivEntry, FoamDict};
use ofgpu::models::coupled::CoupledTurbulence;
use ofgpu::models::k_omega_sst::{RhoSstOptions, SstSust};
use ofgpu::models::registry::{build_coupled, select_turbulence_model, RasModel};
use ofgpu::models::{sa_coeffs, SpalartAllmaras};
use ofgpu::momentum::BuoyancyForm;
use ofgpu::timescheme::DdtScheme;
use ofgpu::turbulence::{FlowState, TurbulenceControls};
use ofgpu::walldistance;
use ofgpu::{Error, Gpu, GpuMesh, HostMesh, Result, Scalar, Vec3};

#[path = "common/mod.rs"]
mod common;

use common::{
    device_banner, g, load_case_vented, next_arg, output_root, parse_output_formats, sci,
    CaseNumerics, OutputFormat,
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
        // SPEC-LIT §158.6 lifts CMP-15's non-SA refusal for kOmegaSST ONLY:
        // the dust case's ρ-weighted form (§156) is the one model the
        // mixture coupling carries, and it exists beside a `dust` block
        // alone.
        let is_dust_sst = lowered.is_some_and(|l| l.dust.is_some())
            && sel.model == RasModel::KOmegaSST
            && sel.les.is_none()
            && sel.des.is_none();
        let is_sa = sel.model == RasModel::SpalartAllmaras && sel.les.is_none() && sel.des.is_none();
        // SPEC-LIT §127.1.2: Spalart-Allmaras runs here in its conservative
        // form - but only through the OpenFOAM case format, whose `0/nuTilda`
        // states the field the equation transports. The JSONC format has no
        // nuTilda initial field (`LoweredCase` carries none), so an SA case
        // stated in JSONC would run the model without its unknown.
        if !is_laminar && !is_dust_sst && !(is_sa && lowered.is_none()) {
            let setting = if lowered.is_some() {
                "physics.turbulence"
            } else {
                "constant/momentumTransport"
            };
            let name = sel.describe();
            let why = if is_sa {
                format!(
                    "{name}: SpalartAllmaras runs in ofgpu-compressible in its conservative \
                     form (SPEC-LIT §127.1), which reads a nuTilda initial field - and the \
                     JSONC format has no nuTilda to state it with. Alternative: the OpenFOAM \
                     case format"
                )
            } else if sel.model == RasModel::KOmegaSST && sel.les.is_none() && sel.des.is_none() {
                format!(
                    "{name}: kOmegaSST runs in ofgpu-compressible only with a dust block, as \
                     §156's rho-weighted form (SPEC-LIT §158.6); without the model's own \
                     closures this driver would run it as a substitution (SPEC-LIT §13.4). \
                     Alternative: laminar, or the dust block"
                )
            } else {
                format!(
                    "{name}: every turbulence model but SpalartAllmaras is refused \
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

/// The `nut` patch types the SA coupling accepts (SPEC-LIT §56.7, §127.1):
/// SA has no `k`, so a k-based wall function would be a wall treatment with
/// no model behind it here - `nutLowReWallFunction` is the one wall type.
const SA_NUT_TYPES: &[&str] = &[
    "calculated",
    "fixedValue",
    "zeroGradient",
    "nutLowReWallFunction",
    "empty",
    "symmetry",
    "symmetryPlane",
    "wedge",
    "cyclic",
];

/// `nut`'s patch types, checked apart from `nuTilda`'s - the wall-function
/// families the incompressible drivers take name a `k` this model does not
/// have (SPEC-LIT §56.7).
fn check_nut_bcs(nut: &RawScalarField) -> Result<()> {
    for (patch, spec) in &nut.boundary {
        if !SA_NUT_TYPES.contains(&spec.type_name.as_str()) {
            return Err(refuse(
                &format!("nut patch {patch}"),
                &format!(
                    "{}: SpalartAllmaras has no k, so this wall treatment has no model \
                     behind it here (SPEC-LIT §56.7, §127.1). Accepted: {}",
                    spec.type_name,
                    SA_NUT_TYPES.join(", ")
                ),
            ));
        }
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

/// The `0/` fields and controls of the SA coupling (SPEC-LIT §127.1.2), the
/// pattern of `ofgpu-sa`'s setup. `0/nuTilda` and `0/nut` are REQUIRED - the
/// equation transports nu~, and nut's own patch types are what decide its
/// wall treatment (SPEC-LIT §15.5) - and nuTilda's types go through the same
/// static check U, p and T do. Reached on the OpenFOAM route only (§122.4's
/// refusal above).
fn sa_fields(
    case_path: &Path,
    cc: &CaseControls,
    dc: &DriverControls,
    hm: &HostMesh,
) -> Result<(RawScalarField, RawScalarField, TurbulenceControls)> {
    let dir = case_path.join(find_start_time(case_path)?);
    for name in ["nuTilda", "nut"] {
        if !dir.join(name).exists() {
            return Err(refuse(
                &format!("{}/{}", dir.display(), name),
                &format!(
                    "no {name} field: SpalartAllmaras transports nu~ and nut's own patch \
                     types are its wall treatment (SPEC-LIT §127.1, §15.5). Add the file"
                ),
            ));
        }
    }
    let raw_nt = read_scalar_field(&dir.join("nuTilda"), hm.n_cells)?;
    let raw_nut = read_scalar_field(&dir.join("nut"), hm.n_cells)?;
    check_static_bcs("nuTilda", &raw_nt.boundary)?;
    check_nut_bcs(&raw_nut)?;

    // The model's controls are the case's own `system/` entries, with the
    // ddt pinned to the flow's - §127.1.2 refuses the disagreement, and a
    // steady run takes steadyState and the nuTilda relaxation instead
    // (`localEuler` on the model itself is §86.7's refusal).
    let num = CaseNumerics::read(case_path, cc, None)?;
    let mut ctrl = cc.turb;
    if dc.transient {
        ctrl.steady = false;
        ctrl.ddt = dc.ctrl.momentum.ddt;
        ctrl.delta_t = dc.ctrl.momentum.delta_t;
    } else {
        ctrl.steady = true;
        ctrl.ddt = DdtScheme::SteadyState;
    }
    // A `bounded` entry is refused by name like U's and T's: div(mdot) is
    // -ddt(rho) in a compressible flow (§123.1, §125.1).
    ctrl.eps_div_scheme = unbounded("div(phi,nuTilda)", num.div("div(phi,nuTilda)")?, false)?;
    ctrl.eps_bounded_convection = false;
    Ok((raw_nt, raw_nut, ctrl))
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

/// The c~ equation's controls by its own entry names (§13.4.1), the local
/// copy of `lowmach_dust::c_controls` this driver forms (SPEC-LIT §158.6):
/// `div(phi,c)` where the case names one, otherwise §155.3's bounded Gauss
/// upwind - never the case's `default`; the `c` solver and relaxation
/// (1 where the case names none, a dust run being transient); `grad(c)`;
/// `snGradSchemes default`; Euler at the run's dt, and never steady.
pub fn dust_c_controls(
    num: &CaseNumerics<'_>,
    base: &TurbulenceControls,
    dt: Scalar,
) -> Result<TurbulenceControls> {
    let conv = match num.div_named("div(phi,c)")? {
        Some(a) => a,
        None => DivEntry { scheme: DivScheme::Upwind, bounded: true },
    };
    let mut c = *base;
    c.k_solver = num.solver("c", base.k_solver)?;
    c.k_relax = num.relax("c", 1.0)?;
    c.grad_scheme = num.grad("grad(c)")?;
    c.sn_grad = num.sn_grad("default")?;
    c.div_scheme = conv.scheme;
    c.bounded_convection = conv.bounded;
    c.steady = false;
    c.delta_t = dt;
    Ok(c)
}

// ==========================================================================
//  Fields
// ==========================================================================

struct InitialFields {
    u: RawVectorField,
    p: RawScalarField,
    t: RawScalarField,
    /// The SA fields as read, kept for the written patch types (§127.1.2);
    /// `None` on a laminar run.
    nu_tilda: Option<RawScalarField>,
    nut: Option<RawScalarField>,
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
            let mut f = InitialFields {
                u: l.u_field.to_raw(hm.n_cells),
                p: l.p_field.to_raw(hm.n_cells),
                t: t.to_raw(hm.n_cells),
                nu_tilda: None,
                nut: None,
            };
            // SPEC-LIT 125.8.1: the initial regions override `U`, `p` and/or
            // `T` in the cells whose centre lies in each box, in order. The
            // only driver that applies them; `JsonCase::lower` refuses them on
            // a case without `physics.compressible`.
            let counts = l.apply_initial_regions(hm, &mut f.u, &mut f.p, &mut f.t)?;
            for (i, n) in counts.iter().enumerate() {
                println!("initial.regions[{i}]: {n} cells (SPEC-LIT 125.8.1)");
            }
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
                nu_tilda: None,
                nut: None,
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

/// The four dust fields a dust run adds (SPEC-LIT §158.6): `c` (the
/// progress variable), `Ut` (U_t', m/s), `Cof` (the last measured
/// flame-Courant number) and `HRR` (q, W/m3). An output block may name
/// them only beside a `dust` block.
const DUST_FIELDS: [&str; 4] = ["c", "Ut", "Cof", "HRR"];

/// The field names `check_fields` reads, the five base ones plus the four
/// dust ones exactly when the case has a `dust` block (SPEC-LIT §158.6).
fn available_fields(has_dust: bool) -> Vec<&'static str> {
    let mut v: Vec<&'static str> = OUTPUT_FIELDS.to_vec();
    if has_dust {
        v.extend(DUST_FIELDS);
    }
    v
}

/// Write the five fields - plus `nuTilda` and `nut` when SA is attached
/// (§127.1.2) - at time label `t` (seconds, or the iteration count of a
/// steady run), through the one pipeline. With a dust coupling the
/// write carries the four dust fields as well, and `Mach` is the mixture's
/// - `a^2 = gamma(c~) r(c~) T` of the hConst pair (SPEC-LIT §158.6); the
/// scalar `cof` is the last MEASURED flame-Courant number `Cof` writes.
///
/// `hm` is the CASE'S own mesh in both phases of a vented run: the split
/// mesh's boundary arrays start with the case's boundary faces in the same
/// order (its cells are the same), so the harvest's per-patch slices read
/// the right values of either state (SPEC-LIT §158.1). A dust run is never
/// vented, so its boundary faces are always `hm`'s own.
#[allow(clippy::too_many_arguments)]
fn write_time(
    gpu: &Gpu,
    solver: &Compressible,
    s: &CompressibleState,
    gas: &PerfectGas,
    hm: &HostMesh,
    fields: &InitialFields,
    t: f64,
    pipeline: &mut ofgpu::io::OutputPipeline,
    force: bool,
    volume: Option<&ofgpu::io::VolumeTarget>,
    dust: Option<&CompressibleDust>,
    cof: Option<f64>,
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
    // boundary arrays, as the harvest itself slices them. With a dust
    // coupling the gas law is the mixture's (152.8)-(152.9), so Mach reads
    // `a^2 = gamma(c~) r(c~) T` with `gamma = cp(c~)/(cp(c~) - r(c~))` of
    // the hConst pair, host f64 - not the case gas (SPEC-LIT §158.6).
    let k = gas.constants();
    let mut out_m = calculated_like(&fields.t, "Mach");
    harvest_scalar_field(gpu, &mut out_m, &s.t, hm)?;
    out_m.dimensions = "[0 0 0 0 0 0 0]".to_string();
    let (ub, tb) = (gpu.download(&s.u.bf)?, gpu.download(&s.t.bf)?);
    match dust {
        None => {
            out_m.internal = out_u
                .internal
                .iter()
                .zip(&out_t.internal)
                .map(|(u, tt)| k.mach(*u, *tt))
                .collect();
            let face_mach: Vec<Scalar> = ub.iter().zip(&tb).map(|(u, tt)| k.mach(*u, *tt)).collect();
            for patch in &hm.patches {
                if let Some(spec) = out_m.boundary.get_mut(&patch.name) {
                    spec.value = face_mach[patch.start..patch.start + patch.size].to_vec();
                }
            }
        }
        Some(d) => {
            let hc = d.hconst();
            let cf = gpu.download(&d.c().f)?;
            let cfb = gpu.download(&d.c().bf)?;
            let mach_mix = |c: f64, sp2: f64, tt: f64| -> Scalar {
                let (cp, r) = (hc.cp(c), hc.r(c));
                let gamma = cp / (cp - r);
                (sp2.sqrt() / (gamma * r * tt).sqrt()) as Scalar
            };
            let speed2 = |u: &Vec3| -> f64 {
                let (ux, uy, uz) = (f64::from(u.x), f64::from(u.y), f64::from(u.z));
                ux * ux + uy * uy + uz * uz
            };
            out_m.internal = out_u
                .internal
                .iter()
                .zip(&out_t.internal)
                .zip(&cf)
                .map(|((u, tt), c)| mach_mix(f64::from(*c), speed2(u), f64::from(*tt)))
                .collect();
            let face_mach: Vec<Scalar> = ub
                .iter()
                .zip(&tb)
                .zip(&cfb)
                .map(|((u, tt), c)| mach_mix(f64::from(*c), speed2(u), f64::from(*tt)))
                .collect();
            for patch in &hm.patches {
                if let Some(spec) = out_m.boundary.get_mut(&patch.name) {
                    spec.value = face_mach[patch.start..patch.start + patch.size].to_vec();
                }
            }
        }
    }

    // §127.1.2: the SA fields carry the case's own patch types as read, and
    // kinematic viscosity dimensions. Gathered first, so the writes below
    // can borrow them.
    let (out_nt, out_nut) = match solver.turbulence() {
        Some(sa) => {
            let missing = |f: &str| {
                Error::Config(format!("{DRIVER}: SpalartAllmaras is attached but {f} was not read"))
            };
            let raw_nt = fields.nu_tilda.as_ref().ok_or_else(|| missing("nuTilda"))?;
            let raw_nut = fields.nut.as_ref().ok_or_else(|| missing("nut"))?;
            let mut o1 = raw_nt.types_only();
            harvest_scalar_field(gpu, &mut o1, sa.nu_tilda(), hm)?;
            o1.dimensions = "[0 2 -1 0 0 0 0]".to_string();
            let mut o2 = raw_nut.types_only();
            harvest_scalar_field(gpu, &mut o2, sa.nut(), hm)?;
            o2.dimensions = "[0 2 -1 0 0 0 0]".to_string();
            (Some(o1), Some(o2))
        }
        None => (None, None),
    };

    let mut foam = vec![
        ofgpu::io::FoamField::vector("U", &out_u),
        ofgpu::io::FoamField::scalar("p", &out_p),
        ofgpu::io::FoamField::scalar("T", &out_t),
        ofgpu::io::FoamField::scalar("rho", &out_rho),
        ofgpu::io::FoamField::scalar("Mach", &out_m),
    ];
    let mut vis = vec![
        ofgpu::io::OutputField::vector("U", &out_u.internal),
        ofgpu::io::OutputField::scalar("p", &out_p.internal),
        ofgpu::io::OutputField::scalar("T", &out_t.internal),
        ofgpu::io::OutputField::scalar("rho", &out_rho.internal),
        ofgpu::io::OutputField::scalar("Mach", &out_m.internal),
    ];
    // The four dust fields (SPEC-LIT §158.6): `c` harvested from the c~
    // field into a calculated_like copy; `Ut` (U_t', m/s), `Cof` (the last
    // measured number, uniform) and `HRR` (q~ of (158.4), W/m3) from the
    // coupling's own cell buffers, each boundary face its owner's value.
    let mut out_c = calculated_like(&fields.t, "c");
    let mut out_ut = calculated_like(&fields.t, "Ut");
    let mut out_cof = calculated_like(&fields.t, "Cof");
    let mut out_hrr = calculated_like(&fields.t, "HRR");
    if let Some(d) = dust {
        harvest_scalar_field(gpu, &mut out_c, d.c(), hm)?;
        out_c.dimensions = "[0 0 0 0 0 0 0]".to_string();
        let owner_fill = |raw: &mut RawScalarField, dims: &str, cell: &[Scalar]| {
            raw.dimensions = dims.to_string();
            raw.internal = cell[..hm.n_cells].to_vec();
            for p in &hm.patches {
                if let Some(spec) = raw.boundary.get_mut(&p.name) {
                    spec.value = (p.start..p.start + p.size)
                        .map(|f| cell[hm.b_face_cells[f] as usize])
                        .collect();
                }
            }
        };
        let ut: Vec<Scalar> = gpu.download(d.u_t_local())?;
        let hrr: Vec<Scalar> = gpu.download(d.heat_release())?;
        // (155.9) per cell, as the last measurement formed it; zeros before
        // the first one.
        let cof_cell: Vec<Scalar> = match cof {
            Some(_) => gpu.download(d.courant())?,
            None => vec![0.0 as Scalar; hm.n_cells],
        };
        owner_fill(&mut out_ut, "[0 1 -1 0 0 0 0]", &ut);
        owner_fill(&mut out_cof, "[0 0 0 0 0 0 0]", &cof_cell);
        owner_fill(&mut out_hrr, "[1 -1 -3 0 0 0 0]", &hrr);
        foam.push(ofgpu::io::FoamField::scalar("c", &out_c));
        foam.push(ofgpu::io::FoamField::scalar("Ut", &out_ut));
        foam.push(ofgpu::io::FoamField::scalar("Cof", &out_cof));
        foam.push(ofgpu::io::FoamField::scalar("HRR", &out_hrr));
        vis.push(ofgpu::io::OutputField::scalar("c", &out_c.internal));
        vis.push(ofgpu::io::OutputField::scalar("Ut", &out_ut.internal));
        vis.push(ofgpu::io::OutputField::scalar("Cof", &out_cof.internal));
        vis.push(ofgpu::io::OutputField::scalar("HRR", &out_hrr.internal));
    }
    if let (Some(nt), Some(nut)) = (&out_nt, &out_nut) {
        foam.push(ofgpu::io::FoamField::scalar("nuTilda", nt));
        foam.push(ofgpu::io::FoamField::scalar("nut", nut));
        vis.push(ofgpu::io::OutputField::scalar("nuTilda", &nt.internal));
        vis.push(ofgpu::io::OutputField::scalar("nut", &nut.internal));
    }
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
    /// The membrane's state at the end of a run with a `vent` block; `None`
    /// without one (SPEC-LIT §158.8).
    vent: Option<ofgpu::dust::vent::VentState>,
}

/// Everything a case's `vent` block adds to a run (SPEC-LIT §158.8): the
/// lowered block, the mesh split at the membrane, the host mesh built from
/// the split one, and the closed-stage lowered fields with the two membrane
/// patches. `None` for every case without the block.
struct VentRun {
    spec: ofgpu::dust::vent::VentSpec,
    split: ofgpu::dust::vent::VentSplit,
    hm_split: HostMesh,
    closed: ofgpu::compressible::bc::Lowered,
}

/// One opening check of the membrane (SPEC-LIT §158.1), run at n = 0 (after
/// `initialise`) and after every step n (t = `t` s) while it is closed: read
/// the area-weighted pressure difference from the solved gauge `p`, and when
/// it has reached `pStat`, print the opening and hand the run over to the
/// merged-mesh solver [`vent::open_onto`] builds. Returns the handed-over
/// (solver, state) when the membrane opened - the caller replaces its own -
/// and the reading either way.
#[allow(clippy::too_many_arguments)]
fn check_membrane<'m>(
    gpu: &Gpu,
    v: &VentRun,
    trigger: &mut vent::VentTrigger,
    membrane: &mut vent::Membrane,
    n: u64,
    t: f64,
    solver: &Compressible<'_>,
    s: &CompressibleState,
    mesh: &'m GpuMesh,
    hm: &HostMesh,
    ctrl: CompressibleControls,
    inlets: &ofgpu::compressible::bc::Lowered,
) -> Result<(Option<(Compressible<'m>, CompressibleState)>, vent::VentReading)> {
    let r = trigger.read(gpu, &s.p.f)?;
    if membrane.check(n, t, &r) {
        println!(
            "vent: opened after step {n} (t = {} s): dp = {} Pa >= pStat {} Pa \
             (p_in {} Pa, p_out {} Pa); the {} faces are interior faces from \
             step {} on, SPEC-LIT 158.1",
            g(t),
            g(r.dp),
            g(v.spec.p_stat),
            g(r.p_in),
            g(r.p_out),
            v.split.faces.len(),
            n + 1
        );
        let handed = vent::open_onto(gpu, &v.split, solver, s, mesh, hm, ctrl, inlets)?;
        return Ok((Some(handed), r));
    }
    Ok((None, r))
}

fn run(o: &Options) -> Result<RunEnd> {
    let t0 = Instant::now();
    let gpu = Gpu::new(0)?;
    println!("{}", device_banner(&gpu, "compressible")?);

    let (hm, cc, lowered) = load_case_vented(&o.case_path)?;
    refuse_out_of_scope(&o.case_path, &cc, lowered.as_ref())?;
    hm.print_report();

    let (mut fields, gas) = load_fields_and_gas(&o.case_path, &hm, lowered.as_ref())?;
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

    // ---- SPEC-LIT §158.6: what a dust case may not state here, by name,
    // before anything is built. The c~ equation's mixture gas law forms
    // every boundary face from the c~ of its owner, which a §126-rewritten
    // face has none of, and the hand-over of c~ and the closures on
    // opening is DST-11's.
    let has_dust = lowered.as_ref().is_some_and(|l| l.dust.is_some());
    if has_dust {
        let inl = &inlets;
        if !inl.rules.is_empty()
            || !inl.outlets.is_empty()
            || !inl.farfields.is_empty()
            || !inl.freestreams.is_empty()
        {
            return Err(refuse(
                "boundaryField",
                "a §126 rule (inlet, outlet, far-field or free stream) is attached, and \
                 the mixture gas law forms every boundary face from the c~ of its owner, \
                 which a rule-rewritten face has none of (SPEC-LIT §158.6). Alternative: \
                 fixedValue and zeroGradient conditions only",
            ));
        }
        if lowered.as_ref().and_then(|l| l.vent.as_ref()).is_some() {
            return Err(refuse(
                "vent",
                "the hand-over of c~ and the closures on opening is DST-11's; a dust case \
                 runs sealed here (SPEC-LIT §158.6). Alternative: drop the vent block",
            ));
        }
    }

    let dc = driver_controls(&o.case_path, &cc, lowered.as_ref(), gas, o.iters)?;
    print_controls(&dc);
    // SPEC-LIT §158.6: the c~ equation keeps one old level and the mixture
    // energy equation is built for one Euler step for every cell alike, so
    // a dust run is a transient Euler run and nothing else.
    if has_dust && dc.ctrl.momentum.ddt != DdtScheme::Euler {
        return Err(refuse(
            "numerics.ddt",
            &format!(
                "{:?} with a dust case: the c~ equation (155.3) keeps one old level and \
                 the mixture energy equation (SPEC-LIT §158.4) rides one Euler step for \
                 every cell alike. Alternative: Euler",
                dc.ctrl.momentum.ddt
            ),
        ));
    }

    // ---- the vent (SPEC-LIT §158.8), before the output pipeline: it owns
    // which mesh the run starts on -----------------------------------------
    let vented = match lowered.as_ref().and_then(|l| l.vent.as_ref()) {
        Some(spec) => {
            if !dc.transient {
                return Err(refuse(
                    "vent",
                    "a steady run has no opening time: the vent needs a transient ddt, \
                     Euler or backward (SPEC-LIT §158.8)",
                ));
            }
            let raw = common::case_raw_mesh(&o.case_path, lowered.as_ref())?;
            let split = vent::split_raw(&raw, vent::select_faces(&raw, spec)?)?;
            let hm_split = ofgpu::io::polymesh::build_host_mesh(&split.raw)?;
            let closed = vent::closed_lowered(&inlets, spec.wall)?;
            println!(
                "vent: {} membrane face(s), area {} m^2, pStat {} Pa, wall {}; closed, \
                 the run starts on the split mesh ({} + {} boundary faces), SPEC-LIT §158.8",
                split.faces.len(),
                g(split.area),
                g(spec.p_stat),
                if spec.wall == vent::MembraneWall::NoSlip { "noSlip" } else { "slip" },
                split.n_boundary_merged,
                2 * split.faces.len(),
            );
            Some(VentRun { spec: spec.clone(), split, hm_split, closed })
        }
        None => None,
    };

    // SPEC-LIT §127.1.2: SA reached the solver on the OpenFOAM route (the
    // refusal above); read its fields and controls now, and attach the model
    // once the solver exists, before `initialise`.
    let sa_parts =
        match select_turbulence_model(&cc) {
            Ok(sel) if sel.model == RasModel::SpalartAllmaras
                && sel.les.is_none()
                && sel.des.is_none() =>
            {
                Some(sa_fields(&o.case_path, &cc, &dc, &hm)?)
            }
            _ => None,
        };

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
            // SPEC-LIT §158.6: an output block may name the four dust
            // fields only when the case has a dust block - the list below
            // is the check's whole vocabulary.
            plan.check_fields(&available_fields(has_dust))?;
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
    // The run's first mesh: the split one when there is a vent, else the
    // case's own - which `mesh` already is.
    let split_mesh = match &vented {
        Some(v) => Some(GpuMesh::upload(&gpu, &v.hm_split)?),
        None => None,
    };
    // `vent::setup_state` and `vent::build_solver` make exactly the calls
    // this section made before DST-09, in the same order - on the split mesh
    // with the closed-stage fields when there is a vent, else on the case's
    // own mesh with its inlets.
    let ctrl = dc.ctrl;
    let (mut solver, mut s) = match (&vented, &split_mesh) {
        (Some(v), Some(gm)) => {
            let s = vent::setup_state(&gpu, gm, &v.hm_split, &v.closed)?;
            let solver = vent::build_solver(&gpu, gm, ctrl, &v.closed)?;
            (solver, s)
        }
        _ => {
            let s = vent::setup_state(&gpu, &mesh, &hm, &inlets)?;
            let solver = vent::build_solver(&gpu, &mesh, ctrl, &inlets)?;
            (solver, s)
        }
    };
    // ---- SPEC-LIT §158.6: the dust case's coupling, built on the run's
    // own mesh and attached BEFORE `initialise`, whose gas state is then
    // the mixture law (152.8)-(152.9) from the first outer iteration on.
    let mut dust_attached = false;
    if let Some(d) = lowered.as_ref().and_then(|l| l.dust.as_ref()) {
        // The initial thermodynamic pressure (155.10)'s datum: p_op plus
        // the uniform initial gauge p (SPEC-LIT §122.2).
        let p_gauge = fields.p.internal.first().copied().unwrap_or(0.0 as Scalar);
        if fields.p.internal.iter().any(|&v| v != p_gauge) {
            return Err(refuse(
                "initial.p",
                "not uniform: the dust model's datum p_init = pOperating + p is one \
                 number (SPEC-LIT §158.6)",
            ));
        }
        let kc = gas.constants();
        let t_u = d.t_u;
        let k_mol = f64::from(kc.kappa(t_u as Scalar));
        let setup = DustSetup::new(d, gas.p_op + f64::from(p_gauge), k_mol)?;
        let nu_ref = f64::from(kc.mu(t_u as Scalar)) / setup.rho_u0;
        let num = CaseNumerics::read(&o.case_path, &cc, lowered.as_ref())?;
        let c_ctrl = dust_c_controls(&num, &cc.turb, dc.ctrl.momentum.delta_t)?;
        let raw_c = d.c_field.to_raw(hm.n_cells);
        // §158.6: the rho-weighted SST is the one model the coupling
        // carries; `refuse_out_of_scope` let it through for a dust case
        // only, so a selection here is kOmegaSST's or laminar's.
        let selection = select_turbulence_model(&cc)?;
        let turb: Option<Box<dyn CoupledTurbulence>> =
            if selection.model == RasModel::KOmegaSST
                && selection.les.is_none()
                && selection.des.is_none()
            {
                let lc = lowered.as_ref().ok_or_else(|| {
                    refuse("physics.turbulence", "not a JSONC case (SPEC-LIT §158.6)")
                })?;
                let omega_raw = lc.omega_field.as_ref().ok_or_else(|| {
                    refuse(
                        "physics.turbulence",
                        "kOmegaSST names no omega field: the rho-weighted form's \
                         closures read the case's k and omega (SPEC-LIT §158.6)",
                    )
                })?;
                let omega_raw = omega_raw.to_raw(hm.n_cells);
                let nut_raw = lc.nut_field.as_ref().map(|f| f.to_raw(hm.n_cells));
                // As `ofgpu-lowmach` builds its model: the wall faces from
                // omega's and nut's own patch types, the roughness from
                // nut's, the case's k/omega/nut seeded, then the first
                // nu_t from the initial flow.
                let wall_faces = WallFaces::from_case(&omega_raw, nut_raw.as_ref(), &hm)?;
                let roughness = NutRoughness::from_case(nut_raw.as_ref(), &hm)?;
                let mut t = build_coupled(&gpu, &hm, &mesh, &cc, &selection, &wall_faces, &roughness)?;
                for (fname, field) in t.output_fields_mut() {
                    let raw = match fname {
                        "k" => lc.k_field.as_ref().map(|f| f.to_raw(hm.n_cells)),
                        "omega" => Some(omega_raw.clone()),
                        "nut" => nut_raw.clone(),
                        _ => None,
                    };
                    if let Some(raw) = raw {
                        setup_scalar_field(&gpu, field, &raw, &hm)?;
                    }
                }
                if let Some(sst) = &d.sst {
                    t.set_rho_options(RhoSstOptions {
                        limit_omega_production: sst.limit_omega_production,
                        sust: sst.sust.map(|(k, w)| SstSust {
                            k_amb: k as Scalar,
                            omega_amb: w as Scalar,
                        }),
                    })?;
                }
                t.initialise(&gpu, &FlowState::new(&s.u, &s.mdot, nu_ref as Scalar))?;
                Some(t)
            } else {
                None
            };
        let dust = CompressibleDust::new(&gpu, &hm, &mesh, setup, &raw_c, c_ctrl, turb, nu_ref)?;
        solver.attach_dust(&gpu, dust)?;
        dust_attached = true;
    }
    // SPEC-LIT §127.1.2: SA reaches here on the OpenFOAM route only, and the
    // vent (§158.8) and the dust model (§158.6) on the JSONC route only, so
    // the model is attached to a solver on the case's own mesh.
    if let Some((raw_nt, raw_nut, sa_ctrl)) = sa_parts {
        // SPEC-LIT §6.6: the wall distance, once, at setup; nut's own patch
        // types are its wall treatment (§15.5), and the conservative form
        // takes the case's coefficients (`RAS { ... }`, the defaults absent).
        let wd = walldistance::wall_distance(&gpu, &hm, &mesh, &cc.p_solver, cc.turb.n_non_orth_correctors)?;
        let wf = WallFaces::from_case(&raw_nt, Some(&raw_nut), &hm)?;
        let rough = NutRoughness::from_case(Some(&raw_nut), &hm)?;
        let mut model = SpalartAllmaras::new(
            &gpu,
            &hm,
            &mesh,
            sa_coeffs(&cc)?,
            sa_ctrl,
            wall_coeffs_from_case(&cc.wall),
            &wf,
            &rough,
            &wd.y.f,
        )?;
        setup_scalar_field(&gpu, model.nu_tilda_mut(), &raw_nt, &hm)?;
        setup_scalar_field(&gpu, model.nut_mut(), &raw_nut, &hm)?;
        println!(
            "turbulence: SpalartAllmaras, the conservative rho-weighted form of SPEC-LIT \
             §127.1 on the mass flux (mu_eff = mu + rho nu_t); {} wall faces carry the \
             wall distance",
            wd.n_wall_faces
        );
        fields.nu_tilda = Some(raw_nt);
        fields.nut = Some(raw_nut);
        solver.set_turbulence(&gpu, model)?;
    }
    solver.initialise(&gpu, &mut s)?;

    // SPEC-LIT §158.6: the run's one dust line - what the coupling runs
    // with - then the (155.9) check of the fixed step against the
    // PREDICTED number of the initial field.
    let mut last_cof: Option<f64> = None;
    if dust_attached {
        let d = solver.dust().expect("dust");
        let (st, hc) = (d.setup(), d.hconst());
        let (t_v, p_ratio) = hc.aicc(st.t_u);
        println!(
            "dust (SPEC-LIT §158.6): model {:?}, thermo {:?}, T_u {} K, hConst cp_u {} cp_b {} \
             J/(kg K), delta_h_c {} J/kg, isobaric T_b {} K, AICC T_v {} K (p_max/p_0 {}), \
             S_L {} m/s, sigma {}, max flame Courant {}",
            st.fsc.model,
            st.gas.mode,
            g(st.t_u),
            g(hc.cp_u),
            g(hc.cp_b),
            sci(hc.delta_h_c(), 4),
            g(hc.isobaric_tb(st.t_u)),
            g(t_v),
            sci(p_ratio, 4),
            g(st.s_l),
            g(st.sigma),
            g(st.max_flame_courant),
        );
    }

    // SPEC-LIT §158.1: the trigger and the membrane exist only with a vent.
    let mut membrane = vented
        .as_ref()
        .map(|v| vent::VentTrigger::new(&gpu, &v.split).map(|t| (t, vent::Membrane::new(v.spec.p_stat))))
        .transpose()?;

    // The opening check also runs at n = 0, straight after `initialise` - a
    // membrane whose initial state already meets pStat opens before any step.
    let mut last_dp = 0.0;
    if let (Some((trigger, mem)), Some(v)) = (membrane.as_mut(), vented.as_ref()) {
        let (handed, r) =
            check_membrane(&gpu, v, trigger, mem, 0, 0.0, &solver, &s, &mesh, &hm, ctrl, &inlets)?;
        last_dp = r.dp;
        if let Some((ns, nst)) = handed {
            solver = ns;
            s = nst;
        }
    }

    let dt = f64::from(dc.ctrl.momentum.delta_t);
    if dust_attached {
        let co = solver
            .dust_mut()
            .expect("dust")
            .predicted_courant(&gpu, &hm, &s.u, dt)?;
        println!(
            "flame Courant (155.9): predicted Co_f {} against max {} on the initial field",
            g(co),
            g(solver.dust().expect("dust").setup().max_flame_courant)
        );
    }
    let mut end = RunEnd { steps: 0, converged: false, last: None, seconds: 0.0, vent: None };
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
            // SPEC-LIT §158.6, (155.9): the MEASURED number of the step
            // just taken, and the last c~ solve's out-of-bounds count.
            if dust_attached {
                let co =
                    solver.dust_mut().expect("dust").measured_courant(&gpu, &mesh, &s, dt)?;
                let oob = solver
                    .dust()
                    .expect("dust")
                    .last_solve()
                    .map(|sv| sv.out_of_bounds)
                    .unwrap_or(0);
                println!("flame Courant (155.9): measured Co_f max {}, c~ bounds: {}", g(co), oob);
                last_cof = Some(co);
            }
        }
        // SPEC-LIT §158.1: while the membrane is closed, its check runs
        // after every step n (t = n dt); the hand-over puts the run on the
        // case's own mesh from the next step on. A check at the last step
        // that opens still hands over - it is cheap, and the membrane's
        // state is what the run reports. Once open, nothing is read.
        if let (Some((trigger, mem)), Some(v)) = (membrane.as_mut(), vented.as_ref()) {
            if !mem.is_open() {
                let (handed, r) = check_membrane(
                    &gpu, v, trigger, mem, step, step as f64 * dt, &solver, &s, &mesh, &hm, ctrl,
                    &inlets,
                )?;
                last_dp = r.dp;
                if let Some((ns, nst)) = handed {
                    solver = ns;
                    s = nst;
                }
            }
        }
        let clock = if dc.transient { step as f64 * dt } else { step as f64 };
        if o.do_write
            && !last_step
            && write_time(
                &gpu, &solver, &s, &gas, &hm, &fields, clock, &mut pipeline, false, volume.as_ref(),
                solver.dust(), last_cof,
            )?
        {
            println!("    written at {}", g(clock));
        }
        if last_step {
            if o.do_write {
                write_time(
                    &gpu, &solver, &s, &gas, &hm, &fields, clock, &mut pipeline, true, volume.as_ref(),
                    solver.dust(), last_cof,
                )?;
                println!("    final state written at {} to {}", g(clock), out_root.display());
            }
            break;
        }
    }
    end.seconds = t0.elapsed().as_secs_f64();
    if let (Some((_, mem)), Some(v)) = (&membrane, vented.as_ref()) {
        end.vent = Some(mem.state);
        if !mem.is_open() {
            println!(
                "vent: closed for the whole run; the last dp = {} Pa < pStat {} Pa",
                g(last_dp),
                g(v.spec.p_stat)
            );
        }
    }
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
    // `load_case` is the other drivers' route: the tests below use it to
    // show a vent case is refused there (SPEC-LIT §158.8).
    use common::load_case;

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
                &[], &["physics.turbulence", "SpalartAllmaras", "nuTilda"]),
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

    /// The tube's one `vent` line, inserted before its `numerics` block.
    fn add_vent(case: &Path, p_stat: &str) {
        edit(
            case,
            "  \"numerics\": {",
            &format!(
                "  \"vent\": {{ \"normal\": [1, 0, 0], \"centre\": [0.5, 0.01, 0.01], \
                 \"radius\": 1.0, \"pStat\": {p_stat} }},\n  \"numerics\": {{"
            ),
        );
    }

    /// SPEC-LIT §158.8 (DST-09): the vented tube starts on the split mesh,
    /// the membrane opens once when its pressure difference reaches pStat,
    /// and the run reports the opening's step and reading.
    #[test]
    fn a_vented_tube_opens_once_and_reports() {
        if Gpu::new(0).is_err() {
            return;
        }
        let (_dir, case) = tube("dst09_open");
        edit(&case, r#""endTime": 0.00003"#, r#""endTime": 0.0002"#);
        add_vent(&case, "5000");
        let o = parse(&argv(&[case.to_string_lossy().as_ref()])).unwrap();
        let end = run(&o).expect("the vented tube runs");
        println!("dst09: the membrane ended {:?}", end.vent);
        assert_eq!(end.steps, 20);
        match end.vent {
            Some(vent::VentState::Open { step, dp, .. }) => {
                assert!((1..=20).contains(&step), "opened at step {step}");
                assert!(dp >= 5000.0, "dp {dp} Pa");
            }
            other => panic!("the membrane opened: {other:?}"),
        }
        assert_eq!(end.last.expect("an outer report").bounds.total(), 0);
    }

    /// The same vent at a pStat nothing in the run reaches: three steps, the
    /// membrane still closed, and the run says so.
    #[test]
    fn a_closed_vent_never_opens() {
        if Gpu::new(0).is_err() {
            return;
        }
        let (_dir, case) = tube("dst09_closed");
        add_vent(&case, "1e9");
        let o = parse(&argv(&[case.to_string_lossy().as_ref()])).unwrap();
        let end = run(&o).expect("the closed-vent tube runs");
        assert_eq!(end.steps, 3);
        assert_eq!(end.vent, Some(vent::VentState::Closed));
    }

    /// SPEC-LIT §158.8: a steady run has no opening time, so the driver
    /// refuses the vent; every other driver refuses the case outright.
    #[test]
    fn a_vent_is_refused_where_it_cannot_run() {
        if Gpu::new(0).is_err() {
            return;
        }
        let (_dir, case) = tube("dst09_steady");
        add_vent(&case, "5000");
        edit(&case, r#""ddt": "Euler""#, r#""ddt": "steadyState""#);
        // A steady tube is a SIMPLE one - otherwise the algorithm check
        // answers first and the vent never does.
        edit(&case, r#""kind": "PIMPLE""#, r#""kind": "SIMPLE""#);
        // ...and a steady run takes -iters, not endTime/deltaT.
        let e = refusal(&case, &["-iters", "3"]);
        assert!(e.contains("vent") && e.contains("§158.8"), "{e}");
        let e = common::load_case(&case).err().expect("the vent case is refused");
        assert!(e.to_string().contains("ofgpu-compressible"), "{e}");
    }

    // ==========================================================================
    //  DST-10's driver gates (SPEC-LIT §158.6)
    // ==========================================================================

    /// The sealed dust box: `lowmach_dust::tests`'s 2x2x2 cube of 1 cm
    /// sides, every patch an adiabatic wall, at rest at T 273, with the air
    /// gas block of SPEC-LIT §122.6 about 101325 Pa, PIMPLE 5 outer / 2
    /// correctors, Euler, and that file's TFC pair. `sst` swaps the
    /// kOmegaSST block and its k/omega entries in, and states `dust.sst`
    /// beside the model.
    fn dust_box_text(sst: bool) -> String {
        let text = r#"
{
  "name": "dustBox",
  "mesh": {
    "kind": "cartesian",
    "bounds": { "min": [0, 0, 0], "max": [0.01, 0.01, 0.01] },
    "cells": [2, 2, 2],
    "boundaries": { "xmin": "wallX0", "xmax": "wallX1", "ymin": "wallY0",
      "ymax": "wallY1", "zmin": "wallZ0", "zmax": "wallZ1" }
  },
  "physics": {
    "gravity": [0, 0, 0],
    "fluid": { "nu": 1.5e-5, "Pr": 0.71, "Prt": 0.85, "TRef": 293.15 },
    "buoyancy": "densityRatio",
    "compressible": { "gamma": 1.4, "W": 0.0289647, "Pr": 0.71,
      "viscosity": { "model": "sutherland", "As": 1.458e-6, "Ts": 110.4 },
      "pOperating": 101325 }
  }__TURB__,
  "patches": [
    {
      "match": ".*", "kind": "wall",
      "U": { "type": "fixedValue", "value": [0, 0, 0] },
      "p": { "type": "zeroGradient" },
      "T": { "type": "zeroGradient" }__KPATCH__
    }
  ],
  "initial": { "U": [0, 0, 0], "T": 273, "p": 0.0__KINIT__ },
  "numerics": {
    "algorithm": { "kind": "PIMPLE", "correctors": 2, "outerCorrectors": 5 },
    "ddt": "Euler",
    "div": { "default": "Gauss upwind" },
    "grad": "Gauss linear",
    "laplacian": { "snGrad": "corrected", "nonOrthogonalCorrectors": 0 },
    "relaxation": { "U": 1.0, "p": 1.0, "T": 1.0 },
    "solvers": [
      { "match": "p", "solver": "PBiCGStab", "preconditioner": "DILU", "tolerance": 1e-13, "relTol": 0.0, "maxIter": 2000 },
      { "match": "U", "solver": "PBiCGStab", "preconditioner": "diagonal", "tolerance": 1e-13, "relTol": 0.0, "maxIter": 2000 },
      { "match": "T", "solver": "PBiCGStab", "preconditioner": "diagonal", "tolerance": 1e-13, "relTol": 0.0, "maxIter": 2000 },
      { "match": "c", "solver": "PBiCGStab", "preconditioner": "diagonal", "tolerance": 1e-13, "relTol": 0.0, "maxIter": 2000 }
    ]
  },
  "run": { "endTime": 0.02, "deltaT": 1e-4 },
  "combustion": { "model": "TFC", "PrT": 0.28,
    "ignition": { "point": [0.005, 0.005, 0.005], "sigmaR": 1000, "W0": 1000, "t0": 1e-4 } },
  "dust": { "thermo": "formula", "phi": 0.62__DUSTSST__ }
}
"#;
        dust_box_variants(sst, text)
    }

    /// The four `dust_box_text` substitutions: the kOmegaSST block and its
    /// k/omega entries when `sst`, the laminar box's empty ones otherwise.
    fn dust_box_variants(sst: bool, text: &str) -> String {
        let (turb, kpatch, kinit, dustsst) = if sst {
            (
                r#",
  "turbulence": { "kind": "RAS", "model": "kOmegaSST", "wallFunctions": { "kappa": 0.41, "E": 9.8 } }"#,
                r#",
      "k": { "type": "kqRWallFunction", "value": 3.94 },
      "omega": { "type": "omegaWallFunction", "value": 1101.4 }"#,
                r#", "k": 3.94, "omega": 1101.4"#,
                r#", "sst": { "sust": { "kAmb": 3.94, "omegaAmb": 1101.4 } }"#,
            )
        } else {
            ("", "", "", "")
        };
        text.replace("__TURB__", turb)
            .replace("__KPATCH__", kpatch)
            .replace("__KINIT__", kinit)
            .replace("__DUSTSST__", dustsst)
    }

    /// A fresh dust box in its own scratch directory: `(dir, case file)`.
    fn dust_box(tag: &str, sst: bool) -> (PathBuf, PathBuf) {
        let dir = scratch_dir(tag);
        let case = dir.join("case.jsonc");
        std::fs::write(&case, dust_box_text(sst)).expect("write the dust box case");
        (dir, case)
    }

    /// The one time directory a run wrote (its final state).
    fn final_time_dir(case: &Path) -> PathBuf {
        let root = common::json_case_output_dir(case);
        let last = std::fs::read_dir(&root)
            .expect("the case is written")
            .filter_map(|e| e.ok()?.file_name().into_string().ok())
            .filter(|n| n.parse::<f64>().is_ok_and(|t| t > 0.0))
            .max_by(|a, b| a.parse::<f64>().unwrap().total_cmp(&b.parse::<f64>().unwrap()))
            .expect("a written time");
        root.join(last)
    }

    /// **SPEC-LIT §158.6 (DST-10):** the sealed dust box runs 20 steps of
    /// 1e-4 s, the write carries the four dust fields beside `rho`, `T` and
    /// `p`, and the volume-mean `p` in the final time directory is above
    /// the initial one - the box burns on the (158.4) heat release.
    #[test]
    fn a_dust_box_runs_and_writes_c() {
        if Gpu::new(0).is_err() {
            return;
        }
        let (_dir, case) = dust_box("dst10_box", false);
        let o = parse(&argv(&[case.to_string_lossy().as_ref(), "-iters", "20"])).unwrap();
        let end = run(&o).expect("the dust box runs");
        assert_eq!(end.steps, 20);
        let out = final_time_dir(&case);
        for name in ["c", "Ut", "Cof", "HRR", "rho", "T", "p"] {
            let text = std::fs::read_to_string(out.join(name))
                .unwrap_or_else(|e| panic!("{name} was not written: {e}"));
            assert!(text.contains("object"), "{name} is not a field file");
        }
        let p = read_scalar_field(&out.join("p"), 8).expect("the written p reads");
        let mean: f64 =
            p.internal.iter().map(|v| f64::from(*v)).sum::<f64>() / p.internal.len() as f64;
        println!("dst10 box: 20 steps, volume-mean gauge p {}", g(mean));
        assert!(mean > 0.0, "the box did not burn: volume-mean gauge p {mean}");
        let c = read_scalar_field(&out.join("c"), 8).expect("the written c reads");
        let c_mean: f64 = c.internal.iter().map(|v| f64::from(*v)).sum::<f64>() / 8.0;
        println!("dst10 box: c~ mean {c_mean}");
    }

    /// **SPEC-LIT §158.6:** the same box as kOmegaSST (the ρ-weighted form
    /// of §156, the one model the coupling carries) runs 5 steps at a small
    /// dt - the model stays frozen at its initial k/omega before 2 t_0, as
    /// §155.3's H20 rule reads - and every written field is finite.
    #[test]
    fn a_dust_sst_box_runs() {
        if Gpu::new(0).is_err() {
            return;
        }
        let (_dir, case) = dust_box("dst10_sst_box", true);
        let o = parse(&argv(&[case.to_string_lossy().as_ref(), "-iters", "5"])).unwrap();
        let end = run(&o).expect("the SST dust box runs");
        assert_eq!(end.steps, 5);
        let out = final_time_dir(&case);
        for name in ["U", "p", "T", "rho", "Mach", "c", "Ut", "Cof", "HRR"] {
            if name == "U" {
                let u = read_vector_field(&out.join(name), 8)
                    .unwrap_or_else(|e| panic!("{name} was not written: {e}"));
                assert!(
                    u.internal.iter().all(|v| v.x.is_finite() && v.y.is_finite() && v.z.is_finite()),
                    "{name} holds a non-finite value"
                );
                continue;
            }
            let v = read_scalar_field(&out.join(name), 8)
                .unwrap_or_else(|e| panic!("{name} was not written: {e}"));
            assert!(v.internal.iter().all(|x| x.is_finite()), "{name} holds a non-finite value");
        }
        println!("dst10 SST box: 5 steps in {:.2} s, all written fields finite", end.seconds);
    }

    /// **SPEC-LIT §158.6:** every run a dust case cannot make is refused
    /// with its key in the message - a steady or non-Euler ddt, a vent
    /// beside the model, kOmegaSST without a dust block, a compressible
    /// preset patch, a step above the flame-Courant bound, and the
    /// low-Mach loader's refusal of the gas beside the model.
    #[test]
    fn dust_refusals_name_the_setting() {
        if Gpu::new(0).is_err() {
            return;
        }
        let _g = ofgpu::io::contract::permissive_test_guard();
        ofgpu::io::contract::set_permissive(false);
        // A steady ddt: the c~ equation is transient.
        let (_dir, case) = dust_box("dst10_ref_steady", false);
        edit(&case, r#""ddt": "Euler""#, r#""ddt": "steadyState""#);
        edit(&case, r#""kind": "PIMPLE""#, r#""kind": "SIMPLE""#);
        let e = refusal(&case, &["-iters", "2"]);
        assert!(e.contains("numerics.ddt"), "{e}");
        // backward: the mixture energy equation rides one Euler step.
        let (_dir, case) = dust_box("dst10_ref_backward", false);
        edit(&case, r#""ddt": "Euler""#, r#""ddt": "backward""#);
        let e = refusal(&case, &[]);
        assert!(e.contains("numerics.ddt"), "{e}");
        // A vent block beside the dust model: the opening hand-over is
        // DST-11's.
        let (_dir, case) = dust_box("dst10_ref_vent", false);
        add_vent(&case, "5000");
        let e = refusal(&case, &[]);
        assert!(e.contains("vent"), "{e}");
        // kOmegaSST without a dust block: the §158.6 lift is the model's
        // only route in.
        let (_dir, case) = tube("dst10_ref_sst");
        edit(
            &case,
            r#""initial":"#,
            r#""turbulence": {"kind":"RAS","model":"kOmegaSST","wallFunctions":{"kappa":0.41,"E":9.8}}, "initial":"#,
        );
        let e = refusal(&case, &[]);
        assert!(e.contains("kOmegaSST") && e.contains("physics.turbulence"), "{e}");
        // A compressible preset patch beside the dust model: the lowering
        // refuses it, naming the preset rule (SPEC-LIT §158.6).
        let (_dir, case) = dust_box("dst10_ref_farfield", false);
        edit(
            &case,
            r#"  "patches": [
    {
      "match": ".*", "kind": "wall","#,
            r#"  "patches": [
    { "match": ".*", "kind": "farfield", "freestream": { "U": [0, 0, 0], "p": 0, "T": 273 } },
    {
      "match": ".*", "kind": "wall","#,
        );
        let e = refusal(&case, &[]);
        assert!(e.contains("dust"), "{e}");
        // -iters 1 at a dt whose PREDICTED Co_f exceeds the bound (155.9).
        // The prediction is formed from the initial field's own |u| (the
        // closures' U_t,ISP is zero at the box's zero turbulence), so the
        // step that cannot be taken is the one whose flow Courant alone
        // crosses the bar.
        let (_dir, case) = dust_box("dst10_ref_courant", false);
        edit(&case, r#""U": [0, 0, 0], "T": 273"#, r#""U": [50, 0, 0], "T": 273"#);
        edit(&case, r#""deltaT": 1e-4"#, r#""deltaT": 1e-2"#);
        let e = refusal(&case, &["-iters", "1"]);
        assert!(e.contains("maxFlameCourant"), "{e}");
        // The low-Mach loader refuses the gas block beside the model.
        let (_dir, case) = dust_box("dst10_ref_loader", false);
        let e = common::load_case_with(&case, true).err().expect("the gas block is refused");
        assert!(e.to_string().contains("physics.compressible"), "{e}");
    }

    /// Turn a laminar cavity into the SA case of SPEC-LIT §127.1.2: the RAS
    /// block names `SpalartAllmaras`, the `bounded` prefix is off
    /// `div(phi,nuTilda)`, a `0/nuTilda` at rest sits beside `0/nut`, and
    /// every `nutkWallFunction` becomes the one wall type SA has a model for
    /// (SPEC-LIT §56.7).
    fn sa_cavity(tag: &str) -> PathBuf {
        let dir = openfoam_cavity(tag);
        edit(&dir.join("constant/momentumTransport"), "simulationType  laminar;", "simulationType  RAS;");
        edit(
            &dir.join("constant/momentumTransport"),
            "model           kEpsilon;",
            "model           SpalartAllmaras;",
        );
        edit(
            &dir.join("system/fvSchemes"),
            "div(phi,nuTilda) bounded Gauss upwind;",
            "div(phi,nuTilda) Gauss upwind;",
        );
        // `0/nuTilda`, walls pinned at 0 (the exact Dirichlet §56.7 gives
        // nu~), a freestream-sized seed in the cells - nu~ = 0 is the
        // equation's exact fixed point, and a case seeded there stays there.
        let mut nt = read_scalar_field(&dir.join("0").join("nut"), 64).expect("read the cavity nut");
        nt.name = "nuTilda".to_string();
        nt.dimensions = "[0 2 -1 0 0 0 0]".to_string();
        nt.internal = vec![1.0e-4 as Scalar; 64];
        for spec in nt.boundary.values_mut() {
            if spec.type_name == "nutkWallFunction" {
                spec.type_name = "fixedValue".to_string();
                spec.value = vec![0.0 as Scalar];
            }
        }
        ofgpu::io::fields::write_scalar_field(&dir.join("0").join("nuTilda"), &nt, "0")
            .expect("write nuTilda");
        let nut = dir.join("0").join("nut");
        let s = std::fs::read_to_string(&nut).expect("read nut");
        std::fs::write(&nut, s.replace("nutkWallFunction", "nutLowReWallFunction")).expect("write nut");
        dir
    }

    /// CMP-25 through the driver (SPEC-LIT §127.1.2): the SA cavity runs all
    /// five steps with no bounds violation and writes `nuTilda` and `nut` -
    /// finite, `nut >= 0` - and the same case with `nut` left at
    /// `nutkWallFunction` is refused by name.
    #[test]
    fn an_openfoam_sa_cavity_runs_and_writes_nu_tilda() {
        if Gpu::new(0).is_err() {
            return;
        }
        let dir = sa_cavity("cmp25_sa");
        let o = parse(&argv(&[dir.to_string_lossy().as_ref()])).unwrap();
        let end = run(&o).expect("the SA cavity runs");
        assert_eq!(end.steps, 5);
        let last = end.last.expect("an outer report");
        assert_eq!(last.bounds.total(), 0);
        assert!(last.nu_tilda.is_some(), "the nu~ solve is not reported");

        for name in ["nuTilda", "nut"] {
            let f = read_scalar_field(&dir.join("0.005").join(name), 64).expect("parse");
            assert!(f.internal.iter().all(|x| f64::from(*x).is_finite()), "{name} is not finite");
            assert!(f.internal.iter().all(|x| f64::from(*x) >= 0.0), "{name} went negative");
            let text = std::fs::read_to_string(dir.join("0.005").join(name)).expect("written");
            assert!(text.contains(&format!("object      {name};")), "{name}: {text}");
        }
        let nut = read_scalar_field(&dir.join("0.005").join("nut"), 64).expect("parse nut");
        assert!(nut.internal.iter().any(|x| f64::from(*x) > 0.0), "nut never left zero");

        // The same case with `nut` still asking for the k-based wall
        // function: refused by name - SA has no k (SPEC-LIT §56.7).
        let dir = sa_cavity("cmp25_sa_nutk");
        let nut = dir.join("0").join("nut");
        let s = std::fs::read_to_string(&nut).expect("read nut");
        std::fs::write(&nut, s.replace("nutLowReWallFunction", "nutkWallFunction")).expect("write nut");
        let e = refusal(&dir, &[]);
        for w in ["nut patch", "nutkWallFunction"] {
            assert!(e.contains(w), "{w:?} not in {e}");
        }
    }

    // ==========================================================================
    //  CMP-28: the two shipped compressible recipes load and run as written
    // ==========================================================================

    /// Copy a shipped case into `tag`'s scratch directory - never write into
    /// `cases/` - and answer where it landed.
    fn shipped(tag: &str, from: &str) -> PathBuf {
        let dir = scratch_dir(tag);
        let case = dir.join(from);
        std::fs::copy(
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../cases").join(from),
            &case,
        )
        .unwrap_or_else(|e| panic!("copy cases/{from}: {e}"));
        case
    }

    /// Copy a case DIRECTORY recursively (`cases/nozzle`), into `tag`'s
    /// scratch directory, and answer the copy's root.
    fn shipped_dir(tag: &str, from: &str) -> PathBuf {
        fn rec(src: &Path, dst: &Path) {
            std::fs::create_dir_all(dst).expect("create the copy's directory");
            for entry in std::fs::read_dir(src).expect("read the source directory") {
                let e = entry.expect("directory entry");
                let s = e.path();
                let d = dst.join(e.file_name());
                if e.file_type().expect("file type").is_dir() {
                    rec(&s, &d);
                } else {
                    std::fs::copy(&s, &d).expect("copy one file");
                }
            }
        }
        let dir = scratch_dir(tag).join(from);
        let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("../cases").join(from);
        rec(&src, &dir);
        dir
    }

    /// The newest numeric time directory under a run's output root.
    fn last_time(root: &Path) -> String {
        std::fs::read_dir(root)
            .expect("the run wrote its output")
            .filter_map(|e| e.ok()?.file_name().into_string().ok())
            .filter(|n| n.parse::<f64>().is_ok_and(|t| t > 0.0))
            .max_by(|a, b| a.parse::<f64>().unwrap().total_cmp(&b.parse::<f64>().unwrap()))
            .expect("a written time")
    }

    /// CMP-28: the shipped Sod case - the JSONC `initial.regions` statement
    /// of the split state - loads and runs as written: 176 steps, no bound,
    /// the five fields written at the last time, and a `p` that is no longer
    /// the uniform start.
    #[test]
    fn the_shipped_sod_case_runs() {
        if Gpu::new(0).is_err() {
            return;
        }
        let case = shipped("cmp28_sod_run", "sodShockTube.jsonc");
        let dir = case.parent().expect("a scratch directory").to_path_buf();
        let o = parse(&argv(&[case.to_string_lossy().as_ref(), "-reportEvery", "50"]))
            .expect("the shipped case's command line parses");
        let end = run(&o).expect("the shipped Sod case runs");
        assert_eq!(end.steps, 176, "endTime/deltaT is 176 steps");
        assert_eq!(end.last.expect("an outer report").bounds.total(), 0, "no bound");
        let last = dir.join("sodShockTube_jsonc").join(last_time(&dir.join("sodShockTube_jsonc")));
        assert_eq!(five_fields(&last).len(), 5, "the five fields are written");
        let p = read_scalar_field(&last.join("p"), 200).expect("p reads back");
        let first = f64::from(p.internal[0]);
        assert!(
            p.internal.iter().any(|x| (f64::from(*x) - first).abs() > 1e-6),
            "the written p at the last time is uniform: the run solved nothing"
        );
    }

    /// CMP-28: the shipped Sod case states Gate CMP-SOD's PIMPLE leg - the
    /// coupling that meets the gate's bars (SPEC-LIT 127.6.1) - through
    /// [`driver_controls`], so the case file and the gate cannot drift.
    #[test]
    fn the_shipped_sod_case_states_the_gate_pimple_controls() {
        let case = shipped("cmp28_sod_ctrl", "sodShockTube.jsonc");
        let (hm, cc, l) = load_case(&case).expect("the shipped case loads");
        let (_, gas) = load_fields_and_gas(&case, &hm, l.as_ref()).expect("the fields load");
        let dc = driver_controls(&case, &cc, l.as_ref(), gas, None).expect("the controls");
        assert_eq!(dc.ctrl.n_outer, 5, "PIMPLE's five outer iterations");
        assert_eq!(dc.ctrl.n_corr, 2, "PIMPLE's two correctors");
        assert_eq!(
            (
                f64::from(dc.ctrl.p_relax),
                f64::from(dc.ctrl.momentum.u_relax),
                f64::from(dc.ctrl.energy.t_relax)
            ),
            (
                f64::from(1.0 as Scalar),
                f64::from(1.0 as Scalar),
                f64::from(1.0 as Scalar)
            ),
            "p unrelaxed, U and T unrelaxed"
        );
        assert!(dc.transient, "Euler is a transient run");
        assert_eq!(dc.n_steps, 176);
        assert_eq!(f64::from(dc.ctrl.momentum.delta_t), f64::from(3.593497341100431e-6 as Scalar));
        assert_eq!(dc.ctrl.momentum.ddt, DdtScheme::Euler);
        assert_eq!(dc.ctrl.energy.ddt, DdtScheme::Euler);
        assert_eq!(
            dc.ctrl.momentum.div_scheme,
            DivScheme::Limited(ofgpu::fv::Limiter::VanLeer),
            "van Leer on the momentum convection"
        );
        assert_eq!(
            dc.ctrl.energy.div_scheme,
            DivScheme::Limited(ofgpu::fv::Limiter::VanLeer),
            "van Leer on the energy convection"
        );
        assert_eq!(
            dc.ctrl.pressure.retarded,
            Retarded::Limited(ofgpu::fv::Limiter::VanLeer),
            "van Leer on the retarded density"
        );
        for (name, s) in [
            ("p", &dc.ctrl.pressure.p_solver),
            ("U", &dc.ctrl.momentum.u_solver),
            ("T", &dc.ctrl.energy.t_solver),
        ] {
            assert_eq!(
                (s.solver, s.precon),
                (
                    ofgpu::io::case::LinearSolverKind::PBiCGStab,
                    ofgpu::io::case::Preconditioner::Dilu
                ),
                "{name}'s solver"
            );
            assert_eq!(
                (f64::from(s.tolerance), f64::from(s.rel_tol), s.max_iter),
                (f64::from(1e-12 as Scalar), 0.0, 200),
                "{name}'s tolerance, relTol and maxIter"
            );
        }
        assert_eq!(
            dc.ctrl.gas.viscosity,
            Viscosity::Constant { mu: 0.0 },
            "the inviscid gas"
        );
    }

    /// The duct polyMesh and `0/p`'s internal list, exactly as generated:
    /// `write_poly_mesh_raw` on `nozzle_duct(&DuctSpec::default())`, and the
    /// gate's linear start from the inlet's total pressure to the back
    /// pressure at each cell's axial midpoint (SPEC-LIT 127.6.5). With
    /// `OFGPU_WRITE_NOZZLE_CASE=1` the polyMesh is written into
    /// `cases/nozzle` itself - how the tracked files were created.
    fn generate_nozzle(into: &Path) {
        use ofgpu::compressible::gatemesh::{nozzle_duct, DuctSpec};
        let duct = nozzle_duct(&DuctSpec::default()).expect("the gate duct builds");
        ofgpu::io::polymesh::write_poly_mesh_raw(&into.join("constant/polyMesh"), &duct.raw)
            .expect("write the polyMesh");
        let n = duct.spec.n;
        let pb_pt1 = 0.95;
        let p_op = 1.0e5;
        let p = ofgpu::io::fields::RawScalarField {
            name: "p".into(),
            dimensions: "[1 -1 -2 0 0 0 0]".into(),
            internal: (0..n)
                .map(|i| {
                    (((2 * i + 1) as f64 / (2 * n) as f64) * (pb_pt1 - 1.0) * p_op) as Scalar
                })
                .collect(),
            ..ofgpu::io::fields::RawScalarField::default()
        };
        ofgpu::io::fields::write_scalar_field(&into.join("p_gauge"), &p, "0")
            .expect("write the generated p list");
    }

    /// The `internalField` block of a written `p`, the one part of `0/p`
    /// that is generated rather than stated.
    fn p_internal_block(text: &str) -> String {
        let a = text.find("internalField").expect("an internalField entry");
        let b = text.find("boundaryField").expect("a boundaryField entry");
        text[a..b].replace("\r\n", "\n").trim_end().to_string()
    }

    /// CMP-28: `cases/nozzle`'s polyMesh and `0/p`'s internal list ARE the
    /// generated ones - the case is pinned to
    /// `ofgpu::compressible::gatemesh::nozzle_duct(&DuctSpec::default())`,
    /// so a change to the builder fails here rather than silently
    /// re-meshing every clone of the case.
    #[test]
    fn the_shipped_nozzle_case_is_the_generated_one() {
        let case = Path::new(env!("CARGO_MANIFEST_DIR")).join("../cases/nozzle");
        if std::env::var_os("OFGPU_WRITE_NOZZLE_CASE").is_some() {
            // Only the polyMesh is written into the case: `0/p` carries
            // stated boundary entries, and its internal list is compared below.
            let tmp = scratch_dir("cmp28_nozzle_write");
            generate_nozzle(&tmp);
            let mesh = case.join("constant/polyMesh");
            std::fs::create_dir_all(&mesh).expect("create cases/nozzle/constant/polyMesh");
            for f in ["points", "faces", "owner", "neighbour", "boundary"] {
                std::fs::copy(tmp.join("constant/polyMesh").join(f), mesh.join(f))
                    .unwrap_or_else(|e| panic!("write cases/nozzle/constant/polyMesh/{f}: {e}"));
            }
            println!("wrote {}", mesh.display());
            return;
        }
        let dir = scratch_dir("cmp28_nozzle_pin");
        generate_nozzle(&dir);
        for f in ["points", "faces", "owner", "neighbour", "boundary"] {
            let gen = std::fs::read_to_string(dir.join("constant/polyMesh").join(f))
                .unwrap_or_else(|e| panic!("read the generated {f}: {e}"));
            let ship = std::fs::read_to_string(case.join("constant/polyMesh").join(f))
                .unwrap_or_else(|e| panic!("read cases/nozzle/constant/polyMesh/{f}: {e}"));
            assert_eq!(
                gen.replace("\r\n", "\n").trim_end(),
                ship.replace("\r\n", "\n").trim_end(),
                "cases/nozzle/constant/polyMesh/{f} is not what nozzle_duct(&DuctSpec::default()) \
                 writes; regenerate it with OFGPU_WRITE_NOZZLE_CASE=1"
            );
        }
        let gen = std::fs::read_to_string(dir.join("p_gauge")).expect("the generated p");
        let ship = std::fs::read_to_string(case.join("0").join("p")).expect("cases/nozzle/0/p");
        assert_eq!(
            p_internal_block(&gen),
            p_internal_block(&ship),
            "cases/nozzle/0/p's internal list is not the gate's linear start"
        );
    }

    /// CMP-28: `cases/nozzle` states Gate CMP-NOZ's controls (SPEC-LIT
    /// 127.6.5) through [`driver_controls`], so the case file and the gate
    /// cannot drift.
    #[test]
    fn the_shipped_nozzle_case_states_the_gate_controls() {
        let case = shipped_dir("cmp28_nozzle_ctrl", "nozzle");
        let (hm, cc, l) = load_case(&case).expect("the shipped case loads");
        let (_, gas) = load_fields_and_gas(&case, &hm, l.as_ref()).expect("the fields load");
        let dc = driver_controls(&case, &cc, l.as_ref(), gas, None).expect("the controls");
        assert!(!dc.transient, "localEuler is a steady march");
        assert_eq!(dc.n_steps, 15000, "endTime counts iterations");
        assert_eq!(f64::from(dc.ctrl.lts.cfl), f64::from(1.0 as Scalar), "maxCo 1");
        assert_eq!(dc.ctrl.momentum.ddt, DdtScheme::LocalEuler);
        assert_eq!(dc.ctrl.energy.ddt, DdtScheme::LocalEuler);
        assert_eq!(dc.ctrl.n_outer, 1, "one outer corrector");
        assert_eq!(
            (
                f64::from(dc.ctrl.momentum.u_relax),
                f64::from(dc.ctrl.energy.t_relax),
                f64::from(dc.ctrl.p_relax)
            ),
            (f64::from(0.7 as Scalar), f64::from(0.7 as Scalar), f64::from(0.3 as Scalar)),
            "the gate's measured relaxation"
        );
        assert_eq!(
            dc.ctrl.momentum.div_scheme,
            DivScheme::Limited(ofgpu::fv::Limiter::VanLeer)
        );
        assert_eq!(
            dc.ctrl.energy.div_scheme,
            DivScheme::Limited(ofgpu::fv::Limiter::VanLeer)
        );
        assert_eq!(
            dc.ctrl.pressure.retarded,
            Retarded::Limited(ofgpu::fv::Limiter::VanLeer)
        );
        let (u, p, t) = (
            &dc.ctrl.momentum.u_solver,
            &dc.ctrl.pressure.p_solver,
            &dc.ctrl.energy.t_solver,
        );
        for (name, s, tol, rel) in [
            ("U", u, 1e-9, 0.01),
            ("p", p, 1e-10, 1e-3),
            ("T", t, 1e-9, 0.01),
        ] {
            assert_eq!(
                (s.solver, s.precon),
                (
                    ofgpu::io::case::LinearSolverKind::PBiCGStab,
                    ofgpu::io::case::Preconditioner::Dilu
                ),
                "{name}'s solver"
            );
            assert_eq!(
                (f64::from(s.tolerance), f64::from(s.rel_tol), s.max_iter),
                (f64::from(tol as Scalar), f64::from(rel as Scalar), 1000),
                "{name}'s tolerance, relTol and maxIter"
            );
        }
        assert!(dc.ctrl.residual_control.is_none(), "no residualControl");
        assert_eq!(dc.ctrl.gas.viscosity, Viscosity::Constant { mu: 0.0 });
        assert_eq!(dc.ctrl.gas.p_op, 1e5, "pOperating");
    }

    /// CMP-28: the shipped nozzle recipe runs as written - 200 iterations of
    /// the 200-cell duct, no bound.
    #[test]
    fn the_shipped_nozzle_recipe_runs() {
        if Gpu::new(0).is_err() {
            return;
        }
        let dir = shipped_dir("cmp28_nozzle_run", "nozzle");
        let o = parse(&argv(&[dir.to_string_lossy().as_ref(), "-iters", "200"]))
            .expect("the recipe's command line parses");
        let end = run(&o).expect("the shipped nozzle recipe runs");
        assert_eq!(end.steps, 200, "-iters 200 is 200 iterations");
        assert_eq!(end.last.expect("an outer report").bounds.total(), 0, "no bound");
    }
}
