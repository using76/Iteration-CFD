// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

//! `ofgpu-datacentre` - the data-centre room driver, SPEC-LIT §52 to §55.
//!
//! Provenance: ORIGINAL - a driver, not numerics. Every equation it reaches is
//! specified in SPEC-LIT §52 (the fan curve as a Robin triple), §53 (the
//! porous jump), §54 (humidity and psychrometrics) and §55 (RCI, RTI,
//! SHI/RHI and the PUE inputs), and implemented in `crate::fan`,
//! `crate::psychro` and `crate::dcmetrics`; this file reads a case, runs it,
//! and reports what it did. No GPL-licensed source was consulted.
//!
//! ```text
//! ofgpu-datacentre <case.jsonc> [-json <out.json>] [-run-id <id>] [-csv <out.csv>]
//!                  [-schema] [-permissive]
//! ```
//!
//! # What it solves
//!
//! A steady buoyant room: SIMPLE momentum and pressure with §52's fan patches
//! and §53's tile patches on the pressure, a transported temperature with
//! §18 rack heat-release zones, and - where the case asks for it - §54's
//! water-vapour mass fraction feeding §54.7's virtual temperature into the
//! buoyancy.
//!
//! # What it prints, and what it refuses to print
//!
//! §55's report: `RCI_HI`, `RCI_LO` with their sample count and the ASHRAE
//! class they were measured against; `RTI` with which `dT_equipment` it used;
//! `SHI`/`RHI`; the per-fan operating points and shaft powers; and the PUE
//! **inputs**. It does **not** print a PUE (§55.4): PUE is a facility energy
//! ratio and a room model cannot compute one.
//! When the case carries `metrics.supplyTemperatureSweep`, it also runs §55.4's
//! sweep and prints the free-cooling ceiling beside the PUE inputs.
//!
//! Where the case carries a porous jump it also prints §53.6's caveat, naming
//! what a pressure-jump tile gets wrong.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use ofgpu::dcmetrics::{
    dt_equipment_from_heat, rci_hi, rci_lo, rti, shi_rhi, MetricReport, Metrics, PueInputs,
    RciSamples,
};
use ofgpu::error::Result;
use ofgpu::fan::FlowDevices;
use ofgpu::field::{BcKind, GpuScalarField};
use ofgpu::io::case_dc::{DcCase, LoweredDcCase, SupplySweep};
use ofgpu::mesh::{GpuMesh, PatchKind};
use ofgpu::models::k_epsilon::{KEpsilon, KEpsilonCoeffs};
use ofgpu::momentum::{BuoyancyCoeffs, MomentumControls};
use ofgpu::pressure::{PressureBackend, SystemProbe};
use ofgpu::psychro::Psychrometrics;
use ofgpu::scalar_transport::{ScalarTransport, ScalarTransportCoeffs};
use ofgpu::simple::{Simple, SimpleControls};
use ofgpu::sources::{CellSelector, SourceTerm};
use ofgpu::turbulence::TurbulenceControls;
use ofgpu::{Gpu, Label, Scalar, Vec3};

const USAGE: &str = "\
ofgpu-datacentre <case.jsonc> [-json <out.json>] [-run-id <id>] [-csv <out.csv>]
                 [-schema] [-permissive]

  SPEC-LIT S52 to S55: a data-centre room with fan curves, porous-jump tiles,
  humidity and the RCI/RTI/SHI metrics a customer report must contain.

  -json <path>    write the whole S55 report as one JSON document
  -run-id <id>    the run this document belongs to (default: v_<UTC stamp>)
  -csv <path>     write a final snapshot of the metrics, one row per quantity
  -schema         print the case format's JSON Schema on stdout and exit 0
  -permissive     downgrade unsupported-setting errors to warnings";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    let mut case_path: Option<PathBuf> = None;
    let mut csv: Option<PathBuf> = None;
    let mut json: Option<PathBuf> = None;
    let mut run_id: Option<String> = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "-json" => {
                i += 1;
                match args.get(i) {
                    Some(p) => json = Some(PathBuf::from(p)),
                    None => {
                        eprintln!("-json needs a path");
                        return ExitCode::FAILURE;
                    }
                }
            }
            "-run-id" => {
                i += 1;
                match args.get(i) {
                    Some(p) => run_id = Some(p.clone()),
                    None => {
                        eprintln!("-run-id needs a value");
                        return ExitCode::FAILURE;
                    }
                }
            }
            // -schema needs no case: the schema is generated from the SAME
            // types that parse one, so it exists before any file is read.
            // Every other argument on the line is ignored, the way a schema
            // request reads.
            "-schema" => {
                println!("{}", ofgpu::io::case_dc::emit_dc_schema());
                return ExitCode::SUCCESS;
            }
            "-csv" => {
                i += 1;
                match args.get(i) {
                    Some(p) => csv = Some(PathBuf::from(p)),
                    None => {
                        eprintln!("-csv needs a path");
                        return ExitCode::FAILURE;
                    }
                }
            }
            "-permissive" => ofgpu::io::contract::set_permissive(true),
            "-h" | "--help" => {
                println!("{USAGE}");
                return ExitCode::SUCCESS;
            }
            other if other.starts_with('-') => {
                eprintln!("unknown option `{other}`\n\n{USAGE}");
                return ExitCode::FAILURE;
            }
            other => case_path = Some(PathBuf::from(other)),
        }
        i += 1;
    }

    let Some(case_path) = case_path else {
        eprintln!("{USAGE}");
        return ExitCode::FAILURE;
    };

    match run(&case_path, csv.as_deref(), json.as_deref(), run_id.as_deref()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("\nofgpu-datacentre: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run(
    case_path: &Path,
    csv: Option<&Path>,
    json: Option<&Path>,
    run_id: Option<&str>,
) -> Result<()> {
    let started_ms = epoch_ms(std::time::SystemTime::now());
    let case = DcCase::read(case_path)?;
    let lowered = case.lower()?;

    println!("=== {} ===", lowered.name);
    println!(
        "room: {} cells, {} boundary faces",
        lowered.mesh.n_cells, lowered.mesh.n_boundary_faces
    );
    for n in &lowered.notes {
        println!("  note: {n}");
    }

    let gpu = Gpu::new(0)?;
    let mut sol = solve(&gpu, &lowered)?;
    if let Some(sw) = &lowered.supply_sweep {
        let result = run_sweep(&gpu, &case, sw)?;
        sol.report.pue.free_cooling_ceiling = ceiling_of(&result.points);
        sol.sweep = Some(result);
    }

    print_report(&lowered, &sol);

    if let Some(p) = csv {
        write_csv(p, &lowered, &sol)?;
        println!("\nwrote {}", p.display());
    }
    if let Some(p) = json {
        let ended_ms = epoch_ms(std::time::SystemTime::now());
        let rid = match run_id {
            Some(v) => v.to_string(),
            None => format!("v_{}", compact_stamp(started_ms)),
        };
        // A failed device probe is a `null` in the document, never an aborted
        // run - no `?` on either call.
        let ctx = gpu.ctx();
        let device = match (ctx.name(), ctx.compute_capability()) {
            (Ok(n), Ok((major, minor))) => Some((n, format!("{major}{minor}"))),
            _ => None,
        };
        let meta = DocMeta { run_id: &rid, case_path, started_ms, ended_ms, device };
        write_document(p, &build_document(&lowered, &sol, &meta))?;
        println!("\nwrote {}", p.display());
    }
    Ok(())
}

// ---- D5: per-wall §6.4 y+ and the EQ-B4 mixed-convection ratio -----------

/// EQ-B4's regime thresholds: `forced` below, `natural` above, `mixed`
/// between. The seed's prose carries no numbers; one decade either side of
/// one is the house choice, and both are printed in the caveat text.
const GR_RE2_FORCED_BELOW: f64 = 0.1;
/// See [`GR_RE2_FORCED_BELOW`].
const GR_RE2_NATURAL_ABOVE: f64 = 10.0;

/// Division floor for EQ-B4's approach speed, m/s: a stagnant first cell
/// must not divide by zero. The reported speed is the unfloored mean.
const U_FLOOR: f64 = 1e-6;

/// `Scalar` widened to the f64 the report carries, correct under either
/// precision feature. Identity under the default f64 build - which is why
/// the conversion is spelled once here, behind an `allow`: the house clippy
/// gate counts warnings, and an identity conversion at every call site of
/// the wall table would add ten.
#[allow(clippy::useless_conversion)]
fn wide(x: Scalar) -> f64 {
    f64::from(x)
}

/// §6.4: y+ = C_mu^(1/4) y sqrt(k_P) / nu, the expression `lowmach.rs` prints per patch.
fn y_plus(k_p: f64, y: f64, nu: f64, cmu: f64) -> f64 {
    cmu.powf(0.25) * y * k_p.max(0.0).sqrt() / nu
}

/// EQ-B4 (Incropera & DeWitt section 9.9): Gr/Re^2 = g beta dT L / U^2 - no
/// SPEC-LIT section; the mixed-convection criterion the seed carries.
fn gr_over_re2(g: f64, beta: f64, dt: f64, l: f64, u: f64) -> f64 {
    g * beta * dt * l / (u * u)
}

/// `forced` below [`GR_RE2_FORCED_BELOW`], `natural` above
/// [`GR_RE2_NATURAL_ABOVE`], `mixed` between.
fn regime_word(x: f64) -> &'static str {
    if x < GR_RE2_FORCED_BELOW {
        "forced"
    } else if x > GR_RE2_NATURAL_ABOVE {
        "natural"
    } else {
        "mixed"
    }
}

/// The wall-function coefficients this driver passes to `KEpsilon::new` and
/// reads back for §6.4's y+ - one binding, so §15.6's "the same C_mu in the
/// model and at the wall" is a fact of this file and a test below pins it.
fn wall_coeffs() -> ofgpu::io::case::WallFunctionCoeffs {
    ofgpu::io::case::WallFunctionCoeffs::default()
}

/// One wall patch's §6.4 y+ triple and its EQ-B4 mixed-convection ratio,
/// computed on the host after the last corrector, as `lowmach.rs` does.
#[derive(Debug, Clone)]
struct WallPatchReport {
    patch: String,
    y_plus_min: f64,
    y_plus_mean: f64,
    y_plus_max: f64,
    /// `|T_w - T_P|`, area-weighted; `None` when the case gives the wall no
    /// temperature (`adiabaticWall`), so Gr/Re^2 is undefined there.
    delta_t: Option<f64>,
    /// Area-weighted first-cell speed, m/s - what the log law is fitted to.
    approach_speed: f64,
    /// The room's extent along gravity, m - one length per run.
    length_scale: f64,
    gr_over_re2: Option<f64>,
    /// `forced` | `mixed` | `natural` | `notApplicable`.
    regime: &'static str,
}

/// What one run produced.
struct RoomSolution {
    report: MetricReport,
    /// Per-fan `(patch, Q, dp, shaft power)`.
    fans: Vec<(String, Scalar, Scalar, Scalar)>,
    /// The §53.6 caveat, when there is a jump to caveat.
    jump_caveat: Option<String>,
    /// Peak and mean rack-inlet temperature, K.
    t_inlet_max: Scalar,
    /// Every rack's own sampled inlet temperature.
    rack_inlets: Vec<(String, Scalar)>,
    /// §54's supersaturation report, when humidity was transported.
    supersaturation: Option<(usize, Scalar)>,
    /// The molar-mass caveat of §54.4, when it applies.
    molar_caveat: Option<String>,
    /// Every patch's net volumetric flow, m^3/s, outward positive. They must
    /// sum to zero.
    patch_flow: Vec<(String, Scalar)>,
    /// What the LAST iteration's assembled system turned out to be - the
    /// separability, symmetry and coefficient constancy the printed
    /// paragraph reports. Carried out of the loop so the document can say it
    /// too.
    probe: Option<ProbeSummary>,
    /// §55.4's sweep, when the case asked for one.
    sweep: Option<SweepResult>,
    /// §6.4 y+ and EQ-B4 Gr/Re^2 per wall patch; empty when the case has no wall.
    wall_patches: Vec<WallPatchReport>,
}

/// One point of §55.4's sweep: the supply temperature the case was SET to, the
/// flux-weighted one the solve produced, and the two indices.
#[derive(Debug, Clone, PartialEq)]
struct SweepPoint {
    t_set: Scalar,
    t_supply: Scalar,
    rci_hi: Scalar,
    rci_lo: Scalar,
}

/// The sweep's outcome: every point, and whether `RCI_HI` came out
/// non-increasing in `t_set`.
#[derive(Debug, Clone, PartialEq)]
struct SweepResult {
    points: Vec<SweepPoint>,
    monotone: bool,
}

/// §55.4's ceiling: the highest swept `t_set` whose `RCI_HI` is exactly
/// `100.0` - which `rci_hi` returns exactly when no sample exceeds the
/// recommended limit, so equality is the test and never a tolerance.
/// Order-independent; `None` when no swept value holds.
fn ceiling_of(points: &[SweepPoint]) -> Option<Scalar> {
    let mut best: Option<Scalar> = None;
    for p in points {
        if p.rci_hi == 100.0 {
            best = Some(match best {
                Some(b) if b > p.t_set => b,
                _ => p.t_set,
            });
        }
    }
    best
}

/// §55.4: `RCI_HI` is expected non-increasing in `t_set`. Round-off of one
/// part in a million per cent is not a rise; a real rise is a physical
/// impossibility a fixed iteration budget can still produce, and is
/// reported, never refused.
fn sweep_is_monotone(points: &[SweepPoint]) -> bool {
    let mut by_t: Vec<&SweepPoint> = points.iter().collect();
    by_t.sort_by(|a, b| {
        a.t_set
            .partial_cmp(&b.t_set)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    by_t.windows(2).all(|w| w[1].rci_hi <= w[0].rci_hi + 1e-6)
}

/// The largest rise of `RCI_HI` between consecutive swept temperatures and
/// the pair that produced it - the `NOT monotone` print. `None` when the
/// table is monotone in the `sweep_is_monotone` sense.
fn worst_rise(points: &[SweepPoint]) -> Option<(Scalar, Scalar, Scalar)> {
    let mut by_t: Vec<&SweepPoint> = points.iter().collect();
    by_t.sort_by(|a, b| {
        a.t_set
            .partial_cmp(&b.t_set)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut worst: Option<(Scalar, Scalar, Scalar)> = None;
    for w in by_t.windows(2) {
        let rise = w[1].rci_hi - w[0].rci_hi;
        if rise > 1e-6 && worst.map_or(true, |(d, _, _)| rise > d) {
            worst = Some((rise, w[0].t_set, w[1].t_set));
        }
    }
    worst
}

/// The printed sweep block: empty when the run carried no sweep.
fn sweep_lines(s: &RoomSolution) -> Vec<String> {
    let Some(sw) = &s.sweep else {
        return Vec::new();
    };
    let mut lines = Vec::new();
    lines.push(format!(
        "--- supply-temperature sweep (SPEC-LIT 55.4): {} solve(s) beside the base run ---",
        sw.points.len()
    ));
    lines.push("  T_set          T_supply       RCI_HI       RCI_LO".to_string());
    for p in &sw.points {
        lines.push(format!(
            "  {:8.2} K   {:8.3} K   {:8.3} %   {:8.3} %",
            p.t_set, p.t_supply, p.rci_hi, p.rci_lo
        ));
    }
    if sw.monotone {
        lines.push("  monotone in T_set: yes".to_string());
    } else {
        let (d, a, b) = worst_rise(&sw.points).unwrap_or((0.0, 0.0, 0.0));
        lines.push(format!(
            "  NOT monotone in T_set: RCI_HI rises by {d:.3} between {a:.2} K and {b:.2} K - \
             two solves of the same iteration budget did not reach the same residual; the \
             ceiling below still uses the exact-100 rule"
        ));
    }
    match s.report.pue.free_cooling_ceiling {
        Some(k) => {
            let c = k - 273.15;
            lines.push(format!(
                "  free-cooling ceiling: {k:.2} K ({c:.2} C) - the highest swept supply \
                 temperature holding RCI_HI at exactly 100 %"
            ));
        }
        None => {
            let lowest = sw.points.iter().min_by(|a, b| {
                a.t_set
                    .partial_cmp(&b.t_set)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            if let Some(p) = lowest {
                lines.push(format!(
                    "  free-cooling ceiling: none - no swept supply temperature holds RCI_HI \
                     at 100 %; the lowest, {:.2} K, gives {:.3} %",
                    p.t_set, p.rci_hi
                ));
            }
        }
    }
    lines
}

/// §55.4's sweep: re-solve the case at every swept supply temperature,
/// ascending, ALL of them - the monotone verdict needs every point. The
/// re-lowered cases' notes are not printed again; the base run printed them.
fn run_sweep(gpu: &Gpu, case: &DcCase, sweep: &SupplySweep) -> Result<SweepResult> {
    let n = sweep.temperatures.len();
    let mut points = Vec::new();
    for (i, t) in sweep.temperatures.iter().enumerate() {
        println!(
            "\n=== sweep {}/{}: {} \"{}\" set to {:.2} K ({:.2} C) ===",
            i + 1,
            n,
            sweep.what,
            sweep.patch,
            t,
            t - 273.15
        );
        let lc_t = case.with_supply_temperature(f64::from(*t))?.lower()?;
        let s_t = solve(gpu, &lc_t)?;
        points.push(SweepPoint {
            t_set: *t,
            t_supply: s_t.report.t_supply,
            rci_hi: s_t.report.rci_hi,
            rci_lo: s_t.report.rci_lo,
        });
    }
    let monotone = sweep_is_monotone(&points);
    Ok(SweepResult { points, monotone })
}

#[allow(clippy::too_many_lines)]
fn solve(gpu: &Gpu, lc: &LoweredDcCase) -> Result<RoomSolution> {
    let hm = &lc.mesh;
    let mesh = GpuMesh::upload(gpu, hm)?;
    let rho = lc.air.rho as Scalar;
    let cp = lc.air.cp as Scalar;

    // ---- controls ---------------------------------------------------------
    let mctrl = MomentumControls {
        nu: lc.air.nu as Scalar,
        u_relax: lc.numerics.u_relax as Scalar,
        u_solver: lc.solver.clone(),
        ..MomentumControls::default()
    };
    let sctrl = SimpleControls {
        momentum: mctrl,
        p_solver: lc.solver.clone(),
        p_relax: lc.numerics.p_relax as Scalar,
        ..SimpleControls::default()
    };
    let buoy = BuoyancyCoeffs {
        g: Vec3::new(
            lc.air.gravity[0] as Scalar,
            lc.air.gravity[1] as Scalar,
            lc.air.gravity[2] as Scalar,
        ),
        t_ref: lc.air.t_ref as Scalar,
        ..BuoyancyCoeffs::default()
    };

    let mut simple = Simple::new(gpu, hm, &mesh, sctrl, buoy)?;

    // ---- boundary conditions ---------------------------------------------
    //
    // Written directly onto S4's triple rather than through a `0/` directory:
    // every one of them is decided by the case's own `fans`/`tiles`/`patches`
    // blocks, which have already been checked to name every patch exactly
    // once, so there is no file for them to disagree with.
    let nbf = hm.n_boundary_faces;
    let mut p_kind = vec![BcKind::ZeroGradient as Label; nbf];
    let mut p_fr = vec![0.0 as Scalar; nbf];
    let mut p_rv = vec![0.0 as Scalar; nbf];
    let mut u_kind = vec![BcKind::FixedValue as Label; nbf];
    let mut u_fr = vec![1.0 as Scalar; nbf];
    let mut t_kind = vec![BcKind::ZeroGradient as Label; nbf];
    let mut t_fr = vec![0.0 as Scalar; nbf];
    let mut t_rv = vec![0.0 as Scalar; nbf];
    let mut y_kind = vec![BcKind::ZeroGradient as Label; nbf];
    let mut y_fr = vec![0.0 as Scalar; nbf];
    let mut y_rv = vec![0.0 as Scalar; nbf];
    // SPEC-LIT S15.5: which faces get a wall function is asked of `nut`'s and
    // `epsilon`'s OWN patch types. Here the case's `patches` block decides:
    // every `wall`/`adiabaticWall` rule is a wall, and a fan, a tile or a
    // `fixedPressure` is not.
    let mut wall_faces = ofgpu::field_setup::WallFaces::none(nbf);

    let patch_faces = |name: &str| -> std::ops::Range<usize> {
        let p = hm.patches.iter().find(|p| p.name == name).expect("checked at lowering");
        p.start..p.start + p.size
    };

    // §52.4: `fr` is seeded at 1, not 0. Both conditions have `fr` in
    // `(0, 1]` for every finite curve slope and resistance, so a fan or tile
    // patch ALWAYS pins the pressure level - and `Simple::initialise` decides
    // whether to pin a reference cell by reading `fr` before `crate::fan` has
    // written it. A zero seed makes it pin one as well, and
    // `fix_pressure_level` then subtracts that cell after every solve,
    // fighting the absolute pressure the curve imposes. See
    // `field_setup`'s own note on the same seed.
    for f in &lc.fans {
        for bf in patch_faces(&f.patch) {
            p_kind[bf] = BcKind::FanPressure as Label;
            p_fr[bf] = 1.0;
            p_rv[bf] = f.ambient;
            // ZERO-GRADIENT on the velocity, and NOT
            // `pressureInletOutletVelocity`. The design note SPEC-LIT S52 was
            // written from says kind 12 is "exactly right - the flux sets the
            // normal component on inflow", and in THIS solver it is not:
            // `field_setup` seeds its `refValue` from the interior velocity
            // ONCE, nothing refreshes it from the flux, and
            // `momFluxIsPrescribed` treats any `fr >= 1` face as a prescribed
            // velocity. An inflow face is therefore pinned at whatever it was
            // seeded with - zero, on a room starting from rest - and the fan's
            // pressure can move no air through it at all. Measured: the floor
            // tile carried exactly `0.0` flux on every inflow face.
            //
            // `fr = 0` makes `momFluxIsPrescribed` false, so
            // `phi = phi_HbyA - rAU_f snGrad(p)` and the PRESSURE equation
            // owns the flux - which is the whole point of a fan or a jump on
            // `p`. The cost is that an inflow's velocity is the extrapolated
            // interior one, so the near-opening jet is wrong; that is the same
            // limitation §53.6 already records for a pressure-jump tile, and
            // for the same reason.
            u_kind[bf] = BcKind::ZeroGradient as Label;
            u_fr[bf] = 0.0;
        }
    }
    for j in &lc.jumps {
        if let ofgpu::fan::PorousJump::Boundary { patch, plenum, .. } = j {
            for bf in patch_faces(patch) {
                p_kind[bf] = BcKind::PorousJumpPressure as Label;
                p_fr[bf] = 1.0;
                p_rv[bf] = *plenum;
                // Zero-gradient, for the reason given on the fan patch above.
                u_kind[bf] = BcKind::ZeroGradient as Label;
                u_fr[bf] = 0.0;
            }
        }
    }
    for (name, p) in &lc.patch_pressure {
        for bf in patch_faces(name) {
            if hm.b_kind[bf] == PatchKind::Empty as Label {
                p_kind[bf] = BcKind::Empty as Label;
                u_kind[bf] = BcKind::Empty as Label;
                t_kind[bf] = BcKind::Empty as Label;
                y_kind[bf] = BcKind::Empty as Label;
                continue;
            }
            match p {
                Some(v) => {
                    p_kind[bf] = BcKind::FixedValue as Label;
                    p_fr[bf] = 1.0;
                    p_rv[bf] = *v;
                    u_kind[bf] = BcKind::ZeroGradient as Label;
                    u_fr[bf] = 0.0;
                }
                None => {
                    // A wall: no slip, zero-gradient pressure, and a wall
                    // function on `nut` and `epsilon` (S15.2/S15.5).
                    u_kind[bf] = BcKind::FixedValue as Label;
                    u_fr[bf] = 1.0;
                    wall_faces.constrained_cells[bf] = true;
                    wall_faces.nut[bf] = true;
                }
            }
        }
    }
    // Temperatures: an inflow fan or a tile carries one, a `wall` rule
    // carries one, everything else is adiabatic.
    for (name, t) in &lc.inflow_temperature {
        for bf in patch_faces(name) {
            t_kind[bf] = BcKind::InletOutlet as Label;
            t_fr[bf] = 1.0;
            t_rv[bf] = *t;
        }
    }
    for (name, t) in &lc.patch_temperature {
        if let Some(v) = t {
            for bf in patch_faces(name) {
                if hm.b_kind[bf] == PatchKind::Empty as Label {
                    continue;
                }
                t_kind[bf] = BcKind::FixedValue as Label;
                t_fr[bf] = 1.0;
                t_rv[bf] = *v;
            }
        }
    }
    for (name, yv) in &lc.inflow_humidity {
        for bf in patch_faces(name) {
            y_kind[bf] = BcKind::InletOutlet as Label;
            y_fr[bf] = 1.0;
            y_rv[bf] = *yv;
        }
    }

    {
        let p = simple.p_mut();
        gpu.write(&mut p.bc_kind, &p_kind)?;
        gpu.write(&mut p.fr, &p_fr)?;
        gpu.write(&mut p.ref_value, &p_rv)?;
        let u = simple.u_mut();
        gpu.write(&mut u.bc_kind, &u_kind)?;
        gpu.write(&mut u.fr, &u_fr)?;
    }
    simple.initialise(gpu)?;

    // ---- the fan patches and the tiles ------------------------------------
    // (S52.13)'s density ratio is already in the lowered curve: the reader
    // sets `rho` from `air.rho` so the ratio means something on the lowered
    // case, which is what a pair test on `rhoCurve` can check.
    let devices = FlowDevices::new(gpu, hm, lc.fans.clone(), &lc.jumps, rho)?;
    let jump_caveat = devices.jump_caveat();
    simple.set_flow_devices(devices);

    // ---- temperature -------------------------------------------------------
    // `ScalarTransport` reads its linear solver from `k_solver` and its
    // relaxation from `k_relax` - the same slots the turbulence fields use,
    // because it is the same machinery (S19).
    let tctrl_turb = TurbulenceControls {
        k_solver: lc.solver.clone(),
        epsilon_solver: lc.solver.clone(),
        k_relax: 0.7,
        eps_relax: 0.7,
        steady: true,
        ..TurbulenceControls::default()
    };
    let tctrl = TurbulenceControls {
        k_solver: lc.solver.clone(),
        k_relax: lc.numerics.t_relax as Scalar,
        steady: true,
        ..TurbulenceControls::default()
    };
    let mut heat = ScalarTransport::new(
        gpu,
        hm,
        &mesh,
        "T",
        ScalarTransportCoeffs { pr: lc.air.pr as Scalar, prt: lc.air.prt as Scalar },
        tctrl.clone(),
    )?;
    {
        let t = heat.field_mut();
        gpu.write(&mut t.bc_kind, &t_kind)?;
        gpu.write(&mut t.fr, &t_fr)?;
        gpu.write(&mut t.ref_value, &t_rv)?;
        gpu.write(&mut t.f, &vec![lc.run.initial_temperature as Scalar; hm.n_cells])?;
        gpu.write(&mut t.f0, &vec![lc.run.initial_temperature as Scalar; hm.n_cells])?;
    }
    // S18: the rack heat, as a cell-zone source in K/s.
    for r in &lc.racks {
        // S18's units: a heat release reaching the TEMPERATURE equation is
        // `Qdot/(rho c_p V)` in K/s, and `sources.rs` does that division in
        // one place precisely so no driver has to remember it.
        heat.sources_mut().push(ofgpu::sources::Source::new(
            gpu,
            hm,
            &r.name,
            CellSelector::Cells(r.cells.iter().map(|c| *c as usize).collect()),
            SourceTerm::Explicit(r.q_vol / (rho * cp)),
        )?);
    }
    heat.initialise(gpu)?;

    // ---- humidity, when the case asks for it -------------------------------
    let mut humidity: Option<(ScalarTransport<'_>, Psychrometrics)> = match lc.humidity {
        None => None,
        Some(h) => {
            // §54.1: one more transported scalar on the SAME conservative
            // phi. The diffusivity is carried through the Prandtl-number slot
            // as a Schmidt number, which is the same coefficient in the same
            // place - `D_eff = D + nu_t/Sc_t`.
            let sc = lc.air.nu as Scalar / h.d as Scalar;
            let mut yv = ScalarTransport::new(
                gpu,
                hm,
                &mesh,
                "Yv",
                ScalarTransportCoeffs { pr: sc, prt: h.sc_t as Scalar },
                tctrl,
            )?;
            {
                let f = yv.field_mut();
                gpu.write(&mut f.bc_kind, &y_kind)?;
                gpu.write(&mut f.fr, &y_fr)?;
                gpu.write(&mut f.ref_value, &y_rv)?;
                let seed = lc.inflow_humidity.values().copied().fold(0.0 as Scalar, Scalar::max);
                gpu.write(&mut f.f, &vec![seed; hm.n_cells])?;
                gpu.write(&mut f.f0, &vec![seed; hm.n_cells])?;
            }
            yv.initialise(gpu)?;
            let psy = Psychrometrics::new(gpu, &mesh, h.barometric_pressure as Scalar)?;
            Some((yv, psy))
        }
    };

    // ---- turbulence --------------------------------------------------------
    //
    // A room at U ~ 0.5 m/s over 3 m is Re ~ 1e5. A steady laminar solve
    // there does not converge - it diverges, which is exactly what a first
    // draft of this driver did (Q ran away to 5e3 m^3/s in ten iterations on
    // a 400-cell box). SPEC-LIT S6's standard k-epsilon with S15's wall
    // functions is what a room-airflow model needs and what this uses.
    // §15.6: the SAME C_mu must reach the model and the wall treatment; the
    // one binding below is what both read, and a test pins the two equal.
    let wc = wall_coeffs();
    let mut turb = KEpsilon::new(
        gpu,
        hm,
        &mesh,
        KEpsilonCoeffs::default(),
        tctrl_turb,
        wc,
        &wall_faces,
        &ofgpu::field_setup::NutRoughness::none(nbf),
    )?;
    {
        // A room's inlet turbulence: 10 % intensity on the supply velocity
        // scale, mixing length one tenth of the room height (S6.5's own
        // estimates). Seeded uniformly; the model transports it from there.
        let u_ref = 0.5 as Scalar;
        let k0 = 1.5 * (0.1 * u_ref) * (0.1 * u_ref);
        let l0 = 0.1 * (lc.mesh.c.iter().fold(0.0 as Scalar, |m, c| m.max(c.z))).max(0.1);
        let e0 = KEpsilonCoeffs::default().cmu.powf(0.75) * k0.powf(1.5) / l0;
        gpu.write(&mut turb.k_mut().f, &vec![k0; hm.n_cells])?;
        gpu.write(&mut turb.epsilon_mut().f, &vec![e0; hm.n_cells])?;
        gpu.write(&mut turb.k_mut().bf, &vec![k0; nbf])?;
        gpu.write(&mut turb.epsilon_mut().bf, &vec![e0; nbf])?;
    }
    turb.initialise(gpu, &simple.flow_state())?;

    let mut backend = ofgpu::pressure::PbicgstabBackend::new(lc.solver.clone());
    backend.setup(gpu, hm, &mesh, &SystemProbe::default())?;
    let mut probed = false;
    let mut probe_out: Option<ProbeSummary> = None;

    // ---- the outer loop ----------------------------------------------------
    for it in 0..lc.run.iterations {
        // §54.4: the buoyancy field. With humidity ON it is the virtual
        // temperature; with humidity off it is `T` itself, and
        // `momentum::update_buoyancy` is the SAME unmodified function either
        // way - which is what makes the dry default bit-for-bit unmoved.
        let use_virtual =
            lc.humidity.map(|h| h.virtual_temperature).unwrap_or(false) && humidity.is_some();
        if use_virtual {
            let (yv, psy) = humidity.as_mut().expect("checked");
            psy.update_virtual_temperature(gpu, heat.field(), yv.field())?;
        }

        {
            let t_for_buoyancy: &GpuScalarField = if use_virtual {
                humidity.as_ref().expect("checked").1.virtual_temperature_field()
            } else {
                heat.field()
            };
            simple.correct_outer(gpu, &mut backend, turb.nut(), t_for_buoyancy, false)?;
        }

        // SPEC-LIT §52.8: the cost of a fan curve is the cuFFT direct Poisson
        // backend, and it must be PRINTED rather than quietly fallen back
        // from. Probed off the REAL assembled system at the LAST iteration,
        // not the first: on the first, `Q` is zero, so a quadratic curve has
        // `S = 0` and a jump has `R = 0`, and every one of these faces is
        // still uniformly Dirichlet. Probing there would report that the FFT
        // path is available on a system where it is not.
        if !probed && it + 1 == lc.run.iterations {
            probed = true;
            let (g, bg) = simple.pressure_laplacian_coeffs();
            let probe =
                SystemProbe::probe(gpu, hm, simple.p(), simple.pressure_matrix(), g, bg)?;
            println!(
                "pressure system: {} cells | separable BCs {} | symmetric {} | \
                 constant coefficient {}",
                probe.n_cells,
                if probe.separable_bcs {
                    "yes".to_string()
                } else {
                    format!("NO ({})", probe.non_separable_reason)
                },
                probe.symmetric,
                probe.constant_coefficient
            );
            if !probe.separable_bcs || !probe.constant_coefficient {
                println!("  {FFT_UNAVAILABLE_TEXT}");
            }
            if !probe.symmetric {
                println!("  {ASYMMETRIC_MATRIX_TEXT}");
            }
            probe_out = Some(ProbeSummary {
                separable_bcs: probe.separable_bcs,
                symmetric: probe.symmetric,
                constant_coefficient: probe.constant_coefficient,
                non_separable_reason: probe.non_separable_reason.clone(),
            });
        }

        let flow = simple.flow_state();
        turb.correct_buoyant(gpu, &flow, Some(heat.field()))?;
        heat.correct(gpu, &flow, turb.nut())?;
        if let Some((yv, _)) = humidity.as_mut() {
            yv.correct(gpu, &flow, turb.nut())?;
        }

        if lc.run.report_every > 0 && (it + 1) % lc.run.report_every == 0 {
            if let Some(d) = simple.flow_devices() {
                let st = d.states(gpu)?;
                let line: Vec<String> = d
                    .fans()
                    .iter()
                    .zip(&st)
                    .map(|(f, s)| {
                        format!("{}: Q = {:.4} m^3/s, dp = {:.1} Pa", f.patch, s.q, s.dp)
                    })
                    .collect();
                println!("  iter {:5}  {}", it + 1, line.join("  |  "));
            }
        }
    }

    // ---- the report ---------------------------------------------------------
    let devices = simple.flow_devices().expect("attached above");
    let st = devices.states(gpu)?;
    let (powers, total_power) = devices.shaft_power(gpu)?;
    let fan_list: Vec<(String, Scalar, Scalar, Scalar)> = devices
        .fans()
        .iter()
        .zip(&st)
        .zip(&powers)
        .map(|((f, s), p)| (f.patch.clone(), s.q, s.dp, *p))
        .collect();

    let cap = lc
        .racks
        .iter()
        .map(|r| r.samples.len().max(r.cells.len()))
        .chain([lc.supply_span.size, lc.return_span.size])
        .max()
        .unwrap_or(1);
    let mut mt = Metrics::new(gpu, devices, cap.max(1))?;

    // RCI, over every rack's samples concatenated.
    let mut all_samples: Vec<Label> = Vec::new();
    let mut rack_inlets = Vec::new();
    let t_host = gpu.download(&heat.field().f)?;
    for r in &lc.racks {
        all_samples.extend_from_slice(&r.samples);
        let mean: Scalar = r.samples.iter().map(|c| t_host[*c as usize]).sum::<Scalar>()
            / r.samples.len() as Scalar;
        rack_inlets.push((r.name.clone(), mean));
    }
    all_samples.sort_unstable();
    let n_samples = all_samples.len();
    let (hi, lo) = if n_samples == 0 {
        (0.0, 0.0)
    } else {
        let mut mt2 = Metrics::new(gpu, devices, n_samples)?;
        let d = gpu.upload(&all_samples)?;
        mt2.rci_excess(gpu, heat.field(), &d, n_samples, lc.class)?
    };
    let t_inlet_max = all_samples
        .iter()
        .fold(0.0 as Scalar, |m, c| m.max(t_host[*c as usize]));

    // RTI: flux-weighted supply and return means (S55.2).
    let (t_supply, _) =
        mt.flux_weighted_mean(gpu, lc.supply_span, simple.phi(), heat.field())?;
    let (t_return, _) =
        mt.flux_weighted_mean(gpu, lc.return_span, simple.phi(), heat.field())?;

    let q_it: Scalar = lc.racks.iter().map(|r| r.power).sum();
    let m_it: Scalar = lc.racks.iter().map(|r| r.flow).sum::<Scalar>() * rho;
    let dt_eq = if m_it > 0.0 {
        dt_equipment_from_heat(q_it, m_it, cp)?
    } else {
        1.0
    };
    let rti_v = rti(t_return, t_supply, dt_eq);

    // SHI/RHI (S55.4): the pre-heat of the cold air against the useful
    // pickup, both from the rack-inlet samples and the stated flows.
    let d_q: Scalar = lc
        .racks
        .iter()
        .zip(&rack_inlets)
        .map(|(r, (_, t_in))| r.flow * rho * cp * (t_in - t_supply))
        .sum();
    let (shi, rhi) = shi_rhi(d_q, q_it);

    // Continuity over the whole boundary. A room whose openings do not sum to
    // zero has not converged, whatever else the report says, and a driver
    // that printed metrics without checking would be reporting numbers taken
    // off a field that does not conserve mass.
    let mut patch_flow: Vec<(String, Scalar)> = Vec::new();
    {
        let bphi = gpu.download(&simple.phi().bf)?;
        for pi in &hm.patches {
            if pi.kind == PatchKind::Empty {
                continue;
            }
            let q: Scalar = (pi.start..pi.start + pi.size).map(|bf| bphi[bf]).sum();
            patch_flow.push((pi.name.clone(), q));
        }
    }

    let supersaturation = match humidity.as_mut() {
        None => None,
        Some((yv, psy)) => {
            psy.update(gpu, heat.field(), yv.field())?;
            let s = psy.supersaturation(gpu, heat.field(), yv.field())?;
            Some((s.cells, s.worst))
        }
    };
    let molar_caveat = match humidity.as_ref() {
        None => None,
        Some((yv, _)) => {
            let m = gpu
                .download(&yv.field().f)?
                .iter()
                .take(hm.n_cells)
                .fold(0.0 as Scalar, |a, b| a.max(*b));
            Psychrometrics::molar_mass_caveat(m)
        }
    };

    // D5: per-wall §6.4 y+ and EQ-B4 Gr/Re^2, host work on the state the run
    // just reported - the loop's last `correct_buoyant`/`correct` wrote the
    // fields, so this is the reported state. It runs once per sweep point
    // too, so it stays cheap and mute. The wall rule is the `None` arm of
    // the pressure loop above: a patch is a wall iff its pressure rule is
    // `None` and it is not `empty` - exactly the faces `wall_faces.nut` was
    // set for, which is what y+ diagnoses. No DC patch is
    // `PatchKind::Wall` (the lowering builds every patch as `"patch"`), and
    // a `symmetry` rule lowers to `None` as well, so it is reported as the
    // no-slip wall this driver already treats it as.
    let k_host = gpu.download(&turb.k().f)?;
    let u_host = gpu.download(&simple.u().f)?;
    let g_mag = (lc.air.gravity[0] * lc.air.gravity[0]
        + lc.air.gravity[1] * lc.air.gravity[1]
        + lc.air.gravity[2] * lc.air.gravity[2])
        .sqrt();
    // One length per run: the room's extent along gravity, over EVERY
    // boundary face.
    let length_scale = if g_mag < 1e-30 {
        0.0
    } else {
        let ghat = [
            lc.air.gravity[0] / g_mag,
            lc.air.gravity[1] / g_mag,
            lc.air.gravity[2] / g_mag,
        ];
        let mut lo = f64::INFINITY;
        let mut hi = f64::NEG_INFINITY;
        for c in &hm.b_cf {
            let d =
                wide(c.x) * ghat[0] + wide(c.y) * ghat[1] + wide(c.z) * ghat[2];
            lo = lo.min(d);
            hi = hi.max(d);
        }
        hi - lo
    };
    let beta = 1.0 / lc.air.t_ref;
    let cmu = wide(wc.cmu);
    let mut wall_patches: Vec<WallPatchReport> = Vec::new();
    for pi in &hm.patches {
        if pi.kind == PatchKind::Empty {
            continue;
        }
        // A fan's or a tile's patch is absent from the map entirely, so
        // `Some(None)` is a wall and nothing else is.
        if !matches!(lc.patch_pressure.get(&pi.name), Some(None)) {
            continue;
        }
        if pi.size == 0 {
            continue;
        }
        let t_w = lc.patch_temperature.get(&pi.name).and_then(|t| *t);
        let mut y_min = f64::INFINITY;
        let mut y_max = f64::NEG_INFINITY;
        let mut y_sum = 0.0;
        let mut w_sum = 0.0;
        let mut dt_sum = 0.0;
        let mut u_sum = 0.0;
        for bf in pi.start..pi.start + pi.size {
            let cell = hm.b_face_cells[bf] as usize;
            let yp = y_plus(
                wide(k_host[cell]),
                wide(hm.b_y[bf]),
                lc.air.nu,
                cmu,
            );
            y_min = y_min.min(yp);
            y_max = y_max.max(yp);
            y_sum += yp;
            let w = wide(hm.b_mag_sf[bf]);
            w_sum += w;
            if let Some(t_w) = t_w {
                dt_sum += w * (wide(t_w) - wide(t_host[cell])).abs();
            }
            u_sum += w * wide(u_host[cell].mag());
        }
        // y+ is a FACE-COUNT mean, as `lowmach.rs` prints it; dT and U are
        // area-weighted. The U floor guards the ratio's division only.
        let delta_t = t_w.map(|_| dt_sum / w_sum.max(1e-30));
        let approach_speed = u_sum / w_sum.max(1e-30);
        let gr = delta_t
            .map(|dt| gr_over_re2(g_mag, beta, dt, length_scale, approach_speed.max(U_FLOOR)));
        wall_patches.push(WallPatchReport {
            patch: pi.name.clone(),
            y_plus_min: y_min,
            y_plus_mean: y_sum / pi.size as f64,
            y_plus_max: y_max,
            delta_t,
            approach_speed,
            length_scale,
            gr_over_re2: gr,
            regime: gr.map_or("notApplicable", regime_word),
        });
    }

    Ok(RoomSolution {
        report: MetricReport {
            rci_hi: rci_hi(hi, n_samples, lc.class),
            rci_lo: rci_lo(lo, n_samples, lc.class),
            n_samples,
            rti: rti_v,
            shi,
            rhi,
            t_supply,
            t_return,
            dt_equipment: dt_eq,
            // §55.2: this driver models racks as heat-release zones with a
            // stated flow, so dT_equipment is DERIVED, never measured. Saying
            // so is the point of the flag.
            dt_measured: false,
            pue: PueInputs {
                fan_power: total_power,
                fan_power_each: powers,
                it_heat: q_it,
                free_cooling_ceiling: None,
            },
        },
        fans: fan_list,
        jump_caveat,
        t_inlet_max,
        rack_inlets,
        supersaturation,
        molar_caveat,
        patch_flow,
        probe: probe_out,
        sweep: None,
        wall_patches,
    })
}

fn print_report(lc: &LoweredDcCase, s: &RoomSolution) {
    let r = &s.report;
    println!("\n=== SPEC-LIT S55 report ===");
    println!("{}", lc.class.describe());
    println!(
        "RCI_HI  {:8.3} %   RCI_LO {:8.3} %   over n = {} sample points ({:?})",
        r.rci_hi, r.rci_lo, r.n_samples, lc.samples
    );
    println!(
        "RTI     {:8.3} %   ({})",
        r.rti,
        if r.rti < 99.5 {
            "bypass - more supply air than the racks draw"
        } else if r.rti > 100.5 {
            "recirculation - the racks draw more than the supply delivers"
        } else {
            "balanced"
        }
    );
    println!(
        "        dT_equipment {:.3} K, {}",
        r.dt_equipment,
        if r.dt_measured {
            "MEASURED across the racks"
        } else {
            "DERIVED as Q_IT/(mdot cp) - the racks are heat-release zones with a \
             stated flow (S55.2)"
        }
    );
    println!("SHI     {:8.5}      RHI {:8.5}   (SHI + RHI = {})", r.shi, r.rhi, r.shi + r.rhi);
    println!(
        "T_supply {:.3} K   T_return {:.3} K   (both FLUX-weighted, S55.2)",
        r.t_supply, r.t_return
    );
    println!("worst rack-inlet temperature {:.3} K", s.t_inlet_max);
    for (n, t) in &s.rack_inlets {
        println!("  rack {n}: inlet {t:.3} K");
    }

    println!("
--- flow through every opening (m^3/s, outward positive) ---");
    let (net, _scale, ratio) = continuity_of(&s.patch_flow);
    for (n, q) in &s.patch_flow {
        println!("  {n:<16} {q:12.5}");
    }
    println!(
        "  {:<16} {net:12.5}   <- must be zero; {:.2e} of the largest opening",
        "NET",
        f64::from(ratio)
    );

    println!("\n--- fans (S52) ---");
    for (patch, q, dp, w) in &s.fans {
        println!("  {patch}: Q = {q:.5} m^3/s, dp = {dp:.2} Pa, shaft power {w:.1} W");
        // The fan's own `Q` is gathered at the START of a corrector, from the
        // flux the PREVIOUS one produced (S52.7); the patch flow above is the
        // flux at the end. At a true fixed point they coincide, so the gap
        // between them IS the outer-iteration residual of the operating
        // point, and it is printed rather than left for a reader to notice.
        if let Some((_, qp)) = s.patch_flow.iter().find(|(n, _)| n == patch) {
            println!(
                "    the patch itself carried {qp:.5} m^3/s at the last corrector; the \
                 {:.2} % gap is the operating point's own outer residual (S52.6)",
                outer_residual_pct(*q, *qp)
            );
        }
    }
    println!("\n{}", r.pue.describe());
    let sweep = sweep_lines(s);
    if !sweep.is_empty() {
        println!();
        for l in sweep {
            println!("{l}");
        }
    }

    if let Some(c) = &s.jump_caveat {
        println!("\n--- S53.6 ---\n  {c}");
    }
    if let Some((cells, worst)) = s.supersaturation {
        if cells > 0 {
            println!("\n--- S54.5 ---\n  {}", supersaturation_text(cells, worst));
        } else {
            println!("\nno cell is supersaturated.");
        }
    }
    if let Some(c) = &s.molar_caveat {
        println!("\n--- S54.4 ---\n  {c}");
    }
    if !s.wall_patches.is_empty() {
        println!("\n--- S6.4 ---\n  {}", wall_validity_text(&s.wall_patches));
    }
}

fn write_csv(path: &Path, lc: &LoweredDcCase, s: &RoomSolution) -> Result<()> {
    use std::fmt::Write;
    let mut out = String::new();
    let _ = writeln!(out, "quantity,value,unit");
    let r = &s.report;
    let _ = writeln!(out, "RCI_HI,{},%", r.rci_hi);
    let _ = writeln!(out, "RCI_LO,{},%", r.rci_lo);
    let _ = writeln!(out, "n_samples,{},-", r.n_samples);
    let _ = writeln!(out, "RTI,{},%", r.rti);
    let _ = writeln!(out, "SHI,{},-", r.shi);
    let _ = writeln!(out, "RHI,{},-", r.rhi);
    let _ = writeln!(out, "T_supply,{},K", r.t_supply);
    let _ = writeln!(out, "T_return,{},K", r.t_return);
    let _ = writeln!(out, "dT_equipment,{},K", r.dt_equipment);
    let _ = writeln!(out, "IT_heat,{},W", r.pue.it_heat);
    let _ = writeln!(out, "fan_shaft_power,{},W", r.pue.fan_power);
    match r.pue.free_cooling_ceiling {
        Some(k) => {
            let _ = writeln!(out, "free_cooling_ceiling,{k},K");
        }
        None => {
            let _ = writeln!(out, "free_cooling_ceiling,not swept,K");
        }
    }
    if let Some(sw) = &s.sweep {
        let _ = writeln!(out, "sweep_monotone,{},-", sw.monotone);
        for (i, p) in sw.points.iter().enumerate() {
            let _ = writeln!(out, "sweep_{i}_T_set,{},K", p.t_set);
            let _ = writeln!(out, "sweep_{i}_T_supply,{},K", p.t_supply);
            let _ = writeln!(out, "sweep_{i}_RCI_HI,{},%", p.rci_hi);
            let _ = writeln!(out, "sweep_{i}_RCI_LO,{},%", p.rci_lo);
        }
    }
    for (patch, q, dp, w) in &s.fans {
        let _ = writeln!(out, "fan_{patch}_Q,{q},m3/s");
        let _ = writeln!(out, "fan_{patch}_dp,{dp},Pa");
        let _ = writeln!(out, "fan_{patch}_power,{w},W");
    }
    for (n, t) in &s.rack_inlets {
        let _ = writeln!(out, "rack_{n}_inlet,{t},K");
    }
    let _ = writeln!(out, "ashrae_class,{:?},-", lc.class);
    let _ = writeln!(out, "rci_samples,{:?},-", lc.samples);
    std::fs::write(path, out).map_err(|e| ofgpu::error::Error::Parse {
        path: path.display().to_string(),
        msg: e.to_string(),
    })
}

// ==========================================================================
//  5. The JSON document - one file per run
// ==========================================================================

// The document's camelCase keys come from these field names through
// `rename_all`, never hand-written: `rci_hi` serialises as `rciHi`,
// `n_boundary_faces` as `nBoundaryFaces`, `case_sha256` as `caseSha256`.

/// The `machine` block: where and on what the run happened. `hostname` is
/// `None` rather than a fiction when the environment names no host.
#[derive(serde::Serialize)]
#[serde(rename_all = "camelCase")]
struct DcMachine {
    hostname: Option<String>,
    os: &'static str,
    arch: &'static str,
    device: Option<String>,
    compute_capability: Option<String>,
    precision: &'static str,
}

/// The PUE inputs (S55.5) - not a PUE, as `MetricReport::describe` says.
#[derive(serde::Serialize)]
#[serde(rename_all = "camelCase")]
struct PueDoc<'a> {
    fan_power: Scalar,
    fan_power_each: &'a [Scalar],
    it_heat: Scalar,
    free_cooling_ceiling: Option<Scalar>,
    /// §55.4's sweep, one row per solve, ascending in `t_supply_set`; `null` when no sweep.
    supply_temperature_sweep: Option<Vec<SweepPointDoc>>,
    /// Whether `RCI_HI` was non-increasing in `t_supply_set`; `null` when no sweep.
    sweep_monotone: Option<bool>,
}

/// One row of the sweep table, in document key order.
#[derive(serde::Serialize)]
#[serde(rename_all = "camelCase")]
struct SweepPointDoc {
    t_supply_set: Scalar,
    t_supply: Scalar,
    rci_hi: Scalar,
    rci_lo: Scalar,
}

/// The S55 report, field for field what `print_report` prints.
#[derive(serde::Serialize)]
#[serde(rename_all = "camelCase")]
struct ReportDoc<'a> {
    rci_hi: Scalar,
    rci_lo: Scalar,
    n_samples: usize,
    rti: Scalar,
    shi: Scalar,
    rhi: Scalar,
    t_supply: Scalar,
    t_return: Scalar,
    dt_equipment: Scalar,
    dt_measured: bool,
    pue: PueDoc<'a>,
}

/// One fan's operating point. `outer_residual_pct` is `None` when the fan's
/// patch is absent from `patch_flow`.
#[derive(serde::Serialize)]
#[serde(rename_all = "camelCase")]
struct FanDoc<'a> {
    patch: &'a str,
    q: Scalar,
    dp: Scalar,
    shaft_power: Scalar,
    outer_residual_pct: Option<Scalar>,
}

#[derive(serde::Serialize)]
#[serde(rename_all = "camelCase")]
struct RackInletDoc<'a> {
    name: &'a str,
    inlet_t: Scalar,
}

#[derive(serde::Serialize)]
#[serde(rename_all = "camelCase")]
struct PatchFlowDoc<'a> {
    patch: &'a str,
    q: Scalar,
}

/// The whole-boundary closure, computed once and printed and serialised.
#[derive(serde::Serialize)]
#[serde(rename_all = "camelCase")]
struct ContinuityDoc {
    net: Scalar,
    largest_opening: Scalar,
    net_over_largest: Scalar,
}

#[derive(serde::Serialize)]
#[serde(rename_all = "camelCase")]
struct SupersaturationDoc {
    cells: usize,
    worst_excess: Scalar,
}

/// One caveat: the kind is spelled with the section it comes from, so a
/// reader of the document knows where the text's authority lives.
#[derive(serde::Serialize)]
#[serde(rename_all = "camelCase")]
struct CaveatDoc {
    kind: &'static str,
    spec_ref: &'static str,
    text: String,
}

/// One `wallPatches[]` entry - `WallPatchReport` field for field.
#[derive(serde::Serialize)]
#[serde(rename_all = "camelCase")]
struct WallPatchDoc<'a> {
    patch: &'a str,
    y_plus_min: f64,
    y_plus_mean: f64,
    y_plus_max: f64,
    delta_t: Option<f64>,
    approach_speed: f64,
    length_scale: f64,
    gr_over_re2: Option<f64>,
    regime: &'static str,
}

/// The document itself. The FIELD ORDER is the document's key order and must
/// not be rearranged: a reader diffing two runs reads it top to bottom.
/// No key is ever omitted - an unknown value serialises as `null`.
#[derive(serde::Serialize)]
#[serde(rename_all = "camelCase")]
struct DcReportDoc<'a> {
    schema: &'static str,
    run_id: &'a str,
    case_path: String,
    case_sha256: Option<String>,
    case_name: &'a str,
    started_at: String,
    ended_at: String,
    wall_seconds: f64,
    git_sha: Option<String>,
    git_dirty: Option<bool>,
    machine: DcMachine,
    n_cells: usize,
    n_boundary_faces: usize,
    iterations: u32,
    ashrae_class: String,
    rci_samples: &'static str,
    notes: &'a [String],
    report: ReportDoc<'a>,
    fans: Vec<FanDoc<'a>>,
    t_inlet_max: Scalar,
    rack_inlets: Vec<RackInletDoc<'a>>,
    patch_flow: Vec<PatchFlowDoc<'a>>,
    wall_patches: Vec<WallPatchDoc<'a>>,
    continuity: ContinuityDoc,
    supersaturation: Option<SupersaturationDoc>,
    caveats: Vec<CaveatDoc>,
}

/// The four numbers the outer probe found, carried out of the loop so the
/// document can say what the printed paragraph says.
#[derive(Debug, Clone)]
struct ProbeSummary {
    separable_bcs: bool,
    symmetric: bool,
    constant_coefficient: bool,
    // Carried beside its three booleans so the reason travels with them; the
    // printed paragraph and the caveat text quote it from `SystemProbe`
    // itself, so nothing reads the copy here yet.
    #[allow(dead_code)]
    non_separable_reason: String,
}

/// Everything the document needs that is not in `LoweredDcCase` or
/// `RoomSolution`: the command line's own answers and the clock.
struct DocMeta<'a> {
    run_id: &'a str,
    case_path: &'a Path,
    started_ms: i64,
    ended_ms: i64,
    device: Option<(String, String)>,
}

/// Milliseconds since the Unix epoch. The document's clock is one i64, split
/// with `div_euclid`/`rem_euclid` - never `/` and `%`, which drift on the
/// pre-1970 instants a back-dated stamp would carry.
fn epoch_ms(t: std::time::SystemTime) -> i64 {
    match t.duration_since(std::time::UNIX_EPOCH) {
        Ok(d) => d.as_millis() as i64,
        Err(e) => -(e.duration().as_millis() as i64),
    }
}

/// A civil date from a Unix day: Howard Hinnant's `civil_from_days`
/// (chrono-Compatible Low-Level Date Algorithms,
/// howardhinnant.github.io/date_algorithms.html, public domain), exact over
/// the whole proleptic Gregorian calendar. Unix time ignores leap seconds
/// and so does this.
fn civil_from_days(z: i64) -> (i64, i64, i64) {
    let z = z + 719_468;
    let era = (if z >= 0 { z } else { z - 146_096 }) / 146_097;
    let doe = z - era * 146_097; // [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365; // [0, 399]
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let d = doy - (153 * mp + 2) / 5 + 1; // [1, 31]
    let m = if mp < 10 { mp + 3 } else { mp - 9 }; // [1, 12]
    (y + i64::from(m <= 2), m, d)
}

/// `YYYY-MM-DDTHH:MM:SS.mmmZ`, always UTC.
fn iso8601_utc(epoch_ms: i64) -> String {
    let days = epoch_ms.div_euclid(86_400_000);
    let ms_of_day = epoch_ms.rem_euclid(86_400_000);
    let (y, m, d) = civil_from_days(days);
    let (hh, mm, ss, mmm) = (
        ms_of_day / 3_600_000,
        (ms_of_day % 3_600_000) / 60_000,
        (ms_of_day % 60_000) / 1000,
        ms_of_day % 1000,
    );
    format!("{y:04}-{m:02}-{d:02}T{hh:02}:{mm:02}:{ss:02}.{mmm:03}Z")
}

/// `20260915T011203Z` - the compact stamp a default run id is minted from.
fn compact_stamp(epoch_ms: i64) -> String {
    let days = epoch_ms.div_euclid(86_400_000);
    let ms_of_day = epoch_ms.rem_euclid(86_400_000);
    let (y, m, d) = civil_from_days(days);
    let (hh, mm, ss) =
        (ms_of_day / 3_600_000, (ms_of_day % 3_600_000) / 60_000, (ms_of_day % 60_000) / 1000);
    format!("{y:04}{m:02}{d:02}T{hh:02}{mm:02}{ss:02}Z")
}

/// `(gitSha, gitDirty)` read from the repository the CASE lives in - the
/// same rule and the same failure mode as `ofgpu-validate`'s probe. Any
/// failure - `git` absent, non-zero status, non-UTF-8 output - yields
/// `(None, None)`: a tree with no git yields a document whose `gitSha` and
/// `gitDirty` are `null` together, never one of them alone.
fn git_probe(dir: &Path) -> (Option<String>, Option<bool>) {
    let run = |args: &[&str]| -> Option<String> {
        let out = std::process::Command::new("git")
            .current_dir(dir)
            .args(args)
            .output()
            .ok()?;
        if !out.status.success() {
            return None;
        }
        String::from_utf8(out.stdout).ok()
    };
    let sha = run(&["rev-parse", "HEAD"]).map(|s| s.trim().to_string());
    let dirty = run(&["status", "--porcelain"]).map(|s| !s.trim().is_empty());
    match (sha, dirty) {
        (Some(sha), Some(dirty)) if !sha.is_empty() => (Some(sha), Some(dirty)),
        _ => (None, None),
    }
}

/// The words `bin/common/mod.rs`'s `device_banner` renders - copied, not
/// imported, so this binary keeps its own small dependency surface.
fn precision_word() -> &'static str {
    if std::mem::size_of::<Scalar>() == 4 {
        "float"
    } else {
        "double"
    }
}

fn machine(device: Option<&(String, String)>) -> DcMachine {
    let (dev, cc) = match device {
        Some((n, c)) => (Some(n.clone()), Some(c.clone())),
        None => (None, None),
    };
    DcMachine {
        hostname: std::env::var("COMPUTERNAME")
            .or_else(|_| std::env::var("HOSTNAME"))
            .ok(),
        os: std::env::consts::OS,
        arch: std::env::consts::ARCH,
        device: dev,
        compute_capability: cc,
        precision: precision_word(),
    }
}

/// The case's own spelling of the sample set (`thirds`/`faces`), so one
/// value has one spelling across the axis - never the Rust `Debug` form.
fn rci_samples_word(s: RciSamples) -> &'static str {
    match s {
        RciSamples::Faces => "faces",
        RciSamples::Thirds => "thirds",
    }
}

/// The whole-boundary closure of the flow-through table, computed once and
/// read by `print_report` and the document alike. Returns
/// `(net, largest_opening, net_over_largest)`; the ratio is `0.0` when no
/// opening carries flow, exactly as the print has always shown.
fn continuity_of(patch_flow: &[(String, Scalar)]) -> (Scalar, Scalar, Scalar) {
    let mut net = 0.0 as Scalar;
    let mut scale = 0.0 as Scalar;
    for (_, q) in patch_flow {
        net += *q;
        scale = scale.max(q.abs());
    }
    let ratio = if scale > 0.0 { net.abs() / scale } else { 0.0 as Scalar };
    (net, scale, ratio)
}

/// The operating point's own outer residual, in per cent - the gap the fan
/// block prints.
fn outer_residual_pct(q: Scalar, q_patch: Scalar) -> Scalar {
    100.0 * (q - q_patch).abs() / q_patch.abs().max(1e-30)
}

/// The paragraph `print_report` prints when the direct Poisson path is out
/// of reach - one const so the printed text and the document's caveat text
/// cannot drift.
const FFT_UNAVAILABLE_TEXT: &str = "the cuFFT direct Poisson backend is NOT available on this \
     system, and PBiCGStab is used instead. SPEC-LIT S52.8: a fan \
     curve makes a patch face neither uniformly Dirichlet nor \
     uniformly Neumann, and S53.2's jump makes the face coefficient \
     non-constant. That is the biggest cost of these two features and \
     it is printed, not hidden; S52.9 names the Woodbury correction \
     that would put the direct path back and says it is not built.";

/// The sentence `print_report` prints when the assembled matrix came out
/// asymmetric - same const discipline as [`FFT_UNAVAILABLE_TEXT`].
const ASYMMETRIC_MATRIX_TEXT: &str = "WARNING: the pressure matrix is NOT symmetric. SPEC-LIT S52.2 \
     and S53.2 both say it should be - a fan is a symmetric rank-1 \
     downdate and a jump divides upper and lower by the same number.";

/// The supersaturation sentence (S54.5), without the `--- S54.5 ---` banner,
/// which stays a print-side concern.
fn supersaturation_text(cells: usize, worst: Scalar) -> String {
    format!(
        "{cells} cell(s) are supersaturated, worst excess \
         {worst:.6} kg/kg. Y_v is REPORTED and not clipped: field-level \
         condensation is a different model and is refused by name."
    )
}

/// The §6.4 / EQ-B4 paragraph `print_report` prints and the document carries
/// as the `wallValidity6.4` caveat - one function so they cannot drift.
fn wall_validity_text(rows: &[WallPatchReport]) -> String {
    if rows.is_empty() {
        return String::new();
    }
    let head = format!(
        "wall-function validity per wall patch (S6.4 y+ = Cmu^(1/4) y sqrt(k_P)/nu; \
         EQ-B4 Gr/Re^2 = g beta dT L/U^2 with beta = 1/T_ref, L = {:.3} m along gravity, \
         U = the patch's area-weighted first-cell speed; forced below {}, natural above {}, \
         mixed between): ",
        rows[0].length_scale, GR_RE2_FORCED_BELOW, GR_RE2_NATURAL_ABOVE
    );
    let body: Vec<String> = rows
        .iter()
        .map(|w| {
            let gr = match w.gr_over_re2 {
                None => "not applicable (the case gives this wall no temperature)".to_string(),
                Some(x) => format!(
                    "{:.3} (dT {:.2} K, U {:.3} m/s) -> {}",
                    x,
                    w.delta_t.unwrap_or(f64::NAN),
                    w.approach_speed,
                    w.regime
                ),
            };
            format!(
                "{} y+ min {:.2} | mean {:.2} | max {:.2}, Gr/Re^2 {}",
                w.patch, w.y_plus_min, w.y_plus_mean, w.y_plus_max, gr
            )
        })
        .collect();
    format!("{head}{}", body.join("; "))
}

/// The six caveat kinds and the section each one's text answers to, in the
/// document's fixed order. A kind outside these six does not exist. Both
/// spellings live in this one table so the document and its tests read the
/// same six pairs.
const CAVEATS: [(&str, &str); 6] = [
    ("porousJump53.6", "S53.6"),
    ("molarMass54.4", "S54.4"),
    ("supersaturation54.5", "S54.5"),
    ("fftUnavailable52.8", "S52.8"),
    ("asymmetricMatrix52.2", "S52.2"),
    ("wallValidity6.4", "S6.4"),
];

/// The caveats the run earned, in the document's fixed order, each only when
/// it fired. Six kinds exist; no seventh kind does.
fn caveats_of(s: &RoomSolution) -> Vec<CaveatDoc> {
    let mut out = Vec::new();
    let mut push = |which: usize, text: String| {
        let (kind, spec_ref) = CAVEATS[which];
        out.push(CaveatDoc { kind, spec_ref, text });
    };
    if let Some(c) = &s.jump_caveat {
        push(0, c.clone());
    }
    if let Some(c) = &s.molar_caveat {
        push(1, c.clone());
    }
    if let Some((cells, worst)) = s.supersaturation {
        if cells > 0 {
            push(2, supersaturation_text(cells, worst));
        }
    }
    if let Some(p) = &s.probe {
        if !p.separable_bcs || !p.constant_coefficient {
            push(3, FFT_UNAVAILABLE_TEXT.to_string());
        }
        if !p.symmetric {
            push(4, ASYMMETRIC_MATRIX_TEXT.to_string());
        }
    }
    if !s.wall_patches.is_empty() {
        push(5, wall_validity_text(&s.wall_patches));
    }
    out
}

/// Assemble the document from what the run already produced. It borrows:
/// the allocations are the three `Vec`s, the four `String`s and the caveats.
fn build_document<'a>(
    lc: &'a LoweredDcCase,
    s: &'a RoomSolution,
    m: &'a DocMeta<'a>,
) -> DcReportDoc<'a> {
    let (net, largest_opening, net_over_largest) = continuity_of(&s.patch_flow);
    let (git_sha, git_dirty) = git_probe(m.case_path.parent().unwrap_or(Path::new(".")));
    let fans = s
        .fans
        .iter()
        .map(|(patch, q, dp, w)| {
            let outer = s
                .patch_flow
                .iter()
                .find(|(n, _)| n == patch)
                .map(|(_, qp)| outer_residual_pct(*q, *qp));
            FanDoc {
                patch,
                q: *q,
                dp: *dp,
                shaft_power: *w,
                outer_residual_pct: outer,
            }
        })
        .collect();
    DcReportDoc {
        schema: "ofgpu-datacentre/1",
        run_id: m.run_id,
        // The string the writer already holds: no absolutisation and no
        // canonicalisation, only the path separator made portable.
        case_path: m.case_path.display().to_string().replace('\\', "/"),
        // No digest of the case bytes is computed in Rust in this version;
        // the key stays present and null so a later one can fill it.
        case_sha256: None,
        case_name: &lc.name,
        started_at: iso8601_utc(m.started_ms),
        ended_at: iso8601_utc(m.ended_ms),
        wall_seconds: (m.ended_ms - m.started_ms) as f64 / 1000.0,
        git_sha,
        git_dirty,
        machine: machine(m.device.as_ref()),
        n_cells: lc.mesh.n_cells,
        n_boundary_faces: lc.mesh.n_boundary_faces,
        iterations: lc.run.iterations,
        ashrae_class: format!("{:?}", lc.class),
        rci_samples: rci_samples_word(lc.samples),
        notes: &lc.notes,
        report: ReportDoc {
            rci_hi: s.report.rci_hi,
            rci_lo: s.report.rci_lo,
            n_samples: s.report.n_samples,
            rti: s.report.rti,
            shi: s.report.shi,
            rhi: s.report.rhi,
            t_supply: s.report.t_supply,
            t_return: s.report.t_return,
            dt_equipment: s.report.dt_equipment,
            dt_measured: s.report.dt_measured,
            pue: PueDoc {
                fan_power: s.report.pue.fan_power,
                fan_power_each: &s.report.pue.fan_power_each,
                it_heat: s.report.pue.it_heat,
                free_cooling_ceiling: s.report.pue.free_cooling_ceiling,
                supply_temperature_sweep: s.sweep.as_ref().map(|sw| {
                    sw.points
                        .iter()
                        .map(|p| SweepPointDoc {
                            t_supply_set: p.t_set,
                            t_supply: p.t_supply,
                            rci_hi: p.rci_hi,
                            rci_lo: p.rci_lo,
                        })
                        .collect()
                }),
                sweep_monotone: s.sweep.as_ref().map(|sw| sw.monotone),
            },
        },
        fans,
        t_inlet_max: s.t_inlet_max,
        rack_inlets: s
            .rack_inlets
            .iter()
            .map(|(name, t)| RackInletDoc { name, inlet_t: *t })
            .collect(),
        patch_flow: s
            .patch_flow
            .iter()
            .map(|(patch, q)| PatchFlowDoc { patch, q: *q })
            .collect(),
        wall_patches: s
            .wall_patches
            .iter()
            .map(|w| WallPatchDoc {
                patch: &w.patch,
                y_plus_min: w.y_plus_min,
                y_plus_mean: w.y_plus_mean,
                y_plus_max: w.y_plus_max,
                delta_t: w.delta_t,
                approach_speed: w.approach_speed,
                length_scale: w.length_scale,
                gr_over_re2: w.gr_over_re2,
                regime: w.regime,
            })
            .collect(),
        continuity: ContinuityDoc {
            net,
            largest_opening,
            net_over_largest,
        },
        supersaturation: s
            .supersaturation
            .map(|(cells, worst)| SupersaturationDoc { cells, worst_excess: worst }),
        caveats: caveats_of(s),
    }
}

/// Write the document: parent directory first, then the pretty JSON and one
/// trailing newline. Every failure becomes the same `Error::Parse` the CSV
/// writer raises, so the driver has one file-writing failure shape.
fn write_document(path: &Path, doc: &DcReportDoc) -> Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).map_err(|e| ofgpu::error::Error::Parse {
                path: path.display().to_string(),
                msg: e.to_string(),
            })?;
        }
    }
    let mut text = serde_json::to_string_pretty(doc).map_err(|e| ofgpu::error::Error::Parse {
        path: path.display().to_string(),
        msg: e.to_string(),
    })?;
    text.push('\n');
    std::fs::write(path, text).map_err(|e| ofgpu::error::Error::Parse {
        path: path.display().to_string(),
        msg: e.to_string(),
    })
}

// ==========================================================================
//  6. The tests - host only, one fixture
// ==========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// The 26 keys of the document, in document order. Order is asserted on
    /// the PRETTY TEXT, never on the parsed object: `serde_json::Map` sorts.
    const TOP_KEYS: [&str; 26] = [
        "schema", "runId", "casePath", "caseSha256", "caseName", "startedAt", "endedAt",
        "wallSeconds", "gitSha", "gitDirty", "machine", "nCells", "nBoundaryFaces", "iterations",
        "ashraeClass", "rciSamples", "notes", "report", "fans", "tInletMax", "rackInlets",
        "patchFlow", "wallPatches", "continuity", "supersaturation", "caveats",
    ];

    /// The smallest complete case this format accepts. Transcribed from
    /// `crate::io::case_dc::tests::BASE`, which lives in a `#[cfg(test)]`
    /// module of the library and so cannot be imported from a binary.
    /// `"name"` is the only change.
    const CASE: &str = r#"{
  "name": "d1",
  "room": {
    "bounds": { "min": [0,0,0], "max": [2.0, 1.0, 1.5] },
    "cells": [8, 4, 6],
    "boundaries": {
      "xMin": "west", "xMax": "east",
      "yMin": "south", "yMax": "north",
      "zMin": "supply", "zMax": "ret"
    }
  },
  "air": { "nu": 1.5e-5, "rho": 1.2, "cp": 1005, "pr": 0.71, "prt": 0.85,
           "tRef": 295.15, "gravity": [0,0,-9.81] },
  "fans": [
    { "patch": "ret", "direction": "outflow",
      "curve": { "type": "quadratic", "dpMax": 8.0, "QMax": 1.0,
                 "rhoCurve": 1.2, "speedCurve": 1.0, "speed": 1.0,
                 "efficiency": 0.62 },
      "ambientPressure": 0.0, "relaxation": 0.5 }
  ],
  "tiles": [
    { "patch": "supply", "K": 300.0, "plenumPressure": 4.0,
      "plenumTemperature": 291.15, "plenumRelativeHumidity": 0.45 }
  ],
  "racks": [
    { "name": "r1", "zone": { "min": [0.8,0.2,0.1], "max": [1.2,0.8,1.0] },
      "power": 2000.0, "flow": 0.15,
      "inletSamples": { "min": [0.5,0.2,0.1], "max": [0.8,0.8,1.0] } }
  ],
  "patches": [
    { "patch": "west", "kind": "adiabaticWall" },
    { "patch": "east", "kind": "adiabaticWall" },
    { "patch": "south", "kind": "adiabaticWall" },
    { "patch": "north", "kind": "adiabaticWall" }
  ],
  "humidity": { "d": 2.5e-5, "scT": 0.7, "barometricPressure": 101325.0,
                "virtualTemperature": true },
  "metrics": { "ashraeClass": "A1", "rciSamples": "thirds",
               "supplyPatch": "supply", "returnPatch": "ret" },
  "run": { "iterations": 20, "reportEvery": 0, "initialTemperature": 295.15 },
  "numerics": { "uRelax": 0.7, "pRelax": 0.3, "tRelax": 0.7,
                "tolerance": 1e-8, "maxIterations": 200 }
}"#;

    fn lowered() -> LoweredDcCase {
        DcCase::parse(CASE, "d1").unwrap().lower().unwrap()
    }

    /// The all-firing fixture: every number the tests expect below is one of
    /// these. The `patch_flow` sum is a tiny non-zero residue in binary on
    /// purpose - the closure is a real number, and `continuity_of` is the
    /// only thing allowed to compute it.
    fn solution() -> RoomSolution {
        RoomSolution {
            report: MetricReport {
                rci_hi: 91.733,
                rci_lo: 100.0,
                n_samples: 6,
                rti: 80.576,
                shi: 0.354185,
                rhi: 0.645815,
                t_supply: 291.15,
                t_return: 299.8,
                dt_equipment: 10.7349,
                dt_measured: false,
                pue: PueInputs {
                    fan_power: 19.819,
                    fan_power_each: vec![19.819],
                    it_heat: 14500.0,
                    free_cooling_ceiling: None,
                },
            },
            fans: vec![("ret".to_string(), 2.217, 5.5425, 19.819)],
            jump_caveat: Some("a jump caveat".to_string()),
            t_inlet_max: 301.42,
            rack_inlets: vec![("r1".to_string(), 298.6)],
            supersaturation: Some((3, 0.0004)),
            molar_caveat: Some("a molar caveat".to_string()),
            patch_flow: vec![
                ("supply".to_string(), -1.390),
                ("south".to_string(), -0.827),
                ("ret".to_string(), 2.217),
                ("west".to_string(), 0.0),
                ("east".to_string(), 0.0),
                ("north".to_string(), 0.0),
            ],
            probe: Some(ProbeSummary {
                separable_bcs: false,
                symmetric: false,
                constant_coefficient: false,
                non_separable_reason: "a fan patch".to_string(),
            }),
            sweep: None,
            // D5: one adiabatic wall (no dT, so no Gr/Re^2) and one heated
            // wall, so both JSON shapes have a row. The heated row's ratio is
            // the seed's hall example at this fixture's 1.5 m height and
            // 0.2 m/s.
            wall_patches: vec![
                WallPatchReport {
                    patch: "south".to_string(),
                    y_plus_min: 30.5,
                    y_plus_mean: 32.25,
                    y_plus_max: 34.0,
                    delta_t: None,
                    approach_speed: 0.35,
                    length_scale: 1.5,
                    gr_over_re2: None,
                    regime: "notApplicable",
                },
                WallPatchReport {
                    patch: "heated".to_string(),
                    y_plus_min: 98.0,
                    y_plus_mean: 101.5,
                    y_plus_max: 105.0,
                    delta_t: Some(7.2),
                    approach_speed: 0.2,
                    length_scale: 1.5,
                    gr_over_re2: Some(gr_over_re2(9.81, 1.0 / 295.15, 7.2, 1.5, 0.2)),
                    regime: "mixed",
                },
            ],
        }
    }

    /// The pretty document text for a fixture, with a fixed clock and case
    /// path so the tests are deterministic.
    fn doc_text(lc: &LoweredDcCase, sol: &RoomSolution, run_id: Option<&str>) -> String {
        let rid = run_id
            .map(|s| s.to_string())
            .unwrap_or_else(|| format!("v_{}", compact_stamp(0)));
        let meta = DocMeta {
            run_id: &rid,
            case_path: Path::new("d1.dc.jsonc"),
            started_ms: 0,
            ended_ms: 0,
            device: None,
        };
        serde_json::to_string_pretty(&build_document(lc, sol, &meta)).unwrap()
    }

    fn value(sol: &RoomSolution, run_id: Option<&str>) -> serde_json::Value {
        let lc = lowered();
        serde_json::from_str(&doc_text(&lc, sol, run_id)).unwrap()
    }

    // ---- D5: §6.4 y+ and the EQ-B4 ratio --------------------------------

    #[test]
    fn y_plus_is_the_low_mach_expression() {
        // The hand value for §6.4's y+ = C_mu^(1/4) y sqrt(k_P) / nu.
        let got = y_plus(0.01, 0.05, 1.5e-5, 0.09);
        assert!((got - 182.574).abs() / 182.574 < 1e-3, "{got}");
        // Bit for bit against the inline expression `lowmach.rs` compiles,
        // on three hand triples.
        let triples: [(f64, f64, f64, f64); 3] = [
            (0.0625, 3.0e-4, 1.5e-5, 0.09),
            (1.0, 0.01, 1.0e-5, 0.09),
            (0.0, 0.05, 1.5e-5, 0.09),
        ];
        for &(k_p, y, nu, cmu) in &triples {
            let inline = cmu.powf(0.25) * y * k_p.max(0.0).sqrt() / nu;
            assert_eq!(y_plus(k_p, y, nu, cmu).to_bits(), inline.to_bits());
        }
        // A negative k clamps, exactly as the `.max(0.0)` in `lowmach.rs`.
        assert_eq!(y_plus(-1.0, 0.05, 1.5e-5, 0.09), 0.0);
    }

    #[test]
    fn gr_over_re2_reproduces_the_seed_examples() {
        // EQ-B4's seed prose: "A hall at 2 m/s gives about 0.18 and a
        // stalled aisle at 0.2 m/s about 18".
        let hall = gr_over_re2(9.81, 1.0 / 295.15, 7.2, 3.0, 2.0);
        assert!((hall - 0.18).abs() / 0.18 < 0.01, "{hall}");
        let stalled = gr_over_re2(9.81, 1.0 / 295.15, 7.2, 3.0, 0.2);
        assert!((stalled - 18.0).abs() / 18.0 < 0.01, "{stalled}");
    }

    #[test]
    fn the_regime_word_switches_at_a_tenth_and_at_ten() {
        assert_eq!(regime_word(0.0999), "forced");
        assert_eq!(regime_word(0.1), "mixed");
        assert_eq!(regime_word(0.18), "mixed");
        assert_eq!(regime_word(10.0), "mixed");
        assert_eq!(regime_word(10.001), "natural");
        assert_eq!(regime_word(18.0), "natural");
    }

    #[test]
    fn wall_coefficients_agree_with_the_model() {
        // §15.6: one C_mu in the model and at the wall (both 0.09 today,
        // `io/case.rs` and `k_epsilon.rs`).
        assert_eq!(
            wide(wall_coeffs().cmu),
            wide(KEpsilonCoeffs::default().cmu)
        );
    }

    #[test]
    fn json_wall_numbers_equal_the_computed_ones_bit_for_bit() {
        let s = solution();
        let v = value(&s, Some("r_1"));
        let rows = v["wallPatches"].as_array().unwrap();
        assert_eq!(rows.len(), 2);
        let heated = &s.wall_patches[1];
        assert_eq!(
            rows[1]["yPlusMean"].as_f64().unwrap().to_bits(),
            heated.y_plus_mean.to_bits()
        );
        assert_eq!(
            rows[1]["grOverRe2"].as_f64().unwrap().to_bits(),
            heated.gr_over_re2.unwrap().to_bits()
        );
        // The adiabatic row carries nulls, never a stale zero.
        assert!(rows[0]["grOverRe2"].is_null());
    }

    #[test]
    fn an_adiabatic_wall_has_no_gr_over_re2() {
        let text = wall_validity_text(&solution().wall_patches);
        assert!(text.contains("not applicable (the case gives this wall no temperature)"));
        // The heated row's regime word - with the arrow, so the prefix's
        // "mixed between" cannot satisfy it.
        assert!(text.contains("-> mixed"));
        // The conventions are named so a reader can recompute the ratio.
        assert!(text.contains("beta = 1/T_ref"));
        assert!(text.contains("L = 1.500 m along gravity"));
        assert!(text.contains("U = the patch's area-weighted first-cell speed"));
    }

    #[test]
    fn the_document_has_the_twenty_six_keys_in_order() {
        let text = doc_text(&lowered(), &solution(), Some("r_1"));
        let v: serde_json::Value = serde_json::from_str(&text).unwrap();
        assert_eq!(v.as_object().unwrap().len(), 26);
        let nl = '\n';
        let mut last = 0;
        for k in TOP_KEYS {
            let at = text
                .find(&format!("{nl}  \"{k}\":"))
                .unwrap_or_else(|| panic!("top-level key `{k}` missing from the document"));
            assert!(at > last, "top-level key `{k}` is out of order");
            last = at;
        }
        // Each wallPatches entry carries its nine keys in document order too,
        // read off the pretty text like the top level.
        let mut last = 0;
        for k in [
            "patch", "yPlusMin", "yPlusMean", "yPlusMax", "deltaT", "approachSpeed",
            "lengthScale", "grOverRe2", "regime",
        ] {
            let at = text
                .find(&format!("{nl}      \"{k}\":"))
                .unwrap_or_else(|| panic!("wallPatches key `{k}` missing from the document"));
            assert!(at > last, "wallPatches key `{k}` is out of order");
            last = at;
        }
    }

    #[test]
    fn every_absent_value_is_null_not_a_missing_key() {
        let mut s = solution();
        s.jump_caveat = None;
        s.molar_caveat = None;
        s.supersaturation = None;
        s.probe = None;
        // The wall table is a caveat trigger too, so the zero-caveat
        // assertion below wants it empty; its nulls are checked further
        // down, on the all-firing fixture.
        s.wall_patches = Vec::new();
        s.fans[0].0 = "absent".to_string();
        let v = value(&s, None);
        for k in ["caseSha256", "supersaturation"] {
            assert!(v[k].is_null(), "{k} must be null, never omitted");
        }
        assert!(v["machine"]["device"].is_null());
        assert!(v["machine"]["computeCapability"].is_null());
        assert!(v["report"]["pue"]["freeCoolingCeiling"].is_null());
        assert!(v["report"]["pue"]["supplyTemperatureSweep"].is_null());
        assert!(v["report"]["pue"]["sweepMonotone"].is_null());
        assert!(v["fans"][0]["outerResidualPct"].is_null());
        assert_eq!(v["caveats"].as_array().unwrap().len(), 0);
        for k in TOP_KEYS {
            assert!(v.get(k).is_some(), "key `{k}` is missing");
        }

        // D5: the wall table's absent numbers are nulls with every key
        // present, on the all-firing fixture whose first row is adiabatic.
        let full = value(&solution(), None);
        let rows = full["wallPatches"].as_array().unwrap();
        assert_eq!(rows.len(), 2);
        for row in rows {
            for k in [
                "patch", "yPlusMin", "yPlusMean", "yPlusMax", "deltaT", "approachSpeed",
                "lengthScale", "grOverRe2", "regime",
            ] {
                assert!(row.get(k).is_some(), "wallPatches key `{k}` is missing");
            }
        }
        assert!(rows[0]["deltaT"].is_null());
        assert!(rows[0]["grOverRe2"].is_null());
        assert!(rows[1]["deltaT"].as_f64().is_some());

        // A run with no wall: an empty table and no wallValidity6.4 caveat.
        let mut bare = solution();
        bare.wall_patches = Vec::new();
        let v = value(&bare, None);
        assert_eq!(v["wallPatches"].as_array().unwrap().len(), 0);
        assert!(!v["caveats"]
            .as_array()
            .unwrap()
            .iter()
            .any(|c| c["kind"].as_str().unwrap() == "wallValidity6.4"));
    }

    #[test]
    fn json_rci_hi_equals_the_printed_one_bit_for_bit() {
        let s = solution();
        let v = value(&s, Some("r_1"));
        let json_hi = v["report"]["rciHi"].as_f64().unwrap();
        #[cfg(not(feature = "single"))]
        assert_eq!(json_hi.to_bits(), f64::from(s.report.rci_hi).to_bits());
        assert_eq!(
            format!("{json_hi:8.3}"),
            format!("{:8.3}", f64::from(s.report.rci_hi))
        );
    }

    /// A swept `t_set` of one degree above the base's 295.15-ish fixture
    /// values, so the points read like the shipped case's kelvin grid.
    fn sp(t_set: Scalar, rci_hi: Scalar) -> SweepPoint {
        SweepPoint {
            t_set,
            t_supply: t_set + 2.0,
            rci_hi,
            rci_lo: 100.0,
        }
    }

    #[test]
    fn the_ceiling_is_the_highest_swept_temperature_holding_exactly_one_hundred() {
        let pts = vec![
            sp(285.15, 100.0),
            sp(287.15, 100.0),
            sp(289.15, 99.999_999_9),
            sp(291.15, 97.0),
        ];
        assert_eq!(ceiling_of(&pts), Some(287.15), "99.999_999_9 is not exactly 100");
        let all_hold: Vec<SweepPoint> = [285.15, 287.15, 289.15]
            .iter()
            .map(|t| sp(*t, 100.0))
            .collect();
        assert_eq!(ceiling_of(&all_hold), Some(289.15), "every point holds: the last");
        let none_hold: Vec<SweepPoint> = [285.15, 287.15].iter().map(|t| sp(*t, 99.9)).collect();
        assert_eq!(ceiling_of(&none_hold), None, "nothing holds: no ceiling");
        let mut shuffled = pts.clone();
        shuffled.swap(0, 3);
        shuffled.swap(1, 2);
        assert_eq!(ceiling_of(&shuffled), Some(287.15), "the rule is order-independent");
    }

    #[test]
    fn the_sweep_verdict_tolerates_round_off_and_flags_a_rise() {
        let m = |rcis: &[Scalar]| {
            let pts: Vec<SweepPoint> = rcis
                .iter()
                .enumerate()
                .map(|(i, r)| sp(285.15 + i as Scalar, *r))
                .collect();
            sweep_is_monotone(&pts)
        };
        assert!(m(&[100.0, 100.0, 98.0, 95.0]), "a fall is never a rise");
        assert!(!m(&[100.0, 98.0, 99.0]), "a real rise is flagged");
        assert!(m(&[100.0, 100.0 + 5e-7]), "two solves at 100 differ by round-off only");
        assert!(!m(&[100.0, 100.0 + 5e-6]), "beyond round-off is a rise");
    }

    #[test]
    fn the_sweep_block_names_the_ceiling_the_document_carries() {
        let mut s = solution();
        s.report.pue.free_cooling_ceiling = Some(287.15);
        s.sweep = Some(SweepResult {
            points: vec![sp(283.15, 100.0), sp(287.15, 100.0), sp(291.15, 97.5)],
            monotone: true,
        });
        let lines = sweep_lines(&s);
        assert!(
            lines.iter().any(|l| l.contains("287.15 K (14.00 C)")),
            "the printed ceiling is the document's: {lines:?}"
        );
        assert!(
            lines.iter().any(|l| l.contains("monotone in T_set: yes")),
            "{lines:?}"
        );
        s.sweep = None;
        assert!(sweep_lines(&s).is_empty(), "no sweep, no block");
    }

    #[test]
    fn a_swept_solution_carries_its_table_and_ceiling_bit_for_bit() {
        let mut s = solution();
        s.report.pue.free_cooling_ceiling = Some(287.15);
        s.sweep = Some(SweepResult {
            points: vec![sp(285.15, 100.0), sp(289.15, 98.25), sp(293.15, 91.7)],
            monotone: true,
        });
        let text = doc_text(&lowered(), &s, Some("r_1"));
        let v: serde_json::Value = serde_json::from_str(&text).unwrap();
        let json_k = v["report"]["pue"]["freeCoolingCeiling"].as_f64().unwrap();
        #[cfg(not(feature = "single"))]
        assert_eq!(
            json_k.to_bits(),
            f64::from(s.report.pue.free_cooling_ceiling.unwrap()).to_bits()
        );
        let table = v["report"]["pue"]["supplyTemperatureSweep"]
            .as_array()
            .expect("the sweep table is an array");
        assert_eq!(table.len(), 3);
        assert_eq!(v["report"]["pue"]["sweepMonotone"], serde_json::json!(true));
        // The four keys, in document order, on the PRETTY TEXT - parsed
        // objects sort, so the order is asserted on the text itself.
        let anchor = text.find("\"supplyTemperatureSweep\"").expect("the table key");
        let mut at = anchor;
        for k in ["tSupplySet", "tSupply", "rciHi", "rciLo"] {
            let at_k = text[at..]
                .find(&format!("\"{k}\":"))
                .unwrap_or_else(|| panic!("key `{k}` missing from the first row"))
                + at;
            assert!(at_k > at, "key `{k}` out of order");
            at = at_k;
        }
    }

    #[test]
    fn the_shipped_case_declares_a_sweep_the_reader_turns_into_five_solves() {
        let c = DcCase::read(Path::new("../cases/coldAisle.dc.jsonc"))
            .expect("the shipped case parses");
        let lc = c.lower().expect("the shipped case lowers");
        let sw = lc
            .supply_sweep
            .as_ref()
            .expect("the shipped case declares a sweep");
        assert_eq!(sw.patch, "floorSupply");
        assert_eq!(sw.temperatures.len(), 5, "285.15 to 293.15 step 2");
        let r = |a: Scalar, b: Scalar| (a - b).abs() / a.abs().max(b.abs()).max(1e-300);
        assert!(r(sw.temperatures[0], 285.15) < 1e-12);
        assert!(r(sw.temperatures[4], 293.15) < 1e-12);
    }

    #[test]
    fn continuity_is_one_computation_printed_and_serialised() {
        let s = solution();
        let v = value(&s, None);
        let (n, l, r) = continuity_of(&s.patch_flow);
        assert_eq!(
            v["continuity"]["net"].as_f64().unwrap().to_bits(),
            f64::from(n).to_bits()
        );
        assert_eq!(
            v["continuity"]["largestOpening"].as_f64().unwrap().to_bits(),
            f64::from(l).to_bits()
        );
        assert_eq!(
            v["continuity"]["netOverLargest"].as_f64().unwrap().to_bits(),
            f64::from(r).to_bits()
        );
        assert_eq!(l, 2.217);
        assert!(n.abs() < 1e-15);
    }

    #[test]
    fn every_caveat_text_is_the_printed_one() {
        let s = solution();
        let v = value(&s, None);
        let caveats = v["caveats"].as_array().unwrap();
        assert_eq!(
            caveats[0]["text"].as_str().unwrap(),
            s.jump_caveat.as_deref().unwrap()
        );
        assert_eq!(
            caveats[1]["text"].as_str().unwrap(),
            s.molar_caveat.as_deref().unwrap()
        );
        assert_eq!(
            caveats[2]["text"].as_str().unwrap(),
            supersaturation_text(3, 0.0004)
        );
        assert_eq!(caveats[3]["text"].as_str().unwrap(), FFT_UNAVAILABLE_TEXT);
        assert_eq!(
            caveats[4]["text"].as_str().unwrap(),
            ASYMMETRIC_MATRIX_TEXT
        );
        assert_eq!(
            caveats[5]["text"].as_str().unwrap(),
            wall_validity_text(&s.wall_patches)
        );
    }

    #[test]
    fn the_six_caveat_kinds_are_spelled_exactly_once_each() {
        let v = value(&solution(), None);
        let caveats = v["caveats"].as_array().unwrap();
        assert_eq!(caveats.len(), CAVEATS.len());
        for (c, (kind, spec_ref)) in caveats.iter().zip(CAVEATS) {
            assert_eq!(c["kind"].as_str().unwrap(), kind);
            assert_eq!(c["specRef"].as_str().unwrap(), spec_ref);
        }
        // The spelling itself, against the contract and not only against the
        // table that produced it: each kind ends in its section number, and
        // the specRef is that number with the S in front.
        assert_eq!(
            CAVEATS.map(|(k, _)| k),
            [
                "porousJump53.6",
                "molarMass54.4",
                "supersaturation54.5",
                "fftUnavailable52.8",
                "asymmetricMatrix52.2",
                "wallValidity6.4"
            ]
        );
        for (k, r) in CAVEATS {
            let number = k.trim_start_matches(|c: char| c.is_ascii_alphabetic());
            assert!(!number.is_empty() && number.contains('.'), "{k}");
            assert_eq!(r, format!("S{number}"));
        }

        let mut nothing = solution();
        nothing.jump_caveat = None;
        nothing.molar_caveat = None;
        nothing.supersaturation = None;
        nothing.probe = None;
        nothing.wall_patches = Vec::new();
        assert_eq!(value(&nothing, None)["caveats"].as_array().unwrap().len(), 0);

        // `cells == 0` fires no caveat but keeps the object: the case DID ask
        // for humidity, so the key says so with a zero count.
        let mut dry = solution();
        dry.jump_caveat = None;
        dry.molar_caveat = None;
        dry.probe = None;
        dry.wall_patches = Vec::new();
        dry.supersaturation = Some((0, 0.0));
        let v2 = value(&dry, None);
        assert_eq!(v2["caveats"].as_array().unwrap().len(), 0);
        assert!(v2["supersaturation"].is_object());
    }

    #[test]
    fn the_sample_set_is_spelled_the_way_the_case_spells_it() {
        let s = solution();
        let v = value(&s, None);
        assert_eq!(v["rciSamples"], "thirds");
        assert_eq!(v["ashraeClass"], "A1");

        let faces_case = CASE.replacen("\"thirds\"", "\"faces\"", 1);
        assert_ne!(faces_case, CASE);
        let lc = DcCase::parse(&faces_case, "d1").unwrap().lower().unwrap();
        let vf: serde_json::Value = serde_json::from_str(&doc_text(&lc, &s, None)).unwrap();
        assert_eq!(vf["rciSamples"], "faces");

        let a2_case = CASE.replacen("\"A1\"", "\"A2\"", 1);
        assert_ne!(a2_case, CASE);
        let lc2 = DcCase::parse(&a2_case, "d1").unwrap().lower().unwrap();
        let v2: serde_json::Value = serde_json::from_str(&doc_text(&lc2, &s, None)).unwrap();
        assert_eq!(v2["ashraeClass"], "A2");

        for w in ["Thirds", "Faces"] {
            assert_ne!(v["rciSamples"].as_str().unwrap(), w);
            assert_ne!(vf["rciSamples"].as_str().unwrap(), w);
        }
    }

    #[test]
    fn iso8601_utc_renders_three_known_instants() {
        assert_eq!(iso8601_utc(0), "1970-01-01T00:00:00.000Z");
        // What `div_euclid` buys: the instant BEFORE the epoch renders whole.
        assert_eq!(iso8601_utc(-1), "1969-12-31T23:59:59.999Z");
        assert_eq!(iso8601_utc(951_782_400_000), "2000-02-29T00:00:00.000Z");
    }

    #[test]
    fn compact_stamp_drops_the_separators_and_the_milliseconds() {
        assert_eq!(compact_stamp(951_782_400_000), "20000229T000000Z");
    }

    #[test]
    fn the_run_id_is_the_flag_else_the_stamp() {
        let v = value(&solution(), Some("r_412"));
        assert_eq!(v["runId"], "r_412");
        // doc_text stamps runs with `started_ms = 0`, so the default id is
        // the compact stamp of the epoch.
        let v2 = value(&solution(), None);
        assert_eq!(v2["runId"], "v_19700101T000000Z");
    }

    #[test]
    fn a_non_finite_metric_serialises_as_null() {
        let mut s = solution();
        s.report.rci_hi = Scalar::NAN;
        let v = value(&s, None);
        assert!(v["report"]["rciHi"].is_null());
        let path = std::env::temp_dir().join("d1_nan_test.json");
        let rid = "r_nan".to_string();
        let meta = DocMeta {
            run_id: &rid,
            case_path: Path::new("d1.dc.jsonc"),
            started_ms: 0,
            ended_ms: 0,
            device: None,
        };
        let lc = lowered();
        write_document(&path, &build_document(&lc, &s, &meta)).unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        let w: serde_json::Value = serde_json::from_str(&text).unwrap();
        assert!(w["report"]["rciHi"].is_null());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn the_document_is_written_whole_and_rereads() {
        let path = std::env::temp_dir().join("d1_doc_test.json");
        let lc = lowered();
        let rid = "r_1".to_string();
        let meta = DocMeta {
            run_id: &rid,
            case_path: Path::new("d1.dc.jsonc"),
            started_ms: 0,
            ended_ms: 0,
            device: None,
        };
        write_document(&path, &build_document(&lc, &solution(), &meta)).unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(text.ends_with('\n'));
        let v: serde_json::Value = serde_json::from_str(&text).unwrap();
        assert_eq!(v["schema"], "ofgpu-datacentre/1");
        std::fs::remove_file(&path).unwrap();
    }

    /// The test that GENERATES the shipped schema: written once here, then
    /// only ever compared against, never hand-edited.
    #[test]
    fn dc_schema_writes_to_docs_schema_directory() {
        let out_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../docs/schema");
        std::fs::create_dir_all(&out_dir).expect("create docs/schema");
        let out_path = out_dir.join("dc-1.json");
        std::fs::write(&out_path, ofgpu::io::case_dc::emit_dc_schema())
            .expect("write docs/schema/dc-1.json");
        assert!(out_path.exists());
    }

    #[test]
    fn the_shipped_dc_schema_is_the_generated_one() {
        let path =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../docs/schema/dc-1.json");
        let shipped = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
        let norm = |s: &str| s.replace("\r\n", "\n").trim_end().to_string();
        assert_eq!(
            norm(&shipped),
            norm(&ofgpu::io::case_dc::emit_dc_schema()),
            "docs/schema/dc-1.json has drifted from emit_dc_schema(); regenerate it"
        );
    }

    #[test]
    fn the_schema_names_every_key_the_shipped_case_uses() {
        let text = ofgpu::io::case_dc::emit_dc_schema();
        let schema: serde_json::Value = serde_json::from_str(&text).unwrap();
        let case =
            DcCase::read(Path::new("../cases/coldAisle.dc.jsonc")).expect("the shipped case parses");
        case.lower().expect("the shipped case lowers");
        let used = serde_json::to_value(&case).unwrap();
        let props = schema["properties"].as_object().expect("schema properties");
        for k in used.as_object().unwrap().keys() {
            assert!(props.contains_key(k), "the schema does not name `{k}`");
        }
        assert!(
            schema.get("$defs").or_else(|| schema.get("definitions")).is_some(),
            "the nested case types must be defined somewhere"
        );
        for word in [
            "supplyTemperatureSweep",
            "ashraeClass",
            "plenumPressure",
            "inletSamples",
            "openAreaRatio",
            "baffle",
        ] {
            assert!(text.contains(word), "the schema must document '{word}'");
        }
    }

    #[test]
    fn a_misspelt_sweep_key_is_refused_naming_the_key() {
        let bad = CASE.replacen(
            "\"metrics\": {",
            "\"metrics\": { \"supplyTemperatureSwep\": [10.0, 20.0, 5.0],",
            1,
        );
        assert_ne!(bad, CASE);
        let err = DcCase::parse(&bad, "d1").unwrap_err().to_string();
        assert!(err.contains("supplyTemperatureSwep"), "{err}");
        assert!(err.contains("metrics"), "{err}");
        let schema: serde_json::Value =
            serde_json::from_str(&ofgpu::io::case_dc::emit_dc_schema()).unwrap();
        let defs = schema
            .get("$defs")
            .or_else(|| schema.get("definitions"))
            .expect("schema definitions");
        assert_eq!(
            defs["DcMetricsSpec"]["additionalProperties"],
            serde_json::json!(false)
        );
    }
}
