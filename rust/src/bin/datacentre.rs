// meteor-cfd - Copyright (c) 2026 주식회사 메테오시뮬레이션 (Meteo Simulation Co., Ltd.)
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
//! ofgpu-datacentre <case.jsonc> [-csv <out.csv>] [-permissive]
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
//!
//! Where the case carries a porous jump it also prints §53.6's caveat, naming
//! what a pressure-jump tile gets wrong.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use ofgpu::dcmetrics::{
    dt_equipment_from_heat, rci_hi, rci_lo, rti, shi_rhi, MetricReport, Metrics, PueInputs,
};
use ofgpu::error::Result;
use ofgpu::fan::FlowDevices;
use ofgpu::field::{BcKind, GpuScalarField};
use ofgpu::io::case_dc::{DcCase, LoweredDcCase};
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
ofgpu-datacentre <case.jsonc> [-csv <out.csv>] [-permissive]

  SPEC-LIT S52 to S55: a data-centre room with fan curves, porous-jump tiles,
  humidity and the RCI/RTI/SHI metrics a customer report must contain.

  -csv <path>     write the metric history
  -permissive     downgrade unsupported-setting errors to warnings";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    let mut case_path: Option<PathBuf> = None;
    let mut csv: Option<PathBuf> = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
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

    match run(&case_path, csv.as_deref()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("\nofgpu-datacentre: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run(case_path: &Path, csv: Option<&Path>) -> Result<()> {
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
    let sol = solve(&gpu, &lowered)?;

    print_report(&lowered, &sol);

    if let Some(p) = csv {
        write_csv(p, &lowered, &sol)?;
        println!("\nwrote {}", p.display());
    }
    Ok(())
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

    // S52.4: `fr` is seeded at 1, not 0. Both conditions have `fr` in
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
            // limitation S53.6 already records for a pressure-jump tile, and
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
            // S54.1: one more transported scalar on the SAME conservative
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
    let mut turb = KEpsilon::new(
        gpu,
        hm,
        &mesh,
        KEpsilonCoeffs::default(),
        tctrl_turb,
        ofgpu::io::case::WallFunctionCoeffs::default(),
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

    // ---- the outer loop ----------------------------------------------------
    for it in 0..lc.run.iterations {
        // S54.4: the buoyancy field. With humidity ON it is the virtual
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

        // SPEC-LIT S52.8: the cost of a fan curve is the cuFFT direct Poisson
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
                println!(
                    "  the cuFFT direct Poisson backend is NOT available on this \
                     system, and PBiCGStab is used instead. SPEC-LIT S52.8: a fan \
                     curve makes a patch face neither uniformly Dirichlet nor \
                     uniformly Neumann, and S53.2's jump makes the face coefficient \
                     non-constant. That is the biggest cost of these two features and \
                     it is printed, not hidden; S52.9 names the Woodbury correction \
                     that would put the direct path back and says it is not built."
                );
            }
            if !probe.symmetric {
                println!(
                    "  WARNING: the pressure matrix is NOT symmetric. SPEC-LIT S52.2 \
                     and S53.2 both say it should be - a fan is a symmetric rank-1 \
                     downdate and a jump divides upper and lower by the same number."
                );
            }
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
    let mut net = 0.0 as Scalar;
    let mut scale = 0.0 as Scalar;
    for (n, q) in &s.patch_flow {
        println!("  {n:<16} {q:12.5}");
        net += q;
        scale = scale.max(q.abs());
    }
    println!(
        "  {:<16} {net:12.5}   <- must be zero; {:.2e} of the largest opening",
        "NET",
        f64::from(if scale > 0.0 { net.abs() / scale } else { 0.0 })
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
            let d = (q - qp).abs() / qp.abs().max(1e-30);
            println!(
                "    the patch itself carried {qp:.5} m^3/s at the last corrector; the \
                 {:.2} % gap is the operating point's own outer residual (S52.6)",
                100.0 * d
            );
        }
    }
    println!("\n{}", r.pue.describe());

    if let Some(c) = &s.jump_caveat {
        println!("\n--- S53.6 ---\n  {c}");
    }
    if let Some((cells, worst)) = s.supersaturation {
        if cells > 0 {
            println!(
                "\n--- S54.5 ---\n  {cells} cell(s) are supersaturated, worst excess \
                 {worst:.6} kg/kg. Y_v is REPORTED and not clipped: field-level \
                 condensation is a different model and is refused by name."
            );
        } else {
            println!("\nno cell is supersaturated.");
        }
    }
    if let Some(c) = &s.molar_caveat {
        println!("\n--- S54.4 ---\n  {c}");
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
