// meteor-cfd - Copyright (c) 2026 주식회사 메테오시뮬레이션 (Meteo Simulation Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

//! `ofgpu-vof` - two-phase flow by volume of fluid, and the dam break.
//!
//! ```text
//! ofgpu-vof <caseDir> [options]
//!
//!   -endTime T        physical end time in seconds (default: controlDict)
//!   -deltaT dt        time step (default: controlDict deltaT)
//!   -maxCo C          adapt the step to hold the material Courant number at
//!                     C; 0 (the default) keeps deltaT fixed
//!   -maxDeltaT dt     ceiling on the adapted step
//!   -writeInterval W  write every W seconds of physical time
//!   -noWrite          do not write fields
//!   -surge            report the surge-front position against time in the
//!                     dimensionless variables of Martin & Moyce (1952)
//!   -a A              the column width those variables are scaled by; the
//!                     default is read from the initial alpha field
//!   -permissive       SPEC-LIT 13.4's escape hatch
//! ```
//!
//! Written from:
//!   ofgpu `SPEC-LIT.md` §20 (the method), §22 (the dam break benchmark),
//!     §13.4 (what happens to a setting this solver does not have)
//!   C. W. Hirt, B. D. Nichols, *J. Comput. Phys.* 39 (1981) 201
//!   S. T. Zalesak, *J. Comput. Phys.* 31 (1979) 335
//!   J. U. Brackbill, D. B. Kothe, C. Zemach, *J. Comput. Phys.* 100 (1992) 335
//!   J. C. Martin, W. J. Moyce, *Phil. Trans. R. Soc. A* 244 (1952) 312 - the
//!     collapsing-column experiment this driver's `-surge` report is written to
//!     be compared against
//!   R. I. Issa, *J. Comput. Phys.* 62 (1986) 40 - PISO
//! No GPL-licensed source was consulted.
//!
//! # What `-surge` reports, and what it does not
//!
//! Martin & Moyce released a column of liquid of width `a` and height `n^2 a`
//! and photographed the position `z` of the surge front against time. The
//! natural dimensionless groups of that problem are
//!
//! ```text
//! Z = z/a          T = t sqrt(g/a)
//! ```
//!
//! and `-surge` prints exactly those, together with the raw `t` and `z` in
//! seconds and metres so a reader can rescale into whichever convention the
//! copy of the paper in front of them uses. **The paper's tabulated data is
//! not reproduced in this source**, because nobody here has the paper to
//! transcribe it from, and a table of numbers attributed to a 1952 experiment
//! and actually remembered is worse than no table at all.
//!
//! What the code *does* assert about the surge front is in
//! `ofgpu::vof`'s tests and in `ofgpu-validate`: the phase volume is conserved,
//! `alpha` stays in `[0, 1]`, and the front never outruns `2 sqrt(g h0)`, the
//! characteristic speed limit of the frictionless shallow-water dam break
//! (Ritter 1892) - which is an analytic bound and not somebody's output.
//!
//! # The front, as measured here
//!
//! The furthest `x` at which the depth-integrated liquid fraction along the
//! bottom row of cells still exceeds half a cell. Reading it off the bottom row
//! rather than from the whole field is what makes it the SURGE front - the
//! tongue running along the floor - rather than the outline of the collapsing
//! bulk.

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use ofgpu::field_setup::{
    harvest_scalar_field, harvest_vector_field, setup_scalar_field, setup_vector_field,
};
use ofgpu::io::case::{find_start_time, format_time_name};
use ofgpu::io::dict::FoamDict;
use ofgpu::io::fields::{read_scalar_field, read_vector_field};
use ofgpu::io::polymesh::{build_host_mesh, read_poly_mesh};
use ofgpu::restart::{self, RestartData};
use ofgpu::vof::{phase_names, Vof, VofControls, VofProperties};
use ofgpu::{Error, Gpu, GpuMesh, HostMesh, Result, Scalar, Vec3};

#[path = "common/mod.rs"]
mod common;

use common::{
    build_writers, device_banner, find_restart_field, from_restart_scalars, from_restart_vectors,
    g, mean, next_arg, parse_output_formats, restart_scalar, restart_shell, restart_surface,
    restart_vector, sci, OutputFormat,
};

// ==========================================================================
//  Command line
// ==========================================================================

struct Options {
    case_dir: PathBuf,
    end_time: f64,
    delta_t: f64,
    max_co: f64,
    max_delta_t: f64,
    write_interval: f64,
    do_write: bool,
    surge: bool,
    /// Non-positive means "work it out from the initial condition".
    column_width: f64,
    /// `-output foam|vtu|nvdb|vdb|usda`, comma list.
    output: Vec<OutputFormat>,
    /// `-restartWrite N` - write a `.mcr` checkpoint every N STEPS.
    restart_write: Option<u64>,
    /// `-restartFrom FILE` - load state from a `.mcr` checkpoint instead of
    /// the case's own start-time directory, and skip straight past the
    /// zero-velocity flux initialisation that directory would otherwise get.
    restart_from: Option<PathBuf>,
    /// `-reportEvery N` - print the per-step residual line every N steps
    /// (always also on step 1 and the last step).
    report_every: u64,
}

fn usage() {
    eprintln!(
        "usage: ofgpu-vof <caseDir> [-endTime T] [-deltaT dt] [-maxCo C] \
         [-maxDeltaT dt]\n       \
         [-writeInterval W] [-noWrite] [-surge] [-a A] [-output LIST]\n       \
         [-restartWrite N] [-restartFrom FILE] [-reportEvery N]\n\
         {}",
        ofgpu::io::contract::PERMISSIVE_USAGE
    );
}

/// A physical time or length from the command line.
///
/// Deliberately strict: `-deltaT 0.001` read by an integer parser is zero, and
/// a zero time step does not fail - it makes `1/dt` infinite and fills every
/// matrix with NaN several hundred launches away from the mistake.
fn parse_number(flag: &str, s: &str) -> Result<f64> {
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
        return Err(Error::Config("no case directory given".to_string()));
    }

    let mut o = Options {
        case_dir: PathBuf::from(&args[1]),
        end_time: -1.0,
        delta_t: -1.0,
        max_co: 0.0,
        max_delta_t: -1.0,
        write_interval: -1.0,
        do_write: true,
        surge: false,
        column_width: -1.0,
        output: vec![OutputFormat::Foam],
        restart_write: None,
        restart_from: None,
        report_every: 20,
    };

    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "-endTime" => o.end_time = parse_number("-endTime", &next_arg(args, &mut i)?)?,
            "-deltaT" => o.delta_t = parse_number("-deltaT", &next_arg(args, &mut i)?)?,
            "-maxCo" => o.max_co = parse_number("-maxCo", &next_arg(args, &mut i)?)?,
            "-maxDeltaT" => {
                o.max_delta_t = parse_number("-maxDeltaT", &next_arg(args, &mut i)?)?
            }
            "-writeInterval" => {
                o.write_interval = parse_number("-writeInterval", &next_arg(args, &mut i)?)?
            }
            "-noWrite" => o.do_write = false,
            "-surge" => o.surge = true,
            "-a" => o.column_width = parse_number("-a", &next_arg(args, &mut i)?)?,
            "-output" => o.output = parse_output_formats(&next_arg(args, &mut i)?)?,
            "-restartWrite" => {
                let n = parse_number("-restartWrite", &next_arg(args, &mut i)?)?;
                if n <= 0.0 {
                    return Err(Error::Config("-restartWrite needs a positive step count".into()));
                }
                o.restart_write = Some(n as u64);
            }
            "-restartFrom" => o.restart_from = Some(PathBuf::from(next_arg(args, &mut i)?)),
            "-reportEvery" => {
                let n = parse_number("-reportEvery", &next_arg(args, &mut i)?)?;
                if n <= 0.0 {
                    return Err(Error::Config("-reportEvery needs a positive step count".into()));
                }
                o.report_every = n as u64;
            }
            // Set before any dictionary is opened, so every rejection this run
            // makes sees the same policy.
            "-permissive" => ofgpu::io::contract::set_permissive(true),
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
//  The surge front
// ==========================================================================

/// The bottom row of cells, in ascending `x`, and the column width the initial
/// condition implies.
///
/// "Bottom" is the row whose centres have the smallest coordinate along
/// gravity. Found from the geometry rather than assumed, so a case whose
/// gravity is along `-z` reports the same thing as one along `-y`.
struct FloorRow {
    /// Cell indices, ascending in the horizontal coordinate.
    cells: Vec<usize>,
    /// The horizontal coordinate of each, in the same order.
    x: Vec<Scalar>,
    /// Which component of a position is horizontal, and which vertical.
    horizontal: usize,
    /// Cell size along `horizontal`, for the half-cell threshold.
    dx: Scalar,
}

impl FloorRow {
    fn find(m: &HostMesh, gravity: Vec3) -> Result<Self> {
        // The vertical is the component gravity is largest in; the horizontal
        // is the largest of the two that are left, measured by how far the
        // mesh extends in each - which on a 2-D case excludes the direction
        // the `empty` patches suppress.
        let gcmpt = |i: usize| gravity.component(i).abs();
        let vertical = (0..3).max_by(|a, b| gcmpt(*a).total_cmp(&gcmpt(*b))).unwrap_or(2);

        let extent = |i: usize| -> Scalar {
            let mut lo = Scalar::MAX;
            let mut hi = Scalar::MIN;
            for c in 0..m.n_cells {
                let v = m.c[c].component(i);
                lo = lo.min(v);
                hi = hi.max(v);
            }
            hi - lo
        };

        let horizontal = (0..3)
            .filter(|i| *i != vertical)
            .max_by(|a, b| extent(*a).total_cmp(&extent(*b)))
            .ok_or_else(|| Error::Config("the mesh has no cells".to_string()))?;

        let mut zmin = Scalar::MAX;
        for c in 0..m.n_cells {
            zmin = zmin.min(m.c[c].component(vertical));
        }

        // Half the layer spacing, so exactly one layer of cells qualifies.
        //
        // The spacing is the smallest difference from the bottom that is
        // BIGGER THAN THE NOISE, and the noise is real: a cell centre comes
        // out of a pyramid decomposition over the cell's faces (SPEC-LIT
        // §2.2), so two cells nominally on the same level differ in the last
        // few bits. Taking the smallest positive difference instead - which is
        // what this did first - picks up one of those bit-level differences,
        // and the "bottom row" comes out as the two cells that happened to
        // round the same way.
        let extent_v = extent(vertical).max(Scalar::MIN_POSITIVE);
        let noise = 1e-6 * extent_v;

        let mut spacing = Scalar::MAX;
        for c in 0..m.n_cells {
            let d = m.c[c].component(vertical) - zmin;
            if d > noise {
                spacing = spacing.min(d);
            }
        }
        let tol = if spacing < Scalar::MAX { 0.5 * spacing } else { noise };

        let mut rows: Vec<(Scalar, usize)> = (0..m.n_cells)
            .filter(|c| m.c[*c].component(vertical) - zmin < tol)
            .map(|c| (m.c[c].component(horizontal), c))
            .collect();
        rows.sort_by(|a, b| a.0.total_cmp(&b.0));

        if rows.len() < 2 {
            return Err(Error::Config(format!(
                "the surge report needs a bottom row of at least two cells                  and found {}",
                rows.len()
            )));
        }

        // The mean spacing, not the first gap: on a graded mesh the first gap
        // is the smallest cell and would understate the half-cell the front
        // interpolation is offset by.
        let dx = (rows[rows.len() - 1].0 - rows[0].0) / (rows.len() - 1) as Scalar;

        Ok(Self {
            cells: rows.iter().map(|r| r.1).collect(),
            x: rows.iter().map(|r| r.0).collect(),
            horizontal,
            dx,
        })
    }

    /// The furthest `x` along the floor at which `alpha` is still above a half.
    ///
    /// Linearly interpolated between the last cell above and the first below,
    /// so the answer moves smoothly rather than in cell-sized jumps.
    fn front(&self, alpha: &[Scalar]) -> Scalar {
        let mut last = None;
        for (i, c) in self.cells.iter().enumerate() {
            if alpha[*c] >= 0.5 {
                last = Some(i);
            }
        }
        let Some(i) = last else { return self.x[0] };
        if i + 1 >= self.cells.len() {
            return self.x[i];
        }

        let a0 = alpha[self.cells[i]];
        let a1 = alpha[self.cells[i + 1]];
        let f = if (a0 - a1).abs() > 1e-30 {
            ((a0 - 0.5) / (a0 - a1)).clamp(0.0, 1.0)
        } else {
            0.0
        };
        self.x[i] + f * (self.x[i + 1] - self.x[i])
    }

    /// The column width the initial condition implies: the front at `t = 0`,
    /// measured from the wall the column stands against.
    fn initial_width(&self, alpha: &[Scalar]) -> Scalar {
        // The wall is half a cell before the first cell centre.
        self.front(alpha) - (self.x[0] - 0.5 * self.dx)
    }
}

// ==========================================================================
//  Driving
// ==========================================================================

fn run(o: &Options) -> Result<()> {
    let case = o.case_dir.as_path();
    let gpu = Gpu::new(0)?;
    println!("{}", device_banner(&gpu, "vof")?);

    // ---- mesh -------------------------------------------------------------
    let raw = read_poly_mesh(case)?;
    let hm = build_host_mesh(&raw)?;
    hm.print_report();
    let m = GpuMesh::upload(&gpu, &hm)?;
    let mesh_hash = restart::mesh_hash(&hm);

    // ---- restart, if asked for ---------------------------------------------
    //
    // Read (and hash-check) up front, before anything else touches the case,
    // so a bad `-restartFrom` fails before a single kernel launches rather
    // than after the mesh and case dictionaries were already read for
    // nothing.
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

    // ---- the case ---------------------------------------------------------
    let names = phase_names(case)?;
    let props = VofProperties::from_case(case)?;
    let mut ctrl = VofControls::from_case(case)?;

    let control_dict = case.join("system").join("controlDict");
    let cd = if control_dict.exists() {
        FoamDict::read(&control_dict)?
    } else {
        FoamDict::default()
    };

    if o.delta_t > 0.0 {
        ctrl.delta_t = o.delta_t as Scalar;
    }
    let end_time = if o.end_time > 0.0 {
        o.end_time as Scalar
    } else {
        cd.scalar("endTime", 1.0)
    };
    let write_interval = if o.write_interval > 0.0 {
        o.write_interval as Scalar
    } else {
        cd.scalar("writeInterval", end_time)
    };
    // SPEC-LIT §13.4 and §20.2. `maxDeltaT`/`maxCo`/`adjustTimeStep` come off
    // `VofControls` rather than being re-read from `controlDict` here, so
    // there is ONE reader of the case's adaptive-step settings and not two -
    // the flags still win where they are given, and the case's own numbers
    // are what a run with neither gets. `-maxCo` used to be the ONLY way to
    // reach the adaptive branch: `controlDict/maxCo` and `adjustTimeStep`
    // were read by nothing, so a case asking for an adaptive step ran fixed.
    let max_delta_t = if o.max_delta_t > 0.0 {
        o.max_delta_t as Scalar
    } else {
        ctrl.max_delta_t
    };
    // `-maxCo` wins; otherwise the case's own `maxCo`, and only when it also
    // wrote `adjustTimeStep yes` - writing a Courant ceiling and switching
    // adaptation off is a case saying "not now", and honouring the number
    // anyway would be this driver overruling it.
    let max_co = if o.max_co > 0.0 {
        o.max_co as Scalar
    } else if ctrl.adjust_time_step {
        ctrl.max_co
    } else {
        0.0
    };

    println!("\n{}", props.describe(&names));
    println!("Atwood number {}", g(f64::from(props.atwood())));
    println!(
        "PISO: {} corrector(s), {} non-orthogonal; momentum predictor {}",
        ctrl.n_correctors,
        ctrl.n_non_orth_correctors,
        if ctrl.momentum_predictor { "on" } else { "off" }
    );
    println!(
        "alpha: maxAlphaCo {}, up to {} sub-cycles, {} Zalesak limiter \
         iteration(s), cAlpha {}",
        g(f64::from(ctrl.max_alpha_co)),
        ctrl.max_sub_cycles,
        ctrl.n_limiter_iters,
        g(f64::from(props.c_alpha))
    );
    // SPEC-LIT §13.4.2: the numerics this run will actually use, per entry
    // and by that entry's own key. Everything but the first line here was
    // read by nobody before this sweep - see `VofControls::from_case`.
    println!(
        "div(rhoPhi,U) {} (unbounded - SPEC-LIT §20.3) | ddt Euler | grad(U) {} | \
         grad(p_rgh) {} | grad({alpha_key}) {} | laplacian snGrad {} | relax U {}",
        ctrl.div_scheme.describe(),
        ctrl.grad_u.describe(),
        ctrl.grad_p.describe(),
        ctrl.grad_alpha.describe(),
        ctrl.sn_grad.describe(),
        g(f64::from(ctrl.u_relax)),
        alpha_key = format!("alpha.{}", names[0]),
    );
    println!(
        "time step: deltaT {}{} | endTime {} | writeInterval {}",
        g(f64::from(ctrl.delta_t)),
        if max_co > 0.0 {
            format!(
                ", adaptive to maxCo {} (ceiling maxDeltaT {})",
                g(f64::from(max_co)),
                g(f64::from(max_delta_t))
            )
        } else {
            ", fixed".to_string()
        },
        g(f64::from(end_time)),
        g(f64::from(write_interval))
    );

    // ---- fields -----------------------------------------------------------
    let start = find_start_time(case)?;
    let dir = case.join(&start);
    let alpha_name = format!("alpha.{}", names[0]);

    let alpha_path = dir.join(&alpha_name);
    if !alpha_path.exists() {
        return Err(Error::Config(format!(
            "{}: no phase fraction field. This solver expects \"{alpha_name}\", \
             named for the first entry of `phases` in \
             constant/transportProperties.",
            alpha_path.display()
        )));
    }

    let mut raw_alpha = read_scalar_field(&alpha_path, hm.n_cells)?;
    let mut raw_u = read_vector_field(&dir.join("U"), hm.n_cells)?;
    let mut raw_p = read_scalar_field(&dir.join("p_rgh"), hm.n_cells)?;

    let mut vof = Vof::new(&gpu, &hm, &m, props, ctrl)?;

    setup_scalar_field(&gpu, vof.alpha_mut(), &raw_alpha, &hm)?;
    setup_vector_field(&gpu, vof.u_mut(), &raw_u, &hm)?;
    setup_scalar_field(&gpu, vof.p_rgh_mut(), &raw_p, &hm)?;

    // SPEC-LIT S39: the contact angle, out of the SAME alpha field entries
    // `setup_scalar_field` just read. A case naming neither
    // `constantAlphaContactAngle` nor `dynamicAlphaContactAngle` on any patch
    // leaves the model off, and `vofFaceUnitNormalBoundary` then writes the
    // literal zero it always wrote - which is what makes every VOF result
    // recorded before S39 bit-identical.
    let (ca_faces, ca_banner) =
        ofgpu::contact_angle::ContactAngleFaces::from_alpha_field(&raw_alpha, &hm)?;
    vof.set_contact_angle(&gpu, &ca_faces)?;
    if vof.n_contact_angle_faces() > 0 {
        println!(
            "contact angle (SPEC-LIT 39): {} faces | {}",
            vof.n_contact_angle_faces(),
            ca_banner.join(" | ")
        );
    }

    if let Some(rd) = &restart_data {
        // Overwrite only the INTERNAL cell values with the restart's numbers.
        // The boundary condition TYPES just read from `dir` above
        // (`fixedValue`, `zeroGradient`, ...) are case parameters that do not
        // change during a run - only the field itself does - and
        // `vof.initialise` below re-derives every boundary cell from the
        // restored internal ones regardless.
        let a = find_restart_field(rd, &alpha_name)?;
        gpu.write(&mut vof.alpha_mut().f, &from_restart_scalars(&a.internal))?;
        let u = find_restart_field(rd, "U")?;
        gpu.write(&mut vof.u_mut().f, &from_restart_vectors(&u.internal))?;
        let p = find_restart_field(rd, "p_rgh")?;
        gpu.write(&mut vof.p_rgh_mut().f, &from_restart_scalars(&p.internal))?;
    }

    if let Some(rd) = &restart_data {
        // The conservative flux this restart was written with - SPEC-LIT
        // 5.1. `initialise_flux_from_velocity` would throw it away for
        // `interpolate(U) & Sf`, exactly the non-conservative fallback a
        // restart exists to skip.
        let phi = find_restart_field(rd, "phi")?;
        gpu.write(&mut vof.phi_mut().f, &from_restart_scalars(&phi.internal))?;
        gpu.write(&mut vof.phi_mut().bf, &from_restart_scalars(&phi.boundary))?;
        println!("restart: phi loaded from the checkpoint - not re-derived from U");
    } else {
        // A flux to start from. The first pressure correction makes it
        // conservative; this only decides how much work that correction has
        // to do, and starting from a velocity of zero it is exactly zero
        // anyway.
        vof.initialise_flux_from_velocity(&gpu)?;
    }
    vof.initialise(&gpu)?;

    if let Some(rd) = &restart_data {
        // `initialise` re-derives every boundary cell generically
        // (`correct_boundary_conditions[_vector]`), which assumes a COLD
        // start: it does not first run `update_inlet_outlet[_vector]`, so on
        // a patch whose flow direction has flipped since the case's own `0/`
        // files were written, it evaluates the WRONG branch of an
        // `inletOutlet`/`pressureInletOutletVelocity` condition. The
        // checkpoint's own boundary values do not have that problem - they
        // are exactly what the continuous run had - so they overwrite
        // whatever `initialise` just computed.
        let a = find_restart_field(rd, &alpha_name)?;
        gpu.write(&mut vof.alpha_mut().bf, &from_restart_scalars(&a.boundary))?;
        let u = find_restart_field(rd, "U")?;
        gpu.write(&mut vof.u_mut().bf, &from_restart_vectors(&u.boundary))?;
        let p = find_restart_field(rd, "p_rgh")?;
        gpu.write(&mut vof.p_rgh_mut().bf, &from_restart_scalars(&p.boundary))?;

        // `initialise` also reseeds `alpha_phi`/`rho_phi` with the cold-start
        // upwind approximation (`seed_alpha_flux`) and rebuilds `rho0` from
        // the just-restored `alpha` (`rho0 == rho`, i.e. a zero density ddt
        // for the next step) - both wrong for a resumed run. Put back the
        // exact values the checkpoint carries, same reasoning as the
        // boundary restore above - see `Vof::alpha_phi_mut`.
        let aphi = find_restart_field(rd, "alphaPhi")?;
        gpu.write(&mut vof.alpha_phi_mut().f, &from_restart_scalars(&aphi.internal))?;
        gpu.write(&mut vof.alpha_phi_mut().bf, &from_restart_scalars(&aphi.boundary))?;
        let rphi = find_restart_field(rd, "rhoPhi")?;
        gpu.write(&mut vof.rho_phi_mut().f, &from_restart_scalars(&rphi.internal))?;
        gpu.write(&mut vof.rho_phi_mut().bf, &from_restart_scalars(&rphi.boundary))?;
        let rho0 = find_restart_field(rd, "rho0")?;
        gpu.write(vof.rho_old_mut(), &from_restart_scalars(&rho0.internal))?;
    }

    println!(
        "interface normal eps = {} (SPEC-LIT 20.1, stated as the section asks)",
        sci(f64::from(vof.interface_eps()), 3)
    );
    println!(
        "p_rgh is {}",
        if vof.pressure_is_pinned() {
            "pinned: no fixedValue anywhere, so its level is set by the solver"
        } else {
            "set by a boundary Dirichlet"
        }
    );

    // ---- the surge report -------------------------------------------------
    let floor = if o.surge {
        Some(FloorRow::find(&hm, props.g)?)
    } else {
        None
    };
    let mut column_width = o.column_width as Scalar;
    if let Some(f) = &floor {
        let a = gpu.download(&vof.alpha().f)?;
        if !(column_width > 0.0) {
            column_width = f.initial_width(&a);
        }
        let axis = ["x", "y", "z"][f.horizontal];
        println!(
            "surge front measured along {axis} on the bottom row of {} cells; \
             column width a = {} m",
            f.cells.len(),
            g(f64::from(column_width))
        );
        println!("\n{:>10} {:>10} {:>12} {:>10} {:>10}", "t", "T", "z", "Z", "dZ/dT");
    }

    let v0 = vof.phase_volume(&gpu)?;
    let gmag = props.g.mag();

    // ---- the time loop ----------------------------------------------------
    let mut t = restart_data.as_ref().map_or(0.0, |d| d.time as Scalar);
    let mut dt = ctrl.delta_t;
    let mut next_write = write_interval;
    let mut step = 0u64;
    let mut z_prev = 0.0 as Scalar;
    let mut t_prev = 0.0 as Scalar;

    let clock = Instant::now();

    while t < end_time - 1e-12 * end_time.max(1.0) {
        // Do not step past the end.
        if t + dt > end_time {
            dt = end_time - t;
        }

        let perf = vof.step(&gpu, dt)?;
        t += dt;
        step += 1;

        if let Some(f) = &floor {
            let a = gpu.download(&vof.alpha().f)?;
            let z = f.front(&a) - (f.x[0] - 0.5 * f.dx);
            let tt = t * (gmag / column_width).sqrt();
            let zz = z / column_width;
            let rate = if t > t_prev {
                (zz - z_prev) / ((t - t_prev) * (gmag / column_width).sqrt())
            } else {
                0.0
            };
            println!(
                "{:>10} {:>10} {:>12} {:>10} {:>10}",
                g(f64::from(t)),
                g(f64::from(tt)),
                g(f64::from(z)),
                g(f64::from(zz)),
                g(f64::from(rate))
            );
            z_prev = zz;
            t_prev = t;
        } else if step == 1 || step % o.report_every == 0 || t >= end_time {
            let (lo, hi) = vof.alpha_bounds(&gpu)?;
            println!(
                "step {step:>5}  t = {:>10}  dt {:>10}  alphaCo {:>8}  x{} sub  p_rgh {} -> {} in {} \
                 iters  continuity {}  alpha [{}, {}]",
                g(f64::from(t)),
                g(f64::from(dt)),
                g(f64::from(perf.alpha_courant)),
                perf.n_sub_cycles,
                sci(f64::from(perf.p_rgh.initial_residual), 16),
                sci(f64::from(perf.p_rgh.final_residual), 2),
                perf.p_rgh.n_iterations,
                sci(f64::from(perf.continuity_error), 2),
                sci(f64::from(lo), 2),
                g(f64::from(hi))
            );
        }

        // Adaptive stepping, if asked for. The Courant number the alpha
        // equation reports is the material one, so it is the right thing to
        // hold: the momentum equation is implicit and does not need it, but a
        // VOF interface moving more than a cell a step is a VOF interface
        // whose sub-cycling is doing all the work.
        if max_co > 0.0 && perf.alpha_courant > 0.0 {
            let want = dt * max_co / perf.alpha_courant;
            // Never more than a 20 % rise in one step: an adaptive step that
            // doubles lands on a state the previous one has not settled into.
            dt = want.min(1.2 * dt).min(max_delta_t);
        }

        if o.do_write && t + 1e-12 >= next_write {
            write_time(
                &gpu, &vof, &hm, case, t, &o.output, &alpha_name, &mut raw_alpha, &mut raw_u,
                &mut raw_p,
            )?;
            next_write += write_interval;
        }

        if let Some(interval) = o.restart_write {
            if step % interval == 0 {
                write_restart_checkpoint(&gpu, &vof, &hm, case, mesh_hash, t, &alpha_name)?;
            }
        }

    }

    let elapsed = clock.elapsed().as_secs_f64();
    let v1 = vof.phase_volume(&gpu)?;
    let (lo, hi) = vof.alpha_bounds(&gpu)?;

    println!(
        "\n{step} steps to t = {} in {:.2} s ({:.1} steps/s)",
        g(f64::from(t)),
        elapsed,
        step as f64 / elapsed.max(1e-12)
    );
    println!(
        "phase volume {} -> {} (relative change {})",
        sci(f64::from(v0), 6),
        sci(f64::from(v1), 6),
        sci(f64::from((v1 - v0) / v0.max(Scalar::MIN_POSITIVE)), 2)
    );
    println!("alpha in [{}, {}]", sci(f64::from(lo), 3), g(f64::from(hi)));

    if o.do_write {
        write_time(
            &gpu, &vof, &hm, case, t, &o.output, &alpha_name, &mut raw_alpha, &mut raw_u,
            &mut raw_p,
        )?;
    }

    Ok(())
}

/// Everything one restart interval needs: `alpha`, `U`, `p_rgh` and `phi`,
/// Everything one restart interval needs: `alpha`, `U`, `p_rgh` and `phi`,
/// including their boundary face values (`phi`'s in particular, since a
/// `SurfaceScalar` restart field's boundary array is `phi.bf` itself, not a
/// per-patch `PatchFieldSpec`). See `common::restart_scalar` and friends.
fn write_restart_checkpoint(
    gpu: &Gpu,
    vof: &Vof<'_>,
    hm: &HostMesh,
    case: &Path,
    mesh_hash: u64,
    t: Scalar,
    alpha_name: &str,
) -> Result<()> {
    let alpha_i = gpu.download(&vof.alpha().f)?;
    let alpha_b = gpu.download(&vof.alpha().bf)?;
    let u_i = gpu.download(&vof.u().f)?;
    let u_b = gpu.download(&vof.u().bf)?;
    let p_i = gpu.download(&vof.p_rgh().f)?;
    let p_b = gpu.download(&vof.p_rgh().bf)?;
    let phi_i = gpu.download(&vof.phi().f)?;
    let phi_b = gpu.download(&vof.phi().bf)?;
    // The Zalesak-limited advective fluxes `solve_alpha` last produced, and
    // the density level the momentum ddt is differenced against - not part
    // of the case's own fields, but load-bearing state a restart loses
    // silently otherwise. See `Vof::alpha_phi_mut`'s doc.
    let aphi_i = gpu.download(&vof.alpha_phi().f)?;
    let aphi_b = gpu.download(&vof.alpha_phi().bf)?;
    let rphi_i = gpu.download(&vof.rho_phi().f)?;
    let rphi_b = gpu.download(&vof.rho_phi().bf)?;
    let rho0 = gpu.download(vof.rho_old())?;

    let p0 = mean(&p_i);
    let mut data = restart_shell(mesh_hash, t, p0, hm);
    data.fields.push(restart_scalar(alpha_name, &alpha_i, &alpha_b));
    data.fields.push(restart_vector("U", &u_i, &u_b));
    data.fields.push(restart_scalar("p_rgh", &p_i, &p_b));
    data.fields.push(restart_surface("phi", &phi_i, &phi_b));
    data.fields.push(restart_surface("alphaPhi", &aphi_i, &aphi_b));
    data.fields.push(restart_surface("rhoPhi", &rphi_i, &rphi_b));
    // `rho0` has no natural boundary array (it is a plain cell buffer, not a
    // `GpuScalarField`) - padded with zeros to satisfy the `CellScalar`
    // layout; never read back on the boundary side.
    data.fields.push(restart_scalar("rho0", &rho0, &vec![0.0 as Scalar; hm.n_boundary_faces]));

    let path = case.join("restart.mcr");
    restart::write_restart(&path, &data)?;
    println!(
        "  restart checkpoint written to {} (t = {}, p0 = {})",
        path.display(),
        g(f64::from(t)),
        sci(f64::from(p0), 3)
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn write_time(
    gpu: &Gpu,
    vof: &Vof<'_>,
    hm: &HostMesh,
    case: &Path,
    t: Scalar,
    output: &[OutputFormat],
    alpha_name: &str,
    raw_alpha: &mut ofgpu::io::fields::RawScalarField,
    raw_u: &mut ofgpu::io::fields::RawVectorField,
    raw_p: &mut ofgpu::io::fields::RawScalarField,
) -> Result<()> {
    let name = format_time_name(t);

    harvest_scalar_field(gpu, raw_alpha, vof.alpha(), hm)?;
    harvest_vector_field(gpu, raw_u, vof.u(), hm)?;
    harvest_scalar_field(gpu, raw_p, vof.p_rgh(), hm)?;

    // One seam call per requested format, replacing what used to be three
    // scattered `fields::write_*` sites - see `ofgpu::io::writer`.
    let foam_fields = [
        ofgpu::io::FoamField::scalar(alpha_name, raw_alpha),
        ofgpu::io::FoamField::vector("U", raw_u),
        ofgpu::io::FoamField::scalar("p_rgh", raw_p),
    ];
    let vis_fields = [
        ofgpu::io::OutputField::scalar(alpha_name, &raw_alpha.internal),
        ofgpu::io::OutputField::vector("U", &raw_u.internal),
        ofgpu::io::OutputField::scalar("p_rgh", &raw_p.internal),
    ];
    let cart = ofgpu::pressure::cartesian::detect(hm)
        .ok()
        .map(|c| ofgpu::io::cartesian_info(hm, &c));
    let ctx = ofgpu::io::WriteCtx {
        time: t,
        step: 0,
        name: &name,
        mesh: hm,
        cart: cart.as_ref(),
        fields: &vis_fields,
        foam: &foam_fields,
    };
    let mut writers = build_writers(case, "vof", output, ofgpu::io::nvdb::Precision::F32)?;
    for w in &mut writers {
        w.write_step(&ctx)?;
    }

    println!("  written to {}", case.join(&name).display());
    Ok(())
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();

    let o = match parse(&args) {
        Ok(o) => o,
        Err(e) => {
            eprintln!("ofgpu-vof: {e}");
            return ExitCode::from(2);
        }
    };

    match run(&o) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("\nofgpu-vof: {e}");
            ExitCode::from(1)
        }
    }
}

// ==========================================================================
//  Tests - SPEC-LIT 13.4.1's standing requirement
// ==========================================================================
//
// Named `vof_tests` rather than `tests` because `common/mod.rs` is included
// by `#[path]` and already contributes a `tests` module to this crate.
//
// Before this sweep `VofControls::from_case` read nine entries and this
// driver read `controlDict` for three more; everything else a two-phase case
// can write - `gradSchemes` in its entirety, `laplacianSchemes`, the
// `bounded` prefix on the momentum convection, `ddtSchemes`,
// `nOuterCorrectors`, `relaxationFactors/fields/p_rgh`, `residualControl`,
// `adjustTimeStep`/`maxCo`, and every entry a case wrote in a `SIMPLE` or
// `PISO` dictionary rather than a `PIMPLE` one - parsed perfectly and
// reached nothing.

#[cfg(test)]
mod vof_tests {
    use super::*;
    use common::knobs::{apply, assert_none_inert, scratch_dir, written_time_dirs, Knob, NO_PRE};
    use ofgpu::blockgen::{write_case, CaseKind};

    fn argv(v: &[&str]) -> Vec<String> {
        std::iter::once("ofgpu-vof".to_string())
            .chain(v.iter().map(|s| (*s).to_string()))
            .collect()
    }

    fn dam_break(tag: &str) -> PathBuf {
        let dir = scratch_dir(tag);
        let case = dir.join("case");
        write_case(&case, CaseKind::DamBreak, 20, 30, 1).expect("generate the damBreak case");
        case
    }

    /// Build a fresh case, apply `k` if `side` is set, run `ofgpu-vof`'s own
    /// `parse` + `run`, and return every TIME DIRECTORY it wrote.
    ///
    /// Deliberately not `written_state(&case)`: the case root also holds the
    /// dictionaries the knob edits, so comparing it would compare the knob
    /// with itself.
    fn run_knob(k: &Knob, side: bool, tag: &str) -> Vec<(String, String)> {
        let case = dam_break(tag);
        apply(&case, k, side);

        let args = argv(&[
            case.to_string_lossy().as_ref(),
            "-endTime",
            "0.004",
            "-deltaT",
            "0.0005",
            "-reportEvery",
            "1000",
        ]);
        let o = parse(&args).expect("the knob command line must parse");
        run(&o).expect("the knob case must run");

        let out = written_time_dirs(&case);
        assert!(!out.is_empty(), "the run wrote nothing to compare");
        out
    }

    /// **The standing test SPEC-LIT 13.4.1 requires of every setting this
    /// driver claims to honour.**
    ///
    /// `gradSchemes/grad(p_rgh)`, `laplacianSchemes` and `snGradSchemes` are
    /// absent for 13.4.1's one admissible reason: §2.4's correction is
    /// `k = Sf - |Sf|^2/(Sf.d) d`, identically zero on the rectangular
    /// Cartesian box `blockgen` builds, so the entries that scale it cannot
    /// change a single bit here whatever the reader does. They are asserted
    /// on `VofControls` instead, in
    /// `the_pressure_gradient_and_the_laplacian_entry_reach_the_controls`.

    /// What the two sides of the §39.6 regression gate may be compared on.
    ///
    /// The knob edits `0/alpha.water`'s patch TYPE, and this driver writes
    /// the type back out with the solution, so the two runs' `alpha` files
    /// differ by the knob itself whatever the solver did - and by more than
    /// the token, because the writer emits a `value` list for a
    /// `constantAlphaContactAngle` patch and not for a `zeroGradient` one.
    /// Comparing them raw would compare the knob with itself, which is the
    /// failure `written_time_dirs` warns about one level up.
    ///
    /// So `alpha.water` is compared on its INTERNAL field alone, and every
    /// other written file - `U`, `p_rgh`, `phi` - in full, boundary values
    /// included. Those are what carry the physics a wrong contact angle would
    /// change, and none of them is touched by the knob.
    fn solution_only(files: Vec<(String, String)>) -> Vec<(String, String)> {
        files
            .into_iter()
            .map(|(name, text)| {
                if name.ends_with("alpha.water") {
                    let cut = text.find("boundaryField").unwrap_or(text.len());
                    (name, text[..cut].to_string())
                } else {
                    (name, text)
                }
            })
            .collect()
    }

    /// **SPEC-LIT §39.6's regression gate, end to end.**
    ///
    /// `theta = 90` is what the pre-§39 solver applied to every wall - it is
    /// what `n_hat . Sf = 0` MEANS. So a case that names
    /// `constantAlphaContactAngle; theta0 90;` on every wall must write
    /// exactly what a case that names `zeroGradient` writes, BIT FOR BIT.
    ///
    /// This is the test the `cos(pi/2)` trap breaks. `cos(pi/2)` in double
    /// precision is `6.123233995736766e-17`; `|Sf|` times that is far too
    /// small to see in a plot and far too large to be nothing, so a solver
    /// that wrote `|Sf| cos(theta)` unconditionally would move every recorded
    /// VOF measurement and no test would say why. Two guards stop it - the
    /// `enabled` flag in `vofFaceUnitNormalBoundary` and the exact-90 case in
    /// `contact_angle::cos_deg` - and this is what proves BOTH of them,
    /// through the driver, on a real dam break.
    #[test]
    fn a_ninety_degree_contact_angle_is_bit_identical_to_no_model() {
        if Gpu::new(0).is_err() {
            return;
        }
        let k = Knob {
            label: "0/alpha.water theta0 90",
            file: "0/alpha.water",
            from: "    leftWall\n    {\n        type            zeroGradient;\n    }\n    lowerWall\n    {\n        type            zeroGradient;\n    }\n    rightWall\n    {\n        type            zeroGradient;\n    }",
            to: "    leftWall\n    {\n        type            constantAlphaContactAngle;\n        theta0          90;\n    }\n    lowerWall\n    {\n        type            constantAlphaContactAngle;\n        theta0          90;\n    }\n    rightWall\n    {\n        type            constantAlphaContactAngle;\n        theta0          90;\n    }",
            pre: NO_PRE,
        };
        let none = solution_only(run_knob(&k, false, "ca90a"));
        let ninety = solution_only(run_knob(&k, true, "ca90b"));
        assert_eq!(
            none, ninety,
            "a contact angle of ninety degrees is the no-wall-adhesion angle the \
             solver applied before SPEC-LIT 39, so the two runs must be \
             bit-identical - cos(pi/2) is 6.1e-17 and not zero, and this is what \
             catches a kernel that forgot to special-case it (SPEC-LIT 39.6)"
        );
    }

    /// And the same pair with a real angle must NOT be bit-identical, or the
    /// test above would pass for a model that is simply never called.
    #[test]
    fn a_forty_five_degree_contact_angle_is_not_bit_identical_to_no_model() {
        if Gpu::new(0).is_err() {
            return;
        }
        let k = Knob {
            label: "0/alpha.water theta0 45",
            file: "0/alpha.water",
            from: "    leftWall\n    {\n        type            zeroGradient;\n    }\n    lowerWall\n    {\n        type            zeroGradient;\n    }\n    rightWall\n    {\n        type            zeroGradient;\n    }",
            to: "    leftWall\n    {\n        type            constantAlphaContactAngle;\n        theta0          45;\n    }\n    lowerWall\n    {\n        type            constantAlphaContactAngle;\n        theta0          45;\n    }\n    rightWall\n    {\n        type            constantAlphaContactAngle;\n        theta0          45;\n    }",
            pre: NO_PRE,
        };
        assert_ne!(
            solution_only(run_knob(&k, false, "ca45a")),
            solution_only(run_knob(&k, true, "ca45b")),
            "a 45-degree contact angle wrote the same fields as no model at all"
        );
    }

    #[test]
    fn every_wired_setting_changes_what_the_run_writes() {
        if Gpu::new(0).is_err() {
            return;
        }

        let cases: Vec<Knob> = vec![
            Knob {
                label: "divSchemes/div(rhoPhi,U)",
                file: "system/fvSchemes",
                from: "div(rhoPhi,U)    Gauss upwind;",
                to: "div(rhoPhi,U)    Gauss linear;",
                pre: NO_PRE,
            },
            // The interface normal reads `grad(alpha.water)` on every step,
            // unconditionally (§20.1/§20.4), so this knob turns ONE entry.
            Knob {
                label: "gradSchemes/grad(alpha.water)",
                file: "system/fvSchemes",
                from: "gradSchemes\n{\n    default         Gauss linear;\n}",
                to: "gradSchemes\n{\n    default         Gauss linear;\n    grad(alpha.water) cellLimited Gauss linear 1;\n}",
                pre: NO_PRE,
            },
            // `grad(U)` is read only by a scheme that carries a deferred
            // correction, so the knob turns the convection entry with it.
            Knob {
                label: "gradSchemes/grad(U)",
                file: "system/fvSchemes",
                from: "gradSchemes\n{\n    default         Gauss linear;\n}\n\ndivSchemes\n{\n    default         none;\n// The convecting flux of the two-phase momentum equation is the MASS\n// flux rhoPhi (SPEC-LIT S20.3), and it is not phi.\n    div(rhoPhi,U)    Gauss upwind;",
                to: "gradSchemes\n{\n    default         Gauss linear;\n    grad(U)         cellLimited Gauss linear 1;\n}\n\ndivSchemes\n{\n    default         none;\n    div(rhoPhi,U)    Gauss linearUpwind grad(U);",
                pre: NO_PRE,
            },
            Knob {
                label: "PIMPLE/nCorrectors",
                file: "system/fvSolution",
                from: "    nCorrectors     3;",
                to: "    nCorrectors     1;",
                pre: NO_PRE,
            },
            Knob {
                label: "PIMPLE/momentumPredictor",
                file: "system/fvSolution",
                from: "    momentumPredictor yes;",
                to: "    momentumPredictor no;",
                pre: NO_PRE,
            },
            // 0.01 rather than a round fraction of 0.5: over the four
            // 0.5 ms steps this test runs, the released column reaches an
            // alpha Courant number of about 0.024, so anything above that is
            // below BOTH limits and one sub-cycle satisfies them equally -
            // the pair would be bit-identical for a reason that has nothing
            // to do with whether the entry is read. 0.01 forces three
            // sub-cycles, comfortably inside maxAlphaSubCycles.
            Knob {
                label: "PIMPLE/maxAlphaCo",
                file: "system/fvSolution",
                from: "    maxAlphaCo      0.5;",
                to: "    maxAlphaCo      0.01;",
                pre: NO_PRE,
            },
            // SPEC-LIT 39. The contact angle is the only wall setting an
            // `alpha` file can carry, and before 39 the solver had no model
            // for it at all: `vofFaceUnitNormalBoundary` wrote a literal zero
            // on every wall face, which IS ninety degrees. Two knobs, because
            // a reader can wire the condition and still drop the angle.
            Knob {
                label: "0/alpha.water walls constantAlphaContactAngle",
                file: "0/alpha.water",
                from: "    leftWall\n    {\n        type            zeroGradient;\n    }\n    lowerWall\n    {\n        type            zeroGradient;\n    }\n    rightWall\n    {\n        type            zeroGradient;\n    }",
                to: "    leftWall\n    {\n        type            constantAlphaContactAngle;\n        theta0          45;\n    }\n    lowerWall\n    {\n        type            constantAlphaContactAngle;\n        theta0          45;\n    }\n    rightWall\n    {\n        type            constantAlphaContactAngle;\n        theta0          45;\n    }",
                pre: NO_PRE,
            },
            // The SAME condition on both sides, differing in one angle. The
            // condition goes in through `pre`, so the pair differs in
            // leftWall's `theta0` alone.
            Knob {
                label: "0/alpha.water theta0",
                file: "0/alpha.water",
                from: "        type            constantAlphaContactAngle;\n        theta0          45;",
                to: "        type            constantAlphaContactAngle;\n        theta0          135;",
                pre: ("0/alpha.water", "    leftWall\n    {\n        type            zeroGradient;\n    }\n    lowerWall\n    {\n        type            zeroGradient;\n    }\n    rightWall\n    {\n        type            zeroGradient;\n    }", "    leftWall\n    {\n        type            constantAlphaContactAngle;\n        theta0          45;\n    }\n    lowerWall\n    {\n        type            constantAlphaContactAngle;\n        theta0          45;\n    }\n    rightWall\n    {\n        type            constantAlphaContactAngle;\n        theta0          45;\n    }"),
            },
            // Static against dynamic, on the lowerWall - where the surge
            // front runs, so there is a contact line actually moving for the
            // correlation to see. Anchored on the patch NAME because `edit`
            // replaces the first match and leftWall comes first in the file.
            Knob {
                label: "0/alpha.water dynamicAlphaContactAngle",
                file: "0/alpha.water",
                from: "    lowerWall\n    {\n        type            constantAlphaContactAngle;\n        theta0          45;\n    }",
                to: "    lowerWall\n    {\n        type            dynamicAlphaContactAngle;\n        theta0          45;\n        correlation     JiangOhSlattery;\n    }",
                pre: ("0/alpha.water", "    leftWall\n    {\n        type            zeroGradient;\n    }\n    lowerWall\n    {\n        type            zeroGradient;\n    }\n    rightWall\n    {\n        type            zeroGradient;\n    }", "    leftWall\n    {\n        type            constantAlphaContactAngle;\n        theta0          45;\n    }\n    lowerWall\n    {\n        type            constantAlphaContactAngle;\n        theta0          45;\n    }\n    rightWall\n    {\n        type            constantAlphaContactAngle;\n        theta0          45;\n    }"),
            },
            Knob {
                label: "PIMPLE/nAlphaLimiterIters",
                file: "system/fvSolution",
                from: "    nAlphaLimiterIters 3;",
                to: "    nAlphaLimiterIters 0;",
                pre: NO_PRE,
            },
            Knob {
                label: "PIMPLE/cAlpha",
                file: "system/fvSolution",
                from: "    cAlpha          1;",
                to: "    cAlpha          0;",
                pre: NO_PRE,
            },
            Knob {
                label: "relaxationFactors/equations/U",
                file: "system/fvSolution",
                from: "        U               1;",
                to: "        U               0.4;",
                pre: NO_PRE,
            },
            Knob {
                label: "solvers/p_rgh/tolerance",
                file: "system/fvSolution",
                from: "        tolerance       1e-09;\n        relTol          0.001;",
                to: "        tolerance       1e-02;\n        relTol          0.5;",
                pre: NO_PRE,
            },
            // The whole algorithm dictionary under its OTHER legal name.
            // `AlgorithmControls::read` accepts all three; this module used
            // to look up the literal `PIMPLE/...` strings, so every entry
            // below moved to `PISO` fell back to `VofControls::default()`.
            Knob {
                label: "the algorithm dictionary spelled PISO rather than PIMPLE",
                file: "system/fvSolution",
                from: "PIMPLE\n{\n    momentumPredictor yes;\n    nCorrectors     3;",
                to: "PISO\n{\n    momentumPredictor no;\n    nCorrectors     1;",
                pre: NO_PRE,
            },
            // Honoured as of this sweep: `controlDict` used to be read for
            // `deltaT` and `maxDeltaT` only, so a case asking for an adaptive
            // step ran fixed unless `-maxCo` was typed.
            Knob {
                label: "controlDict/adjustTimeStep + maxCo",
                file: "system/controlDict",
                from: "deltaT          0.0002;",
                to: "deltaT          0.0002;\nadjustTimeStep  yes;\nmaxCo           0.02;",
                pre: NO_PRE,
            },
            Knob {
                label: "constant/transportProperties sigma",
                file: "constant/transportProperties",
                from: "sigma           [1 0 -2 0 0 0 0] 0.0728;",
                to: "sigma           [1 0 -2 0 0 0 0] 0.5;",
                pre: NO_PRE,
            },
            Knob {
                label: "constant/transportProperties mu (water)",
                file: "constant/transportProperties",
                from: "    mu              [1 -1 -1 0 0 0 0] 1.002e-03;",
                to: "    mu              [1 -1 -1 0 0 0 0] 1.002e-01;",
                pre: NO_PRE,
            },
            Knob {
                label: "constant/g",
                file: "constant/g",
                from: "(0 -9.81 0)",
                to: "(0 -1.62 0)",
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

    /// 13.4.1's one admissible exception, on the controls the solver is
    /// CONSTRUCTED from and through the same `from_case` the driver calls.
    ///
    /// `laplacianSchemes` must WIN over `snGradSchemes` where the case names
    /// one - `resolve_sn_grad`'s rule, which this module bypassed by reading
    /// `snGradSchemes/default` directly.
    #[test]
    fn the_pressure_gradient_and_the_laplacian_entry_reach_the_controls() {
        let case = dam_break("lap");
        apply(
            &case,
            &Knob {
                label: "gradSchemes/grad(p_rgh) and laplacianSchemes/default",
                file: "system/fvSchemes",
                from: "gradSchemes\n{\n    default         Gauss linear;\n}",
                to: "gradSchemes\n{\n    default         Gauss linear;\n    grad(p_rgh)     leastSquares;\n}",
                pre: NO_PRE,
            },
            true,
        );
        apply(
            &case,
            &Knob {
                label: "laplacianSchemes/default",
                file: "system/fvSchemes",
                from: "    default         Gauss linear uncorrected;",
                to: "    default         Gauss linear corrected;",
                pre: NO_PRE,
            },
            true,
        );

        let c = VofControls::from_case(&case).expect("controls");
        assert_eq!(c.grad_p.describe(), "leastSquares");
        assert_eq!(
            c.grad_u.describe(),
            "Gauss linear",
            "grad(p_rgh) must not move the momentum equation's gradient"
        );
        assert_eq!(
            c.sn_grad,
            ofgpu::fv::SnGradScheme::Corrected,
            "laplacianSchemes must win over snGradSchemes/default uncorrected"
        );
    }

    // ----------------------------------------------------------------------
    //  SPEC-LIT 13.4: recognised, not implemented -> a named error
    // ----------------------------------------------------------------------

    fn refusal(k: Knob, tag: &str) -> String {
        let case = dam_break(tag);
        apply(&case, &k, true);
        let e = VofControls::from_case(&case)
            .err()
            .unwrap_or_else(|| panic!("{} must be refused", k.label));
        format!("{e}")
    }

    /// The `bounded` prefix used to be parsed and dropped. The substituted
    /// answer is the RIGHT one here (§20.3), which is exactly why it has to
    /// be said out loud rather than applied in silence.
    #[test]
    fn a_bounded_prefix_on_the_momentum_convection_is_a_named_error() {
        let msg = refusal(
            Knob {
                label: "divSchemes/div(rhoPhi,U) bounded",
                file: "system/fvSchemes",
                from: "div(rhoPhi,U)    Gauss upwind;",
                to: "div(rhoPhi,U)    bounded Gauss upwind;",
                pre: NO_PRE,
            },
            "bnd",
        );
        assert!(msg.contains("div(rhoPhi,U)"), "must name the entry: {msg}");
        assert!(msg.contains("bounded"), "must name what was written: {msg}");
        assert!(msg.contains("20.3") || msg.contains("§20.3"), "must name the section: {msg}");
    }

    #[test]
    fn a_ddt_scheme_other_than_euler_is_a_named_error() {
        let msg = refusal(
            Knob {
                label: "ddtSchemes/default",
                file: "system/fvSchemes",
                from: "ddtSchemes\n{\n    default         Euler;\n}",
                to: "ddtSchemes\n{\n    default         backward;\n}",
                pre: NO_PRE,
            },
            "ddt",
        );
        assert!(msg.contains("ddtSchemes"), "must name the entry: {msg}");
        assert!(msg.contains("Euler"), "must name what IS available: {msg}");
    }

    #[test]
    fn more_than_one_outer_corrector_is_a_named_error() {
        let msg = refusal(
            Knob {
                label: "PIMPLE/nOuterCorrectors",
                file: "system/fvSolution",
                from: "    nCorrectors     3;",
                to: "    nCorrectors     3;\n    nOuterCorrectors 2;",
                pre: NO_PRE,
            },
            "outer",
        );
        assert!(msg.contains("nOuterCorrectors"), "must name the entry: {msg}");
        assert!(msg.contains("nCorrectors"), "must name what IS available: {msg}");
    }

    #[test]
    fn relaxing_the_pressure_is_a_named_error() {
        let msg = refusal(
            Knob {
                label: "relaxationFactors/fields/p_rgh",
                file: "system/fvSolution",
                from: "relaxationFactors\n{\n    equations\n    {",
                to: "relaxationFactors\n{\n    fields\n    {\n        p_rgh           0.3;\n    }\n\n    equations\n    {",
                pre: NO_PRE,
            },
            "prelax",
        );
        assert!(msg.contains("p_rgh"), "must name the entry: {msg}");
        assert!(msg.contains("PISO"), "must say why: {msg}");
    }

    #[test]
    fn residual_control_is_a_named_error() {
        let msg = refusal(
            Knob {
                label: "PIMPLE/residualControl",
                file: "system/fvSolution",
                from: "    nCorrectors     3;",
                to: "    nCorrectors     3;\n    residualControl { p_rgh 1e-4; U 1e-4; }",
                pre: NO_PRE,
            },
            "rc",
        );
        assert!(msg.contains("residualControl"), "must name the entry: {msg}");
        assert!(msg.contains("-endTime"), "must name what stops the run: {msg}");
    }
}
