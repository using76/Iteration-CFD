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

use ofgpu::error::IoContext;
use ofgpu::field_setup::{
    harvest_scalar_field, harvest_vector_field, setup_scalar_field, setup_vector_field,
};
use ofgpu::io::case::{find_start_time, format_time_name};
use ofgpu::io::dict::FoamDict;
use ofgpu::io::fields::{
    read_scalar_field, read_vector_field, write_scalar_field, write_vector_field,
};
use ofgpu::io::polymesh::{build_host_mesh, read_poly_mesh};
use ofgpu::vof::{phase_names, Vof, VofControls, VofProperties};
use ofgpu::{Error, Gpu, GpuMesh, HostMesh, Result, Scalar, Vec3};

#[path = "common/mod.rs"]
mod common;

use common::{device_banner, g, next_arg, sci};

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
}

fn usage() {
    eprintln!(
        "usage: ofgpu-vof <caseDir> [-endTime T] [-deltaT dt] [-maxCo C] \
         [-maxDeltaT dt]\n       \
         [-writeInterval W] [-noWrite] [-surge] [-a A]\n\
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
    let max_delta_t = if o.max_delta_t > 0.0 {
        o.max_delta_t as Scalar
    } else {
        cd.scalar("maxDeltaT", Scalar::INFINITY)
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
    println!("div(rhoPhi,U) {}", ctrl.div_scheme.describe());

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

    // A flux to start from. The first pressure correction makes it
    // conservative; this only decides how much work that correction has to do,
    // and starting from a velocity of zero it is exactly zero anyway.
    vof.initialise_flux_from_velocity(&gpu)?;
    vof.initialise(&gpu)?;

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
    let mut t = 0.0 as Scalar;
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
        } else if step % 20 == 0 || t >= end_time {
            let (lo, hi) = vof.alpha_bounds(&gpu)?;
            println!(
                "t = {:>10}  dt {:>10}  alphaCo {:>8}  x{} sub  p_rgh {} -> {} in {} \
                 iters  continuity {}  alpha [{}, {}]",
                g(f64::from(t)),
                g(f64::from(dt)),
                g(f64::from(perf.alpha_courant)),
                perf.n_sub_cycles,
                sci(f64::from(perf.p_rgh.initial_residual), 2),
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
        if o.max_co > 0.0 && perf.alpha_courant > 0.0 {
            let want = dt * (o.max_co as Scalar) / perf.alpha_courant;
            // Never more than a 20 % rise in one step: an adaptive step that
            // doubles lands on a state the previous one has not settled into.
            dt = want.min(1.2 * dt).min(max_delta_t);
        }

        if o.do_write && t + 1e-12 >= next_write {
            write_time(&gpu, &vof, &hm, case, t, &mut raw_alpha, &mut raw_u, &mut raw_p)?;
            next_write += write_interval;
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
        write_time(&gpu, &vof, &hm, case, t, &mut raw_alpha, &mut raw_u, &mut raw_p)?;
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn write_time(
    gpu: &Gpu,
    vof: &Vof<'_>,
    hm: &HostMesh,
    case: &Path,
    t: Scalar,
    raw_alpha: &mut ofgpu::io::fields::RawScalarField,
    raw_u: &mut ofgpu::io::fields::RawVectorField,
    raw_p: &mut ofgpu::io::fields::RawScalarField,
) -> Result<()> {
    let name = format_time_name(t);
    let dir = case.join(&name);
    std::fs::create_dir_all(&dir).path(&dir)?;

    harvest_scalar_field(gpu, raw_alpha, vof.alpha(), hm)?;
    harvest_vector_field(gpu, raw_u, vof.u(), hm)?;
    harvest_scalar_field(gpu, raw_p, vof.p_rgh(), hm)?;

    write_scalar_field(&dir.join(&raw_alpha.name), raw_alpha, &name)?;
    write_vector_field(&dir.join("U"), raw_u, &name)?;
    write_scalar_field(&dir.join("p_rgh"), raw_p, &name)?;

    println!("  written {}", dir.display());
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
