// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! `ofgpu-automesher` - this crate's own mesher, SPEC-LIT §92's hex-dominant
//! path.
//!
//! ```text
//! ofgpu-automesher <config.json> [-schema] [-check <caseDir>] [-dryRun]
//! ```
//!
//! Tranche 1 unit 1 is the SKELETON of the pipeline: everything around the
//! meshing runs for real - the config is read and validated, the STLs are
//! read, merged and required closed, the surface summary and the plan are
//! printed, and §92.2 stage 0's margin check refuses a domain that cannot
//! hold the geometry - and then the run stops at the first stage that is
//! not built yet:
//!
//! ```text
//! error: not implemented: stage 1 (octree refinement) - SPEC-LIT §92.2;
//! tranche 1 unit 2
//! ```
//!
//! A mesher that promised more than its stages deliver would be the §92.1
//! failure again from the other side, so the refusal is the contract: no
//! file is written, and exit status 1 says the run did not mesh.
//!
//! The modes:
//!
//! - default: the path above - everything up to stage 1's refusal.
//! - `-dryRun`: the same path, stopping after the surface summary with exit
//!   0 - what a config check wants before a long run is queued.
//! - `-check <caseDir>`: §92.3's gate on a mesh that ALREADY exists, read
//!   from `<caseDir>/constant/polyMesh`, with the config's thresholds. This
//!   mode is complete: it prints the measured numbers and exits 0, or prints
//!   the refusal and exits 1. A gate failure is not a crash; its text is the
//!   report §92.3 fixed the format of.
//! - `-schema`: the JSON Schema of the config, generated from the SAME
//!   types that parse it, on stdout with exit 0. No config is read, and
//!   every other argument is ignored.
//!
//! Provenance: ORIGINAL - the command-line front end to
//! `automesher` (SPEC-LIT §92's config tree and §92.3's gate, both covered
//! by that module's own header) and to `surface::stl` and `io::polymesh`,
//! whose readers are covered by their files. This file is argument parsing,
//! the surface summary, stage 0's margin check, the plan, and the refusal.
//! No GPL-licensed source was consulted.

use std::path::Path;
use std::process::ExitCode;

use ofgpu::automesher::{self, AutomeshConfig};
use ofgpu::io::polymesh::read_poly_mesh;
use ofgpu::surface::{stl::read_stl, Surface};
use ofgpu::{Error, Result, Scalar};

fn usage() {
    eprintln!(
        "usage: ofgpu-automesher <config.json> [-schema] [-check <caseDir>] [-dryRun]"
    );
}

fn run(args: &[String]) -> Result<()> {
    // ---- flags and positionals --------------------------------------------
    let mut config_arg: Option<&String> = None;
    let mut schema = false;
    let mut check_dir: Option<&String> = None;
    let mut dry_run = false;

    let mut i = 1usize;
    while i < args.len() {
        match args[i].as_str() {
            "-schema" => schema = true,
            "-dryRun" => dry_run = true,
            "-check" => {
                i += 1;
                let Some(v) = args.get(i) else {
                    usage();
                    return Err(Error::Config(
                        "-check needs a <caseDir> argument".to_string(),
                    ));
                };
                check_dir = Some(v);
            }
            a if a.starts_with('-') => {
                usage();
                return Err(Error::Config(format!("unknown argument '{a}'")));
            }
            a => {
                if config_arg.is_some() {
                    usage();
                    return Err(Error::Config(format!(
                        "one <config.json> positional is taken, got a second one '{a}'"
                    )));
                }
                config_arg = Some(&args[i]);
            }
        }
        i += 1;
    }

    // -schema needs no config: the schema is generated from the SAME types
    // that parse one, so it exists before any file is read. Every other
    // argument on the line is ignored, the way a schema request reads.
    if schema {
        println!("{}", automesher::emit_schema());
        return Ok(());
    }

    let Some(config_arg) = config_arg else {
        usage();
        return Err(Error::Config(
            "a <config.json> positional is required, or pass -schema".to_string(),
        ));
    };
    let cfg = automesher::read_config(Path::new(config_arg))?;
    cfg.validate()?;

    if let Some(case_dir) = check_dir {
        return check_mode(&cfg, Path::new(case_dir));
    }
    meshing_mode(&cfg, dry_run)
}

/// `-check <caseDir>`: §92.3's gate on a mesh that already exists, at
/// `<caseDir>/constant/polyMesh`, measured against THIS config's thresholds.
/// The one mode that is complete today: `check` returns the report when every
/// gate passed, and the refusal - the report §92.3 fixed the format of - as
/// `Error::Mesh` when one failed, which `main` prints with exit status 1.
fn check_mode(cfg: &AutomeshConfig, case_dir: &Path) -> Result<()> {
    let raw = read_poly_mesh(case_dir)?;
    let t = cfg.quality.thresholds();
    let rep = automesher::quality::check(&raw, &t)?;
    // `summary` ends mid-line by design; this is the whole run's output.
    println!("{}", rep.summary());
    Ok(())
}

/// The meshing path: everything the pipeline's front half does, then the
/// refusal of the first stage that is not built.
fn meshing_mode(cfg: &AutomeshConfig, dry_run: bool) -> Result<()> {
    println!("ofgpu-automesher (SPEC-LIT §92, hex-dominant path)");

    let surf = read_input_surface(cfg)?;
    print_surface_summary(&surf);
    if dry_run {
        println!(
            "ofgpu-automesher: dry run - the surface stages ran; meshing was \
             not attempted."
        );
        return Ok(());
    }

    // Stage 0 (§92.2): the domain must hold the surface with a margin before
    // any work is done, and the plan is what stage 1 inherits.
    check_domain(cfg, &surf)?;
    print_plan(cfg);

    Err(Error::Mesh(
        "not implemented: stage 1 (octree refinement) - SPEC-LIT §92.2; \
         tranche 1 unit 2"
            .to_string(),
    ))
}

/// Read every `input.surfaces[]` entry and merge them into the ONE surface
/// the pipeline casts against. A config `name` replaces a file's own solid
/// names wholesale, the way `ofgpu-generate-mesh -stl name=path` does (§23.1
/// patch identity). An open surface is refused by `require_closed`, which
/// names the open- and non-manifold-edge counts `edge_defects` measured:
/// castellation classifies inside/outside by parity (§23.3), and parity
/// through a hole is a coin toss.
fn read_input_surface(cfg: &AutomeshConfig) -> Result<Surface> {
    let mut parts = Vec::with_capacity(cfg.input.surfaces.len());
    for s in &cfg.input.surfaces {
        let mut part = read_stl(Path::new(&s.path))?;
        if let Some(name) = &s.name {
            part.patch_names = vec![name.clone()];
            part.tri_patch = vec![0; part.tris.len()];
            part.patch_area = vec![part.patch_area.iter().sum()];
        }
        println!(
            "[stl] {}: {} triangle(s), {} patch(es){}",
            s.path,
            part.tris.len(),
            part.patch_names.len(),
            if part.degenerate_dropped > 0 {
                format!(", {} degenerate dropped", part.degenerate_dropped)
            } else {
                String::new()
            }
        );
        parts.push(part);
    }
    let surf = if parts.len() == 1 {
        parts.pop().expect("len checked above")
    } else {
        Surface::merge(parts)?
    };
    surf.require_closed()?;
    Ok(surf)
}

/// The surface summary: sizes, the bounding box, and where each patch's
/// triangles and area sit - the numbers every later stage is judged against.
fn print_surface_summary(surf: &Surface) {
    let (lo, hi) = surf.bbox;
    println!(
        "surface: {} triangle(s), {} point(s)",
        surf.tris.len(),
        surf.points.len()
    );
    println!(
        "  bbox lo ({:.3}, {:.3}, {:.3}), hi ({:.3}, {:.3}, {:.3})",
        lo.x, lo.y, lo.z, hi.x, hi.y, hi.z
    );
    let mut tris_per_patch = vec![0usize; surf.patch_names.len()];
    for &p in &surf.tri_patch {
        tris_per_patch[p as usize] += 1;
    }
    for (p, name) in surf.patch_names.iter().enumerate() {
        println!(
            "  {name}: {} triangles, {:.3} m^2",
            tris_per_patch[p], surf.patch_area[p]
        );
    }
}

/// §92.2 stage 0: the background block and the surface have to hold each
/// other with at least one `base_size` of room, or there is no cell layer
/// for the octree to refine against the surface from, and the run refuses
/// naming the axis and the two numbers before any work is done.
///
/// The two ways that holds, and the one way it does not:
///
/// - the domain contains the surface's bounding box - the classic setup.
///   The margin is REQUIRED here: a surface within one base cell of the
///   extent gives castellation no fluid cell to refine from.
/// - the surface contains the domain - the site-solid setup, which is what
///   `tools/automesher/examples/nh3_site.json` deliberately does (its
///   comment: the terrain slab is trimmed at sea level and the fluid box
///   starts a margin above it, so a box that TOUCHED the solid would
///   castellate against a coincident face). The part of the solid outside
///   the domain bounds no fluid cell, so the poke-out is said in the log,
///   not refused.
/// - neither holds - a surface that sticks out of a domain it does not
///   enclose. This is the defect stage 0 exists for: castellation would cut
///   the geometry off at the wall with nothing to refine from.
fn check_domain(cfg: &AutomeshConfig, surf: &Surface) -> Result<()> {
    let (lo, hi) = surf.bbox;
    let (dlo, dhi) = cfg.extent_bounds();
    let s_lo = [lo.x, lo.y, lo.z];
    let s_hi = [hi.x, hi.y, hi.z];
    let d_lo = [dlo.x, dlo.y, dlo.z];
    let d_hi = [dhi.x, dhi.y, dhi.z];
    let margin = cfg.domain.base_size as Scalar;
    let axes = ["x", "y", "z"];

    // The check is PER AXIS, and two arrangements are both legitimate:
    //
    //   (a) the surface lies inside the domain with at least one base cell of
    //       margin on both sides - a building in a wind tunnel;
    //   (b) the surface runs past the domain on BOTH sides - the ammonia
    //       site, whose terrain slab is 2.5 km wide and whose fluid box is a
    //       window cut out of it.
    //
    // What is refused is the third arrangement: the surface ENDS inside the
    // domain without the margin, leaving a hanging edge for castellation to
    // cut against. A per-axis rule matters because the two mix - a ground
    // plane that spans past in x and y under a domain that reaches above the
    // geometry in z is case (b) twice and case (a) once, and an
    // all-axes-or-nothing rule refuses it.
    let mut spanning: Vec<&str> = Vec::new();
    for a in 0..3 {
        let inside = s_lo[a] >= d_lo[a] + margin && s_hi[a] <= d_hi[a] - margin;
        let spans = s_lo[a] <= d_lo[a] && s_hi[a] >= d_hi[a];
        if spans {
            spanning.push(axes[a]);
            continue;
        }
        if inside {
            continue;
        }
        return Err(Error::Mesh(format!(
            "domain.extent: on {} the surface neither sits inside the \
             domain with one base_size of margin nor spans past it on \
             both sides - surface [{:.3}, {:.3}], domain [{:.3}, \
             {:.3}], base_size {:.3}. A surface that ends inside the \
             domain leaves an edge with no solid behind it, and \
             castellation would cut against it (SPEC-LIT §92.2 stage 0)",
            axes[a], s_lo[a], s_hi[a], d_lo[a], d_hi[a], margin
        )));
    }
    if !spanning.is_empty() {
        println!(
            "stage 0: the surface runs past the domain on {} - the \
             site-solid setup: the part outside bounds no fluid cell",
            spanning.join(", ")
        );
    }
    Ok(())
}

/// The plan: the base grid stage 0 builds, the level cap, and the finest
/// cell the octree can reach (`base_size / 2^max_level`).
fn print_plan(cfg: &AutomeshConfig) {
    let e = &cfg.domain.extent;
    let h = cfg.domain.base_size;
    let cells = |lo: f64, hi: f64| ((hi - lo) / h).round();
    let finest = h / 2.0f64.powi(cfg.refinement.max_level as i32);
    println!(
        "plan: base grid {} x {} x {} cells (base_size {:.3} m), \
         max level {}, finest cell {:.5} m",
        cells(e[0], e[1]),
        cells(e[2], e[3]),
        cells(e[4], e[5]),
        h,
        cfg.refinement.max_level,
        finest
    );
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();

    match run(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("\nerror: {e}");
            ExitCode::from(1)
        }
    }
}
