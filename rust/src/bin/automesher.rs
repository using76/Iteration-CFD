// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! `ofgpu-automesher` - this crate's own mesher, SPEC-LIT §92's hex-dominant
//! path, all of it: §92.14's driver runs the four stages of (92.55) behind
//! one command, and this file is the command.
//!
//! ```text
//! ofgpu-automesher <config.json> [-stopAfter STAGE] [-tag NAME]
//!                                 [-check [<caseDir>]] [-dryRun] [-schema]
//! ```
//!
//! The modes:
//!
//! - default: the config is read and validated, the STLs are read, merged
//!   and required closed, the surface summary and the plan are printed, and
//!   §92.2 stage 0's margin check refuses a domain that cannot hold the
//!   geometry - then the stages run. Each one prints its banner BEFORE it
//!   runs (§92.14.1) and gates its own output before returning it (§92.3);
//!   the run writes `<case_dir>/constant/polyMesh` and
//!   `<case_dir>/<name>_summary.json` only after the last stage returned
//!   Ok, so a refusal - a failed gate, a bad `output.patch_names` map -
//!   writes NOTHING and exits 1 (§92.14.4).
//! - `-stopAfter STAGE`: (92.55)'s stop rule. The stages up to and
//!   including STAGE run and the mesh that stage returned is written and
//!   exits 0. STAGE is one of `octree`, `castellate`, `snap`, `layers`,
//!   with `features` accepted as a spelling of `snap`: §92.12 folded the
//!   feature attraction into snap's own loop, so no mesh exists between
//!   them. A stopped run is not a way to get an ungated mesh out - every
//!   stop point has already passed the gate.
//! - `-tag NAME`: this run's output is its own - the case directory becomes
//!   `<output.case_dir>_<NAME>` and the name `<output.name>_<NAME>`,
//!   `tools/mesh/step_mesh.py --tag`'s convention adapted to a case
//!   directory, so two tagged runs of one config do not overwrite each
//!   other's `constant/polyMesh`. NAME is a suffix, not a path: empty, or
//!   carrying `/` or `\`, is refused.
//! - `-check [<caseDir>]`: §92.3's gate on a polyMesh that ALREADY exists,
//!   read from `<caseDir>/constant/polyMesh` with the config's thresholds.
//!   The directory is OPTIONAL: it is consumed when the next token exists,
//!   does not start with `-`, and the `<config.json>` positional has
//!   already been read; otherwise `-check` takes no argument and the
//!   config's own (tag-adjusted) `output.case_dir` is checked. It measures
//!   and prints and does not write: exit 0, or the refusal and exit 1.
//!   A gate failure is not a crash; its text is the report §92.3 fixed the
//!   format of.
//! - `-dryRun`: everything up to the surface summary with exit 0 - what a
//!   config check wants before a long run is queued.
//! - `-schema`: the JSON Schema of the config, generated from the SAME
//!   types that parse it, on stdout with exit 0. No config is read, and
//!   every other argument is ignored.
//!
//! Provenance: ORIGINAL - the command-line front end to `automesher`'s
//! driver (SPEC-LIT §92.14) and to `surface::stl` and `io::polymesh`,
//! whose readers are covered by their files. This file is argument parsing,
//! the surface summary, stage 0's margin check, the plan, and the two
//! writes. No GPL-licensed source was consulted.

use std::io::Write as _;
use std::path::Path;
use std::process::ExitCode;
use std::time::Instant;

use ofgpu::automesher::{self, driver, AutomeshConfig};
use ofgpu::error::IoContext;
use ofgpu::io::polymesh::{read_poly_mesh, write_poly_mesh_raw};
use ofgpu::surface::{stl::read_stl, Surface};
use ofgpu::{Error, Result, Scalar};

fn usage() {
    eprintln!(
        "usage: ofgpu-automesher <config.json> [-stopAfter STAGE] [-tag NAME]
                        [-check [<caseDir>]] [-dryRun] [-schema]
  -stopAfter STAGE: the stop rule of SPEC-LIT §92.14 - the stages up to and
    including STAGE run and the mesh STAGE returned is written. STAGE is
    octree, castellate, snap or layers; features is a spelling of snap.
  -tag NAME: this run's output is its own - the case directory and the mesh
    name each gain _NAME, so two runs of one config do not overwrite each
    other. NAME is a suffix, not a path.
  -check [<caseDir>]: the quality gate on a mesh that already exists
    (SPEC-LIT §92.14.5). The directory is optional: it is consumed when the
    next token exists, does not start with a dash, and the config positional
    has already been read; without it the config's own (tag-adjusted) case
    directory is checked. It measures and prints and does not write.
  -dryRun: everything up to the surface summary, then exit 0.
  -schema: the config's JSON Schema on stdout, no config read."
    );
}

fn run(args: &[String]) -> Result<()> {
    // ---- flags and positionals --------------------------------------------
    let mut config_arg: Option<&String> = None;
    let mut schema = false;
    let mut check = false;
    let mut check_dir: Option<&String> = None;
    let mut dry_run = false;
    let mut stop_after: Option<&String> = None;
    let mut tag: Option<&String> = None;

    let mut i = 1usize;
    while i < args.len() {
        match args[i].as_str() {
            "-schema" => schema = true,
            "-dryRun" => dry_run = true,
            "-stopAfter" => {
                i += 1;
                let Some(v) = args.get(i) else {
                    usage();
                    return Err(Error::Config(
                        "-stopAfter needs a STAGE argument".to_string(),
                    ));
                };
                stop_after = Some(v);
            }
            "-tag" => {
                i += 1;
                let Some(v) = args.get(i) else {
                    usage();
                    return Err(Error::Config(
                        "-tag needs a NAME argument".to_string(),
                    ));
                };
                // The tag is a suffix of the case directory and the mesh
                // name, never a path of its own: an empty one, or one
                // carrying a separator, would smuggle a directory change
                // through a string concatenation.
                if v.is_empty() || v.contains('/') || v.contains('\\') {
                    usage();
                    return Err(Error::Config(format!(
                        "-tag: '{v}' is a NAME - a suffix, not a path; an \
                         empty NAME or one containing / or \\ is refused"
                    )));
                }
                tag = Some(v);
            }
            "-check" => {
                // §92.14.5: the directory is optional. It is consumed when
                // the next token exists, does not start with '-' and the
                // <config.json> positional has already been read; otherwise
                // -check takes no argument and checks the config's own
                // (tag-adjusted) case directory. Either way the flag itself
                // selects check mode.
                check = true;
                if let Some(v) = args.get(i + 1) {
                    if !v.starts_with('-') && config_arg.is_some() {
                        check_dir = Some(v);
                        i += 1;
                    }
                }
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
    let mut cfg = automesher::read_config(Path::new(config_arg))?;
    cfg.validate()?;

    // -tag gives this run an output of its own: both fields are suffixed
    // HERE, before anything downstream reads them, so everything from this
    // point on - check_mode's default directory, driver::run, and (92.57)'s
    // summary, which must record the directory that was actually written -
    // sees the adjusted names.
    if let Some(t) = tag {
        cfg.output.case_dir = format!("{}_{}", cfg.output.case_dir, t);
        cfg.output.name = format!("{}_{}", cfg.output.name, t);
    }

    // §92.14.5: -check measures a mesh that already exists - the directory
    // named after the flag, or the config's own (tag-adjusted)
    // output.case_dir when none was.
    if check {
        let dir = match check_dir {
            Some(d) => Path::new(d),
            None => Path::new(&cfg.output.case_dir),
        };
        return check_mode(&cfg, dir);
    }
    meshing_mode(&cfg, config_arg, dry_run, stop_after)
}

/// `-check <caseDir>`: §92.3's gate on a mesh that already exists, at
/// `<caseDir>/constant/polyMesh`, measured against THIS config's thresholds.
/// The run always prints the summary - the refusal a run most needs it is
/// the one that used to lose it - and then the refusal, the report §92.3
/// fixed the format of, as `Error::Mesh` when a gate failed, which `main`
/// prints with exit status 1.
fn check_mode(cfg: &AutomeshConfig, case_dir: &Path) -> Result<()> {
    println!(
        "ofgpu-automesher: checking {} (SPEC-LIT §92.14.5)",
        case_dir.display()
    );
    let raw = read_poly_mesh(case_dir)?;
    let t = cfg.quality.thresholds();
    let rep = automesher::quality::measure(&raw, &t)?;
    // `summary` ends mid-line by design; this is the whole run's output.
    println!("{}", rep.summary());
    if !rep.passed() {
        return Err(Error::Mesh(rep.refusal_text()));
    }
    Ok(())
}

/// The meshing path: everything the pipeline's front half does, then
/// §92.14's driver through the four stages of (92.55), and on Ok the two
/// writes - `constant/polyMesh` and (92.57)'s summary beside it. The config
/// arrives with `output.case_dir`/`output.name` already tag-adjusted.
fn meshing_mode(
    cfg: &AutomeshConfig,
    config_path: &str,
    dry_run: bool,
    stop_after: Option<&String>,
) -> Result<()> {
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

    // (92.55)'s stop rule. `Stage::parse`'s error message is already the
    // right one - it names the five spellings - so there is no second
    // parser here.
    let stop = stop_after.map(|s| driver::Stage::parse(s)).transpose()?;

    let t0 = Instant::now();
    // The driver owns the text of every progress line (§92.14.1); this
    // binary owns only where the lines go. The flush is the Windows
    // console's doing: a console line-buffers, a pipe block-buffers, and
    // run_automesher.cmd pipes the run into a log, so without it a stage
    // that takes forty minutes would put its banner on screen after it
    // returned instead of before it started.
    let out = driver::run(cfg, &surf, stop, &mut |line| {
        println!("{line}");
        let _ = std::io::stdout().flush();
    })?;

    // §92.14.4: nothing is written until the pipeline has returned Ok, so a
    // refusal anywhere above - a failed gate, a refused rename - leaves no
    // case directory and no summary behind. From here on every write is of
    // a mesh that left its stage through the gate.
    println!("{}", out.quality.summary());
    let case_dir = Path::new(&cfg.output.case_dir);
    let poly_dir = case_dir.join("constant").join("polyMesh");
    write_poly_mesh_raw(&poly_dir, &out.mesh)?;
    let summary_path = case_dir.join(format!("{}_summary.json", cfg.output.name));
    let summary = driver::summary_json(cfg, config_path, &surf, &out);
    let text = serde_json::to_string_pretty(&summary).map_err(|e| {
        Error::Config(format!("summary {}: {e}", summary_path.display()))
    })?;
    std::fs::write(&summary_path, text).path(&summary_path)?;

    println!(
        "ofgpu-automesher: wrote {} ({} cells)",
        poly_dir.display(),
        out.quality.n_cells
    );
    println!("ofgpu-automesher: wrote {}", summary_path.display());
    println!("ofgpu-automesher: total {:.1} s", t0.elapsed().as_secs_f64());
    Ok(())
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
