// meteor-cfd - Copyright (c) 2026 주식회사 메테오시뮬레이션 (Meteo Simulation Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

//! `ofgpu-generate-mesh` - ready-to-run test meshes for ofgpu.
//!
//! ```text
//! ofgpu-generate-mesh <case> <outputDir> [nx ny nz] [-stl [name=]path]... [-permissive]
//! ofgpu-generate-mesh big <outputDir> [n] [-stl ...]  (cube of n per side)
//!
//! channel  plane channel, graded to both walls, 2-D (200 x 120 x 1)
//! cavity   lid-driven cavity, 2-D                   (128 x 128 x 1)
//! step     backward-facing-step BOX, 2-D            (300 x 100 x 1)
//! big      uniform benchmark box                    (160 x 160 x 160)
//! plume    fire plume, floor burner, xMax outlet    (98 x 42 x 20)
//! damBreak two-phase collapsing column, 2-D           (150 x 90 x 1)
//! ```
//!
//! Writes `<outputDir>/constant/polyMesh` and, if they do not already exist,
//! a minimal `<outputDir>/system` so that `checkMesh` can be pointed at the
//! result by anyone who does have an OpenFOAM installation.
//!
//! `-stl [name=]path` (repeatable, any case kind) carves the block against
//! the union of the given triangulated surfaces - SPEC-LIT §23: the STLs are
//! read, merged into one surface with one patch per `solid` (or per file
//! stem for binary STL; `name=` overrides), validated for closure, the block
//! cells inside are removed, and the fluid-solid faces become new `wall`
//! patches named for the surface patches, carrying the same wall boundary
//! conditions the uncarved case writer gives its walls. `-permissive`
//! downgrades the closed-surface requirement (and every other §13.4
//! rejection) to a printed warning.
//!
//! Carried across from this project's own earlier C++ case generator. Every
//! geometric
//! decision the C++ `main` made — extents, grading, patch names and types,
//! the initial profiles — lives in [`ofgpu::blockgen`] in this port, so what
//! is left here really is only the command line.

use std::path::Path;
use std::process::ExitCode;

use ofgpu::blockgen::{write_carved_case, write_case, CaseKind};
use ofgpu::io::contract;
use ofgpu::surface::{stl::read_stl, Surface};
use ofgpu::{Error, Result};

#[path = "common/mod.rs"]
mod common;

use common::atoi;

fn usage() {
    eprintln!(
        "usage: ofgpu-generate-mesh <channel|cavity|step|big|plume|damBreak> <outputDir> \
         [nx ny nz] [-stl [name=]path]... [-permissive]\n       \
         ofgpu-generate-mesh big <outputDir> [n] [-stl ...]   # n^3 cells\n\
         {}",
        contract::PERMISSIVE_USAGE
    );
}

/// One `-stl` argument: an optional `name=` prefix and the file path.
///
/// The `=` split only happens when the prefix looks like a patch name - no
/// path separators, drive colons or dots - so a bare Windows path such as
/// `C:\models\a=b.stl` is never mistaken for a rename.
fn split_stl_arg(arg: &str) -> (Option<&str>, &str) {
    if let Some((name, path)) = arg.split_once('=') {
        if !name.is_empty()
            && !name.contains(['/', '\\', ':', '.'])
        {
            return (Some(name), path);
        }
    }
    (None, arg)
}

/// Read and merge every `-stl` argument into one surface, applying any
/// `name=` overrides (§23.1: the override replaces that FILE's patch
/// identity wholesale, which is what naming a whole file means).
fn read_surfaces(stl_args: &[String]) -> Result<Surface> {
    let mut parts = Vec::with_capacity(stl_args.len());
    for arg in stl_args {
        let (name, path) = split_stl_arg(arg);
        let mut s = read_stl(Path::new(path))?;
        if let Some(name) = name {
            s.patch_names = vec![name.to_string()];
            s.tri_patch = vec![0; s.tris.len()];
            s.patch_area = vec![s.patch_area.iter().sum()];
        }
        println!(
            "[stl] {}: {} triangle(s), {} patch(es){}",
            path,
            s.tris.len(),
            s.patch_names.len(),
            if s.degenerate_dropped > 0 {
                format!(", {} degenerate dropped", s.degenerate_dropped)
            } else {
                String::new()
            }
        );
        parts.push(s);
    }
    Surface::merge(parts)
}

fn run(args: &[String]) -> Result<()> {
    // ---- flags and positionals --------------------------------------------
    let mut positional: Vec<&String> = Vec::new();
    let mut stl_args: Vec<String> = Vec::new();
    let mut i = 1usize;
    while i < args.len() {
        match args[i].as_str() {
            "-permissive" => contract::set_permissive(true),
            "-stl" => {
                i += 1;
                let Some(v) = args.get(i) else {
                    usage();
                    return Err(Error::Config("-stl needs a [name=]path argument".to_string()));
                };
                stl_args.push(v.clone());
            }
            _ => positional.push(&args[i]),
        }
        i += 1;
    }

    if positional.len() < 2 {
        usage();
        return Err(Error::Config("too few arguments".to_string()));
    }

    let name = positional[0].as_str();
    let dir = positional[1].as_str();

    let kind = CaseKind::from_name(name).ok_or_else(|| {
        usage();
        Error::Config(format!("generate_cases: unknown case '{name}'"))
    })?;

    let (mut nx, mut ny, mut nz) = kind.default_resolution();

    // One trailing argument means "cube of n per side", and only for `big`:
    // the 2-D cases have an `empty` front and back that a third dimension
    // would make illegal.
    if kind == CaseKind::Big && positional.len() == 3 {
        let n = atoi(positional[2]);
        nx = clamp_dim(n)?;
        ny = nx;
        nz = nx;
    }

    if positional.len() >= 5 {
        nx = clamp_dim(atoi(positional[2]))?;
        ny = clamp_dim(atoi(positional[3]))?;
        nz = clamp_dim(atoi(positional[4]))?;
    }

    if stl_args.is_empty() {
        return write_case(Path::new(dir), kind, nx, ny, nz);
    }

    // ---- the carved path (SPEC-LIT §23) -------------------------------------
    let surface = read_surfaces(&stl_args)?;
    println!(
        "[stl] merged: {} triangle(s), {} patch(es): {}",
        surface.tris.len(),
        surface.patch_names.len(),
        surface.patch_names.join(", ")
    );

    let s = write_carved_case(Path::new(dir), kind, nx, ny, nz, &surface)?;

    println!(
        "[carve] cells: {} block -> {} fluid / {} solid ({} settled by 3-axis vote, \
         {} arbitrated by winding number)",
        s.n_cells_block, s.n_fluid, s.n_solid, s.voted, s.arbitrated
    );
    println!(
        "[carve] faces: {} internal, {} kept on domain patches",
        s.n_internal_faces, s.n_domain_faces
    );
    if s.wall_faces.is_empty() {
        println!("[carve] no new wall faces - the surface encloses no cell centres");
    } else {
        for (patch, n) in &s.wall_faces {
            println!("[carve] new wall patch {patch}: {n} face(s)");
        }
    }

    Ok(())
}

/// A resolution has to be a positive `usize` before it can index anything,
/// and the C++ rejected non-positive values with the same message.
fn clamp_dim(v: i64) -> Result<usize> {
    if v < 1 {
        return Err(Error::Config(format!(
            "generate_cases: bad resolution component {v}"
        )));
    }
    Ok(v as usize)
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
