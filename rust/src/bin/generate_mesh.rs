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
//! `-cutcell` (only with at least one `-stl`; not `damBreak`, which has no
//! cut-cell VOF path yet) selects embedded-boundary cutting instead of
//! castellation - SPEC-LIT §24: intersected cells keep REDUCED volumes and
//! face areas rather than being removed whole, closed by one new cut face
//! per cut cell. `-s N` sets the supersample lattice size (default 16,
//! §24.2/§24.4) and `-thetaMin X` the small-cell merge threshold (default
//! 0.2, §24.5). Castellation stays the default when `-cutcell` is absent.
//!
//! `-cyclic x|y|z` may be given more than once (SPEC-LIT §34.2: a plane
//! channel needs two axes, a fully periodic box three) - each names a
//! DIFFERENT axis; naming the same axis twice is refused by
//! `BlockSpec::set_cyclic_axis` itself.
//!
//! `-wallModel <standard|spalding|rough|lowRe> [-Ks x [-Cs y]]` - SPEC-LIT
//! §29.1 route (c): expands the named preset into the `k`/`epsilon`/`omega`/
//! `nut` (and, for a case that solves `T`, the thermal wall function of
//! §29.3) types the generated `0/` directory writes, instead of the
//! hardcoded `standard` row `standard stays the default` when the flag is
//! absent. `rough` requires `-Ks`; `-Cs` defaults to `0.5`. Carved STL wall
//! patches follow the same preset as the block's own walls.
//!
//! Carried across from this project's own earlier C++ case generator. Every
//! geometric
//! decision the C++ `main` made — extents, grading, patch names and types,
//! the initial profiles — lives in [`ofgpu::blockgen`] in this port, so what
//! is left here really is only the command line.
//!
//! Provenance: ORIGINAL - the command-line front end to `src/blockgen.rs`. The
//! generator and the case format it writes are covered by that file's own
//! header; this one is argument parsing (`PROVENANCE.md`, `src/bin/*`). No
//! GPL-licensed source was consulted.

use std::path::Path;
use std::process::ExitCode;

use ofgpu::blockgen::{
    write_carved_case, write_carved_case_with_wall_model, write_case, write_case_cyclic,
    write_case_cyclic_with_wall_model, write_case_with_wall_model, write_cutcell_case,
    write_cutcell_case_with_wall_model, CaseKind,
};
use ofgpu::io::case::{Roughness, WallTreatment};
use ofgpu::io::contract;
use ofgpu::surface::cutcell::{DEFAULT_SUPERSAMPLE, DEFAULT_THETA_MIN};
use ofgpu::surface::{stl::read_stl, Surface};
use ofgpu::{Error, Result};

#[path = "common/mod.rs"]
mod common;

use common::atoi;

fn usage() {
    eprintln!(
        "usage: ofgpu-generate-mesh <channel|cavity|step|big|plume|room|damBreak> <outputDir> \
         [nx ny nz] [-stl [name=]path]... [-cutcell [-s N] [-thetaMin X]]\n       \
         [-wallModel standard|spalding|rough|lowRe [-Ks x [-Cs y]]] [-cyclic x|y|z] \
         [-permissive]\n       \
         ofgpu-generate-mesh big <outputDir> [n] [-stl ...]   # n^3 cells\n\
         {}",
        contract::PERMISSIVE_USAGE
    );
}

/// `-cyclic x|y|z` - SPEC-LIT §31.1. Only `translate` exists (the transform
/// implied by the block's own extent along the named axis), so there is
/// nothing to name beyond which axis; a §13.4 error lists the three that are.
fn parse_cyclic_axis(v: &str) -> Result<usize> {
    match v {
        "x" => Ok(0),
        "y" => Ok(1),
        "z" => Ok(2),
        _ => Err(Error::Config(format!(
            "-cyclic: \"{v}\" is not supported by ofgpu; available: x, y, z"
        ))),
    }
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
    let mut cutcell = false;
    let mut supersample = DEFAULT_SUPERSAMPLE;
    let mut theta_min = DEFAULT_THETA_MIN;
    // SPEC-LIT §29.1 route (c): `None` here means "the flag was never given",
    // which is what keeps `write_case`'s legacy `standard` row and adiabatic
    // `T` the exact default when `-wallModel` is absent - not merely
    // `Some(WallTreatment::Standard)`, which would also turn on §29.3's
    // thermal wall function nothing asked for.
    let mut wall_model: Option<WallTreatment> = None;
    let mut ks: Option<ofgpu::Scalar> = None;
    let mut cs: Option<ofgpu::Scalar> = None;
    // SPEC-LIT §31.1/§34.2: empty means "no cyclic pair" - the ordinary case
    // every other flag combination already produces. Repeatable so a plane
    // channel (two axes) or a fully periodic box (three) can be named.
    let mut cyclic: Vec<usize> = Vec::new();
    let mut i = 1usize;
    while i < args.len() {
        match args[i].as_str() {
            "-permissive" => contract::set_permissive(true),
            "-cutcell" => cutcell = true,
            "-cyclic" => {
                i += 1;
                let Some(v) = args.get(i) else {
                    usage();
                    return Err(Error::Config("-cyclic needs an axis: x, y or z".to_string()));
                };
                cyclic.push(parse_cyclic_axis(v)?);
            }
            "-wallModel" => {
                i += 1;
                let Some(v) = args.get(i) else {
                    usage();
                    return Err(Error::Config("-wallModel needs a preset name".to_string()));
                };
                wall_model = Some(WallTreatment::from_name(v)?);
            }
            "-Ks" => {
                i += 1;
                let Some(v) = args.get(i) else {
                    usage();
                    return Err(Error::Config("-Ks needs a sand-grain height".to_string()));
                };
                ks = Some(v.parse::<f64>().map_err(|_| {
                    Error::Config(format!("-Ks: '{v}' is not a number"))
                })? as ofgpu::Scalar);
            }
            "-Cs" => {
                i += 1;
                let Some(v) = args.get(i) else {
                    usage();
                    return Err(Error::Config("-Cs needs a roughness constant".to_string()));
                };
                cs = Some(v.parse::<f64>().map_err(|_| {
                    Error::Config(format!("-Cs: '{v}' is not a number"))
                })? as ofgpu::Scalar);
            }
            "-stl" => {
                i += 1;
                let Some(v) = args.get(i) else {
                    usage();
                    return Err(Error::Config("-stl needs a [name=]path argument".to_string()));
                };
                stl_args.push(v.clone());
            }
            "-s" => {
                i += 1;
                let Some(v) = args.get(i) else {
                    usage();
                    return Err(Error::Config("-s needs a supersample size".to_string()));
                };
                let n = atoi(v);
                if n < 1 {
                    return Err(Error::Config(format!("-s: bad supersample size '{v}'")));
                }
                supersample = n as usize;
            }
            "-thetaMin" => {
                i += 1;
                let Some(v) = args.get(i) else {
                    usage();
                    return Err(Error::Config("-thetaMin needs a threshold".to_string()));
                };
                theta_min = v.parse::<f64>().map_err(|_| {
                    Error::Config(format!("-thetaMin: '{v}' is not a number"))
                })? as ofgpu::Scalar;
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

    if cutcell && stl_args.is_empty() {
        usage();
        return Err(Error::Config("-cutcell needs at least one -stl surface".to_string()));
    }

    // SPEC-LIT §31.1: cyclic pairing is a plain-block feature - carving picks
    // its own wall patches out of the cut surface, and there is no coupled
    // pair left to declare once that has happened.
    if !cyclic.is_empty() && !stl_args.is_empty() {
        return Err(Error::Config(
            "-cyclic cannot be combined with -stl: carving replaces the block's own \
             patches with new wall patches, leaving no cyclic pair to declare"
                .to_string(),
        ));
    }
    // SPEC-LIT §29.1 route (c): `-Ks`/`-Cs` are resolved against whatever
    // preset `-wallModel` actually named - `rough` needs `Ks`, naming it;
    // every other preset (or no `-wallModel` at all) simply ignores them.
    let roughness = wall_model
        .map(|wt| Roughness::resolve(wt, ks, cs, "-wallModel"))
        .transpose()?
        .flatten();

    if stl_args.is_empty() {
        return match (wall_model, cyclic.is_empty()) {
            (None, true) => write_case(Path::new(dir), kind, nx, ny, nz),
            (Some(wt), true) => {
                write_case_with_wall_model(Path::new(dir), kind, nx, ny, nz, wt, roughness)
            }
            (None, false) => write_case_cyclic(Path::new(dir), kind, nx, ny, nz, &cyclic),
            (Some(wt), false) => write_case_cyclic_with_wall_model(
                Path::new(dir), kind, nx, ny, nz, &cyclic, wt, roughness,
            ),
        };
    }

    let surface = read_surfaces(&stl_args)?;
    println!(
        "[stl] merged: {} triangle(s), {} patch(es): {}",
        surface.tris.len(),
        surface.patch_names.len(),
        surface.patch_names.join(", ")
    );

    if cutcell {
        // ---- the cut-cell path (SPEC-LIT §24) -----------------------------
        let s = match wall_model {
            None => write_cutcell_case(
                Path::new(dir), kind, nx, ny, nz, &surface, supersample, theta_min,
            )?,
            Some(wt) => write_cutcell_case_with_wall_model(
                Path::new(dir), kind, nx, ny, nz, &surface, supersample, theta_min, wt, roughness,
            )?,
        };
        // `write_cutcell_case`/`_with_wall_model` already print their own
        // [cutcell] summary.
        let _ = s;
        return Ok(());
    }

    // ---- the carved path (SPEC-LIT §23) -------------------------------------
    let s = match wall_model {
        None => write_carved_case(Path::new(dir), kind, nx, ny, nz, &surface)?,
        Some(wt) => {
            write_carved_case_with_wall_model(Path::new(dir), kind, nx, ny, nz, &surface, wt, roughness)?
        }
    };

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
