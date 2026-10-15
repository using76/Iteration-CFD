// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! `ofgpu-generate-mesh` - ready-to-run test meshes for ofgpu.
//!
//! ```text
//! ofgpu-generate-mesh <case> <outputDir> [nx ny nz] [-stl [name=]path]... [-permissive]
//! ofgpu-generate-mesh big <outputDir> [n | nx ny nz] [-stl ...]  (one n: a cube of n per side; three: per axis, e.g. big dir 128 52 41)
//!
//! channel  plane channel, graded to both walls, 2-D (200 x 120 x 1)
//! cavity   lid-driven cavity, 2-D                   (128 x 128 x 1)
//! step     backward-facing-step BOX, 2-D            (300 x 100 x 1)
//! big      uniform benchmark box                    (160 x 160 x 160)
//! plume    buoyant plume, floor inlet, xMax outlet  (98 x 42 x 20)
//! damBreak two-phase collapsing column, 2-D           (150 x 90 x 1)
//! plate    laminar flat plate, symmetry floor ahead of the edge, 2-D (150 x 80 x 1)
//! ```
//!
//! `plate` (also `flatPlate`) is SPEC-LIT §144.1's flat plate: the leading
//! edge at `x = 0` on a node, a `symmetry` floor over `[-upstream, 0]`, the
//! `plate` wall over `[0, length]`, graded to the wall and to the edge from
//! both sides. `nx` is split a fifth ahead of the edge and the rest on the
//! plate. Its own flags, refused on every other case: `-upstream m`,
//! `-length m`, `-height m`, `-ratios rUp rPlate rY` (inlet cell / edge cell,
//! trailing-edge cell / edge cell, top cell / wall cell), `-uInf U` and
//! `-nu nu`. It prints the Blasius estimate of the wall cell's y+ (144.1c).
//! It takes no `-stl`, `-cutcell`, `-extent`, `-grading` or `-cyclic`.
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
//! `-extent xlo xhi ylo yhi zlo zhi` (metres) and `-grading x|y|z=r`
//! (repeatable, at most once per axis) replace a preset's own block extent
//! and per-axis cell growth, on the plain, castellated (`-stl`) and cut-cell
//! (`-stl -cutcell`) paths alike. `r` is the last cell divided by the first
//! along that axis, so `r > 1` puts the smallest cell at the axis's low end
//! (the ground when the axis is z). The six block patches keep the preset's
//! names and types - rename or retype them in `constant/polyMesh/boundary`
//! and `0/` afterwards. `plume`, `room` and `damBreak` refuse `-extent`:
//! their windows (and, for damBreak, the water column) are placed from the
//! preset's own extents, so an overridden box would leave them behind.
//!
//! `-wallModel <standard|spalding|rough|lowRe> [-Ks x [-Cs y]]` - SPEC-LIT
//! §29.1 route (c): expands the named preset into the `k`/`epsilon`/`omega`/
//! `nut` (and, for a case that solves `T`, the thermal wall function of
//! §29.3) types the generated `0/` directory writes, instead of the
//! hardcoded `standard` row `standard stays the default` when the flag is
//! absent. `rough` requires `-Ks`; `-Cs` defaults to `0.5`. Carved STL wall
//! patches follow the same preset as the block's own walls.
//!
//! `-inflow uniform` - step only, SPEC-LIT 110.3: the free stream U_ref uniform.
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
    write_carved_case_with_override, write_case_with_override, write_cutcell_case_with_override,
    write_flat_plate_case, BlockOverride, CaseKind, FlatPlateSpec,
};
use ofgpu::io::case::{Roughness, WallTreatment};
use ofgpu::io::contract;
use ofgpu::surface::cutcell::{DEFAULT_SUPERSAMPLE, DEFAULT_THETA_MIN};
use ofgpu::surface::{stl::read_stl, Surface};
use ofgpu::{Error, Result};

#[path = "common/mod.rs"]
mod common;

use common::atoi;

fn usage_text() -> String {
    format!(
        "usage: ofgpu-generate-mesh <channel|cavity|step|big|plume|room|damBreak|plate> <outputDir> \
         [nx ny nz] [-stl [name=]path]... [-cutcell [-s N] [-thetaMin X]]\n       \
         [-extent xlo xhi ylo yhi zlo zhi] [-grading x|y|z=r]... [-inflow uniform] \
         [-wallModel standard|spalding|rough|lowRe [-Ks x [-Cs y]]] [-cyclic x|y|z] \
         [-permissive]\n       \
         ofgpu-generate-mesh big <outputDir> [n | nx ny nz] [-stl ...]   # one n: an n^3 cube; three: per axis, e.g. big dir 128 52 41\n       \
         ofgpu-generate-mesh plate <outputDir> [nx ny 1] [-upstream m] [-length m] [-height m] \
         [-ratios rUp rPlate rY] [-uInf U] [-nu nu] [-wallModel ...]   # SPEC-LIT 144.1 flat plate\n       \
         resolution: none (the preset's own), three numbers nx ny nz, or - big only - one n; two are refused\n\
         {}",
        contract::PERMISSIVE_USAGE
    )
}

fn usage() {
    eprintln!("{}", usage_text());
}

/// The `plate` case's own flags (SPEC-LIT §144.1), each `None` until given.
/// Applied on top of [`FlatPlateSpec::with_resolution`]; whether the result
/// can be meshed is [`FlatPlateSpec::validate`]'s to refuse.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
struct PlateFlags {
    upstream: Option<f64>,
    length: Option<f64>,
    height: Option<f64>,
    ratios: Option<[f64; 3]>,
    u_inf: Option<f64>,
    nu: Option<f64>,
}

impl PlateFlags {
    /// The first plate flag given, for refusing it on another case by name.
    fn first_given(&self) -> Option<&'static str> {
        [
            (self.upstream.is_some(), "-upstream"),
            (self.length.is_some(), "-length"),
            (self.height.is_some(), "-height"),
            (self.ratios.is_some(), "-ratios"),
            (self.u_inf.is_some(), "-uInf"),
            (self.nu.is_some(), "-nu"),
        ]
        .into_iter()
        .find_map(|(given, name)| given.then_some(name))
    }

    fn apply(&self, s: &mut FlatPlateSpec) {
        if let Some(v) = self.upstream {
            s.x_upstream = v;
        }
        if let Some(v) = self.length {
            s.length = v;
        }
        if let Some(v) = self.height {
            s.height = v;
        }
        if let Some([a, b, c]) = self.ratios {
            (s.r_upstream, s.r_plate, s.r_y) = (a, b, c);
        }
        if let Some(v) = self.u_inf {
            s.u_inf = v;
        }
        if let Some(v) = self.nu {
            s.nu = v;
        }
    }
}

/// `count` numbers following the flag at `args[*i]`; the index rests on the
/// last one consumed.
fn parse_numbers(args: &[String], i: &mut usize, count: usize) -> Result<Vec<f64>> {
    let flag = args[*i].clone();
    let mut v = Vec::with_capacity(count);
    for _ in 0..count {
        *i += 1;
        let Some(a) = args.get(*i) else {
            usage();
            return Err(Error::Config(format!("{flag} needs {count} number(s)")));
        };
        v.push(
            a.parse::<f64>()
                .map_err(|_| Error::Config(format!("{flag}: '{a}' is not a number")))?,
        );
    }
    Ok(v)
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

/// `-extent xlo xhi ylo yhi zlo zhi` - the six numbers following the flag, in
/// metres, one pair per axis. Only the COUNT and the number-ness are checked
/// here; whether the values can be honoured (hi > lo, a preset that accepts
/// an extent at all) is [`BlockOverride::apply`]'s job, so the library
/// refuses exactly what the command line would.
fn parse_extent(args: &[String], i: &mut usize) -> Result<[ofgpu::Scalar; 6]> {
    let names = ["xlo", "xhi", "ylo", "yhi", "zlo", "zhi"];
    let mut v = [0.0 as ofgpu::Scalar; 6];
    for (k, slot) in v.iter_mut().enumerate() {
        *i += 1;
        let Some(a) = args.get(*i) else {
            usage();
            return Err(Error::Config(format!(
                "-extent needs six numbers xlo xhi ylo yhi zlo zhi (metres), got only {k}"
            )));
        };
        *slot = a
            .parse::<f64>()
            .map_err(|_| Error::Config(format!("-extent {}: '{a}' is not a number", names[k])))?
            as ofgpu::Scalar;
    }
    Ok(v)
}

/// `-grading x|y|z=r` - one axis and its one-sided expansion ratio (last
/// cell / first cell, so r > 1 puts the smallest cell at the axis's low end).
fn parse_grading(arg: &str) -> Result<(usize, ofgpu::Scalar)> {
    let Some((axis, ratio)) = arg.split_once('=') else {
        return Err(Error::Config(format!(
            "-grading: '{arg}' needs 'axis=ratio', e.g. z=6"
        )));
    };
    let axis = match axis {
        "x" => 0usize,
        "y" => 1,
        "z" => 2,
        a => {
            return Err(Error::Config(format!(
                "-grading: '{a}' is not an axis - x, y or z"
            )));
        }
    };
    let r = ratio
        .parse::<f64>()
        .map_err(|_| Error::Config(format!("-grading: '{ratio}' is not a number")))?
        as ofgpu::Scalar;
    if !(r > 0.0) {
        return Err(Error::Config(format!(
            "-grading {arg}: the ratio must be positive (it is last cell / first cell, \
             1 is uniform)"
        )));
    }
    Ok((axis, r))
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
    // `-extent`/`-grading`: `None`/[`None`; 3] mean "flag never given", which
    // keeps the preset exactly as `case_block_spec` built it. An override
    // holding neither would also be a no-op, but the CLI hands the library
    // whatever it parsed and lets `BlockOverride::apply` refuse or apply it -
    // so the library refuses exactly what the CLI would.
    let mut extent: Option<[ofgpu::Scalar; 6]> = None;
    let mut grading: [Option<ofgpu::Scalar>; 3] = [None; 3];
    let mut grading_seen = [false; 3];
    // SPEC-LIT 110.3: `-inflow uniform` replaces the step preset's own 1/7
    // power-law inflow with the free stream `U_ref` (10 m/s), inlet and
    // interior. Whether the preset can be given one at all is
    // `BlockOverride::apply`'s refusal - so the library refuses exactly what
    // the CLI would.
    let mut inflow_uniform = false;
    let mut plate = PlateFlags::default();
    let mut i = 1usize;
    while i < args.len() {
        match args[i].as_str() {
            "-permissive" => contract::set_permissive(true),
            "-upstream" => plate.upstream = Some(parse_numbers(args, &mut i, 1)?[0]),
            "-length" => plate.length = Some(parse_numbers(args, &mut i, 1)?[0]),
            "-height" => plate.height = Some(parse_numbers(args, &mut i, 1)?[0]),
            "-uInf" => plate.u_inf = Some(parse_numbers(args, &mut i, 1)?[0]),
            "-nu" => plate.nu = Some(parse_numbers(args, &mut i, 1)?[0]),
            "-ratios" => {
                let v = parse_numbers(args, &mut i, 3)?;
                plate.ratios = Some([v[0], v[1], v[2]]);
            }
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
            "-extent" => {
                extent = Some(parse_extent(args, &mut i)?);
            }
            "-grading" => {
                i += 1;
                let Some(v) = args.get(i) else {
                    usage();
                    return Err(Error::Config("-grading needs an axis=ratio argument".to_string()));
                };
                let (axis, ratio) = parse_grading(v)?;
                if grading_seen[axis] {
                    return Err(Error::Config(format!(
                        "-grading: axis {} given twice - at most one ratio per axis",
                        ["x", "y", "z"][axis]
                    )));
                }
                grading_seen[axis] = true;
                grading[axis] = Some(ratio);
            }
            "-inflow" => {
                i += 1;
                let Some(v) = args.get(i) else {
                    usage();
                    return Err(Error::Config(
                        "-inflow needs a profile word - the only one is uniform".to_string(),
                    ));
                };
                if v != "uniform" {
                    return Err(Error::Config(format!(
                        "-inflow: '{v}' is not a supported inflow profile - the only one is \
                         uniform"
                    )));
                }
                inflow_uniform = true;
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

    let res: Vec<&str> = positional[2..].iter().map(|s| s.as_str()).collect();
    let (nx, ny, nz) = resolve_dims(kind, &res)?;

    // SPEC-LIT §144.1: the plate is its own spec, written by its own writer.
    if kind == CaseKind::FlatPlate {
        let refused = [
            (!stl_args.is_empty(), "-stl"),
            (cutcell, "-cutcell"),
            (extent.is_some(), "-extent"),
            (grading_seen.iter().any(|&g| g), "-grading"),
            (!cyclic.is_empty(), "-cyclic"),
            // The plate never reaches `BlockOverride::apply` (it is its own
            // spec, refused before one could run), so `-inflow` is named here
            // like the other non-plate flags.
            (inflow_uniform, "-inflow"),
        ];
        if let Some((_, flag)) = refused.iter().find(|(given, _)| *given) {
            return Err(Error::Config(format!(
                "plate takes no {flag}: its block is two graded segments meeting at the \
                 leading edge, set by -upstream, -length, -height and -ratios"
            )));
        }
        let mut spec = FlatPlateSpec::with_resolution(nx, ny, nz)?;
        plate.apply(&mut spec);
        spec.validate()?;
        let roughness = wall_model
            .map(|wt| Roughness::resolve(wt, ks, cs, "-wallModel"))
            .transpose()?
            .flatten();
        write_flat_plate_case(Path::new(dir), &spec, wall_model, roughness)?;
        return Ok(());
    }
    if let Some(flag) = plate.first_given() {
        return Err(Error::Config(format!(
            "{flag} is a plate flag (SPEC-LIT 144.1); case '{}' takes none",
            kind.as_str()
        )));
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

    let over = BlockOverride { extent, grading, uniform_inflow: inflow_uniform };

    if stl_args.is_empty() {
        return write_case_with_override(
            Path::new(dir),
            kind,
            nx,
            ny,
            nz,
            Some(&over),
            &cyclic,
            wall_model,
            roughness,
        );
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
        let s = write_cutcell_case_with_override(
            Path::new(dir),
            kind,
            nx,
            ny,
            nz,
            &surface,
            Some(&over),
            supersample,
            theta_min,
            wall_model,
            roughness,
        )?;
        // `write_cutcell_case_with_override` already prints its own [cutcell]
        // summary, block extent included.
        let _ = s;
        return Ok(());
    }

    // ---- the carved path (SPEC-LIT §23) -------------------------------------
    let s = write_carved_case_with_override(
        Path::new(dir),
        kind,
        nx,
        ny,
        nz,
        &surface,
        Some(&over),
        wall_model,
        roughness,
    )?;

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

/// The cell counts the positional arguments after `<case> <outputDir>` name.
///
/// None: the preset's own resolution. Three: `nx ny nz`, one per axis. One,
/// for `big` only: a cube of `n` per side - the 2-D presets have an `empty`
/// front and back that a third dimension would make illegal. Any other count
/// is refused by name (SPEC-LIT §13.4): two numbers used to fall through to
/// the preset's default without a word.
fn resolve_dims(kind: CaseKind, res: &[&str]) -> Result<(usize, usize, usize)> {
    let (dx, dy, dz) = kind.default_resolution();
    match res {
        [] => Ok((dx, dy, dz)),
        [n] if kind == CaseKind::Big => {
            let n = clamp_dim(atoi(n))?;
            Ok((n, n, n))
        }
        [x, y, z] => Ok((clamp_dim(atoi(x))?, clamp_dim(atoi(y))?, clamp_dim(atoi(z))?)),
        _ => Err(Error::Config(format!(
            "generate_cases: {} takes no resolution (the preset's own {dx} x {dy} x {dz}) or three (nx ny nz){}; got {}: \"{}\"",
            kind.as_str(),
            if kind == CaseKind::Big { " or one n (an n^3 cube)" } else { "" },
            res.len(),
            res.join(" ")
        ))),
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    fn args(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn extent_reads_exactly_six_numbers() {
        let a = args(&["prog", "-extent", "-1240", "400", "-800", "500", "3.5", "200", "leftover"]);
        let mut i = 1usize;
        let e = parse_extent(&a, &mut i).expect("parse");
        assert_eq!(e, [-1240.0, 400.0, -800.0, 500.0, 3.5, 200.0]);
        assert_eq!(i, 7, "the index must rest on the last number consumed");
    }

    #[test]
    fn extent_refuses_fewer_than_six_numbers() {
        let a = args(&["prog", "-extent", "-1240", "400", "-800", "500", "3.5"]);
        let mut i = 1usize;
        assert!(parse_extent(&a, &mut i).is_err(), "five numbers must be refused");
    }

    #[test]
    fn extent_refuses_a_non_number() {
        let a = args(&["prog", "-extent", "-1240", "400", "-800", "500", "3.5", "wide"]);
        let mut i = 1usize;
        let e = parse_extent(&a, &mut i).expect_err("a non-number must be refused");
        assert!(e.to_string().contains("not a number"), "{e}");
    }

    #[test]
    fn grading_parses_axis_and_ratio() {
        for (s, axis, r) in [("x=2", 0usize, 2.0), ("y=1.5", 1, 1.5), ("z=6", 2, 6.0)] {
            let (a, got) = parse_grading(s).expect("parse");
            assert_eq!((a, got), (axis, r));
        }
    }

    #[test]
    fn grading_refuses_malformed_arguments() {
        for s in ["w=2", "=2", "z", "z=", "z=abc", "z=0", "z=-3", "x=1=2"] {
            assert!(parse_grading(s).is_err(), "'{s}' must be refused");
        }
    }

    #[test]
    fn the_usage_documents_the_per_axis_resolution_of_big() {
        let u = usage_text();
        for want in ["big <outputDir> [n | nx ny nz]", "big dir 128 52 41", "[nx ny nz]", "-permissive"] {
            assert!(u.contains(want), "usage must say {want:?}:\n{u}");
        }
    }

    #[test]
    fn big_takes_one_n_or_three_and_refuses_two_by_name() {
        assert_eq!(resolve_dims(CaseKind::Big, &["128", "52", "41"]).unwrap(), (128, 52, 41));
        assert_eq!(resolve_dims(CaseKind::Big, &["128"]).unwrap(), (128, 128, 128));
        assert_eq!(resolve_dims(CaseKind::Big, &[]).unwrap(), (160, 160, 160));
        let e = resolve_dims(CaseKind::Big, &["128", "52"]).unwrap_err().to_string();
        for want in ["big", "nx ny nz", "n^3", "\"128 52\""] {
            assert!(e.contains(want), "two numbers must be refused naming {want:?}: {e}");
        }
        let e = resolve_dims(CaseKind::Channel, &["200"]).unwrap_err().to_string();
        assert!(e.contains("channel") && !e.contains("n^3"), "one n is big-only: {e}");
        assert!(resolve_dims(CaseKind::Big, &["128", "52", "41", "7"]).is_err(), "four must be refused");
    }

    /// SPEC-LIT §144.1: `plate` names the flat plate, its flags land on the
    /// spec, and a plate flag on another case is refused by name.
    #[test]
    fn plate_flags_set_the_flat_plate_spec_and_are_refused_elsewhere() {
        assert_eq!(CaseKind::from_name("plate"), Some(CaseKind::FlatPlate));
        assert_eq!(resolve_dims(CaseKind::FlatPlate, &[]).unwrap(), (150, 80, 1));
        assert!(usage_text().contains("plate <outputDir>"));

        let a = args(&["prog", "-ratios", "10", "30", "150", "rest"]);
        let mut i = 1usize;
        assert_eq!(parse_numbers(&a, &mut i, 3).unwrap(), vec![10.0, 30.0, 150.0]);
        assert_eq!(i, 4, "the index must rest on the last number consumed");
        let a = args(&["prog", "-length", "long"]);
        let mut i = 1usize;
        assert!(parse_numbers(&a, &mut i, 1).unwrap_err().to_string().contains("not a number"));

        let f = PlateFlags { length: Some(2.0), ratios: Some([10.0, 30.0, 150.0]), ..Default::default() };
        assert_eq!(f.first_given(), Some("-length"));
        assert_eq!(PlateFlags::default().first_given(), None);
        let mut s = FlatPlateSpec::with_resolution(150, 80, 1).unwrap();
        f.apply(&mut s);
        assert_eq!((s.length, s.r_upstream, s.r_plate, s.r_y), (2.0, 10.0, 30.0, 150.0));
        assert_eq!((s.n_upstream, s.n_plate, s.n_y), (30, 120, 80));

        let e = run(&args(&["prog", "channel", "unused_dir", "-length", "2"])).unwrap_err();
        assert!(e.to_string().contains("-length is a plate flag"), "{e}");
        let e = run(&args(&["prog", "plate", "unused_dir", "-cyclic", "x"])).unwrap_err();
        assert!(e.to_string().contains("plate takes no -cyclic"), "{e}");
    }

    #[test]
    fn a_zero_resolution_component_is_still_refused() {
        let e = resolve_dims(CaseKind::Big, &["0", "5", "5"]).unwrap_err().to_string();
        assert!(e.contains("bad resolution component 0"), "{e}");
    }

    /// SPEC-LIT 110.3: `-inflow` needs its one profile word, only `uniform`
    /// exists, and the plate - whose own spec never reaches
    /// `BlockOverride::apply` - refuses it by name like its other flags.
    #[test]
    fn inflow_takes_exactly_the_word_uniform_and_is_refused_on_the_plate() {
        let e = run(&args(&["prog", "step", "unused_dir", "-inflow"])).unwrap_err();
        assert!(e.to_string().contains("-inflow"), "no word must name -inflow: {e}");
        let e = run(&args(&["prog", "step", "unused_dir", "-inflow", "powerLaw"])).unwrap_err();
        assert!(
            e.to_string().contains("-inflow"),
            "'powerLaw' must be refused naming -inflow: {e}"
        );
        let e = run(&args(&["prog", "plate", "unused_dir", "-inflow", "uniform"])).unwrap_err();
        assert!(
            e.to_string().contains("plate takes no -inflow"),
            "the plate must refuse -inflow by name: {e}"
        );
        assert!(usage_text().contains("[-inflow uniform]"), "the usage documents the flag");
    }
}
