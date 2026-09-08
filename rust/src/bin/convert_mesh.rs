// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! `ofgpu-convert-mesh` - a Gmsh mesh into a case directory's polyMesh.
//!
//! ```text
//! ofgpu-convert-mesh <in.msh> <outCaseDir> [-type <patchName>=<type>]...
//!                     [-fluent <out.msh>] [-fluentType <patchName>=<zone>]...
//! ```
//!
//! Reads `in.msh` (MSH 4.1 ASCII, `io/msh.rs`) and writes it as
//! `<outCaseDir>/constant/polyMesh` (`io/polymesh.rs`'s
//! `write_poly_mesh_raw`), then prints what it wrote. It writes NOTHING else
//! - no `0/`, no `system/` - because a mesh knows no physics: boundary
//! conditions, discretisation schemes and solver settings belong to a case,
//! and a converter that invented them would produce something that runs and
//! means nothing. Point `ofgpu-generate-mesh` at a fresh directory when a
//! complete case is what is wanted.
//!
//! Every patch comes out the bare `patch` `read_msh` wrote, because a Gmsh
//! physical surface carries a name and no type. The solvers, though, pick
//! wall treatment from the TYPE (`PatchKind::from_type`), so a ground
//! surface converted as-is would never see a wall function. Its type is
//! fixed up two ways, and nothing more:
//!
//! - by convention, applied by default: a name that starts with `wall` gets
//!   `type wall`, one that starts with `empty` gets `type empty`, one that
//!   starts with `symmetry` gets `type symmetry`, case-insensitive
//!   throughout; every other name keeps `patch`. The NAME is kept exactly as
//!   Gmsh wrote it - `wall_ground_land` stays `wall_ground_land`; only the
//!   type changes.
//! - `-type <patchName>=<type>` (repeatable) sets one patch's type outright
//!   and wins over the convention. A value that maps to nothing (and is not
//!   literally `patch`) is refused by name, accepted types listed: a typo
//!   such as `ground=wal` would otherwise write a boundary the solvers read
//!   as an ordinary patch, and no wall function would ever act on it.
//!
//! `-fluent <out.msh>` additionally writes the SAME fixed-up mesh as an
//! ANSYS Fluent ASCII mesh file (Fluent opens it with File > Read > Mesh),
//! in the layout Fluent reads cleanly: every index hexadecimal and 1-based,
//! internal faces in polyMesh node order with c0 = neighbour and c1 =
//! owner, boundary faces with the nodes REVERSED and c0 = owner (Fluent's
//! face normal, right-hand rule through the nodes, points into c0), one
//! face block per patch with zone ids from 10, and one cell element type
//! per cell (2 = tetrahedron). Tetrahedral meshes are all it covers: a mesh
//! with a non-triangular face, or a cell whose face count is not four, is
//! refused by name rather than written as something Fluent would quietly
//! misread.
//!
//! Each patch's Fluent zone type defaults from the polyMesh type just
//! applied - a wall stays `wall`, symmetry stays `symmetry` - then from the
//! names this project gives its inlets: `east`, a name starting with
//! `inlet`, one ending with `_source` (case-insensitive, like the `-type`
//! convention) become `velocity-inlet`, and everything else
//! `pressure-outlet`. Fluent re-zones after reading anyway, so a default
//! only has to be a sane starting point. `-fluentType <patchName>=<zone>`
//! (repeatable) sets one patch's zone outright and wins over the defaults;
//! a zone that is not one of `wall`, `velocity-inlet`, `pressure-inlet`,
//! `pressure-outlet`, `outflow`, `symmetry`, `interior` is refused by name
//! with the accepted list.
//!
//! A directory that already holds a polyMesh is refused rather than
//! overwritten: the files on disk may be the only copy of a mesh some
//! pre-processing chain produced, and nothing about the command line says
//! the operator expected them to be replaced.
//!
//! Provenance: ORIGINAL - the command-line front end to `io/msh.rs` and
//! `io/polymesh.rs`; the reader and the writer are covered by those files'
//! own headers. This one is argument parsing, the type convention and
//! overrides, the overwrite guard, the printed summary, and the Fluent mesh
//! writer - laid out to match this project's own reference writer
//! (`polymesh_to_fluent.py`), whose output Fluent reads cleanly; the Fluent
//! mesh FILE format is the specification here, not any program's source.
//! No GPL-licensed source was consulted.

use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::process::ExitCode;

use ofgpu::error::IoContext;
use ofgpu::io::msh::read_msh;
use ofgpu::io::polymesh::{PolyMeshRaw, write_poly_mesh_raw};
use ofgpu::mesh::PatchKind;
use ofgpu::{Error, Label, Result, Scalar};

/// Every `-type` value that means something - the strings
/// `PatchKind::from_type` maps to anything but `Generic`, plus `patch`
/// itself. Offered verbatim when a typo is refused.
const ACCEPTED_TYPES: [&str; 12] = [
    "patch",
    "wall",
    "mappedWall",
    "empty",
    "symmetry",
    "symmetryPlane",
    "wedge",
    "cyclic",
    "cyclicAMI",
    "cyclicSlip",
    "processor",
    "processorCyclic",
];

/// Which rule fixed a patch's type, printed per patch by the `[convert]`
/// summary so the operator can see why.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TypeRule {
    /// `-type <patchName>=<type>` named this patch.
    Override,
    /// The name's own prefix decided it.
    Convention,
    /// Neither - the `patch` `read_msh` wrote stands.
    Default,
}

impl TypeRule {
    fn as_str(self) -> &'static str {
        match self {
            Self::Override => "override",
            Self::Convention => "convention",
            Self::Default => "default",
        }
    }
}

fn usage() {
    eprintln!(
        "usage: ofgpu-convert-mesh <in.msh> <outCaseDir> [-type <patchName>=<type>]... \
         [-fluent <out.msh>] [-fluentType <patchName>=<zone>]..."
    );
}

/// One `-type <patchName>=<type>` argument, split on the first `=` and the
/// type checked on the spot: `PatchKind::from_type` maps everything it does
/// not know to `Generic`, and a typo such as `ground=wal` would otherwise
/// write a boundary the solvers read as an ordinary patch.
fn parse_type_override(arg: &str) -> Result<(String, String)> {
    let Some((name, t)) = arg.split_once('=') else {
        return Err(Error::Config(format!(
            "-type needs a <patchName>=<type> argument, got '{arg}'"
        )));
    };
    if t != "patch" && PatchKind::from_type(t) == PatchKind::Generic {
        return Err(Error::Config(format!(
            "-type: patch '{name}' asks for type '{t}', which ofgpu does not know; \
             accepted: {}",
            ACCEPTED_TYPES.join(", ")
        )));
    }
    Ok((name.to_string(), t.to_string()))
}

/// The type a patch NAME implies, `None` when it implies nothing.
/// Case-insensitive: `Wall_Ground` is the same wall `wall_ground` is. (A
/// `wall_` name starts with `wall`, so one prefix covers both.)
fn convention_type(name: &str) -> Option<&'static str> {
    let lower = name.to_lowercase();
    if lower.starts_with("wall") {
        Some("wall")
    } else if lower.starts_with("empty") {
        Some("empty")
    } else if lower.starts_with("symmetry") {
        Some("symmetry")
    } else {
        None
    }
}

/// The type one patch gets and the rule that produced it: an override naming
/// the patch wins, then the naming convention, then the `patch` `read_msh`
/// wrote stands.
fn resolve_patch_type(name: &str, overrides: &HashMap<String, String>) -> (String, TypeRule) {
    if let Some(t) = overrides.get(name) {
        return (t.clone(), TypeRule::Override);
    }
    match convention_type(name) {
        Some(t) => (t.to_string(), TypeRule::Convention),
        None => ("patch".to_string(), TypeRule::Default),
    }
}

/// Set `type_name` and `kind` on every patch - kept consistent exactly the
/// way `read_poly_mesh` derives them - and return one [`TypeRule`] per
/// patch, in patch order, for the summary.
fn apply_patch_types(raw: &mut PolyMeshRaw, overrides: &HashMap<String, String>) -> Vec<TypeRule> {
    raw.patches
        .iter_mut()
        .map(|p| {
            let (type_name, rule) = resolve_patch_type(&p.name, overrides);
            p.type_name = type_name;
            p.kind = PatchKind::from_type(&p.type_name);
            rule
        })
        .collect()
}

/// Every `-fluentType` value that means something - the zone types Fluent's
/// File > Read > Mesh accepts for a face zone. Offered verbatim when a typo
/// is refused.
const ACCEPTED_FLUENT_ZONES: [&str; 7] = [
    "wall",
    "velocity-inlet",
    "pressure-inlet",
    "pressure-outlet",
    "outflow",
    "symmetry",
    "interior",
];

/// Fluent's zone-type number for a `-fluentType` value - the number the
/// patch's `(13 ...)` face block header carries. `interior` is in the table
/// because Fluent accepts it, not because a boundary patch would normally
/// want to be one.
fn fluent_zone_number(zone: &str) -> Option<u32> {
    Some(match zone {
        "interior" => 2,
        "wall" => 3,
        "pressure-inlet" => 4,
        "pressure-outlet" => 5,
        "symmetry" => 7,
        "velocity-inlet" => 10,
        "outflow" => 36,
        _ => return None,
    })
}

/// One `-fluentType <patchName>=<zone>` argument, split on the first `=`
/// and the zone checked on the spot: a typo such as `top=nonsense` would
/// otherwise surface only when Fluent refuses the file.
fn parse_fluent_zone_override(arg: &str) -> Result<(String, String)> {
    let Some((name, zone)) = arg.split_once('=') else {
        return Err(Error::Config(format!(
            "-fluentType needs a <patchName>=<zone> argument, got '{arg}'"
        )));
    };
    if fluent_zone_number(zone).is_none() {
        return Err(Error::Config(format!(
            "-fluentType: patch '{name}' asks for Fluent zone '{zone}', which is \
             not one Fluent knows; accepted: {}",
            ACCEPTED_FLUENT_ZONES.join(", ")
        )));
    }
    Ok((name.to_string(), zone.to_string()))
}

/// A patch's Fluent zone: the name the `(39 ...)` record carries and the
/// zone-type number the `(13 ...)` face block header carries - resolved
/// once, so the two sections cannot drift apart.
struct FluentZone {
    zone: String,
    type_no: u32,
}

/// The Fluent zone a patch defaults to when `-fluentType` does not name it:
/// the polyMesh type first (a wall stays a wall, symmetry stays symmetry),
/// then the names this project gives its inlets - `east`, a name starting
/// with `inlet`, one ending with `_source` - then pressure-outlet for
/// everything else. Case-insensitive, like the `-type` convention above.
fn default_fluent_zone(name: &str, kind: PatchKind) -> &'static str {
    match kind {
        PatchKind::Wall => "wall",
        PatchKind::Symmetry => "symmetry",
        _ => {
            let lower = name.to_lowercase();
            if lower == "east" || lower.starts_with("inlet") || lower.ends_with("_source") {
                "velocity-inlet"
            } else {
                "pressure-outlet"
            }
        }
    }
}

/// One patch's zone: an override naming the patch wins over the default.
/// Cannot fail for anything `parse_fluent_zone_override` let through, but
/// the refusal stays a real error - the override map's contents are trusted
/// nowhere else.
fn fluent_zone_of(
    name: &str,
    kind: PatchKind,
    overrides: &HashMap<String, String>,
) -> Result<FluentZone> {
    let zone = match overrides.get(name) {
        Some(z) => z.as_str(),
        None => default_fluent_zone(name, kind),
    };
    match fluent_zone_number(zone) {
        Some(type_no) => Ok(FluentZone {
            zone: zone.to_string(),
            type_no,
        }),
        None => Err(Error::Config(format!(
            "-fluentType: patch '{name}' asks for Fluent zone '{zone}', which is \
             not one Fluent knows; accepted: {}",
            ACCEPTED_FLUENT_ZONES.join(", ")
        ))),
    }
}

/// One zone per patch, in patch order - the `(13 ...)` blocks and the
/// `(39 ...)` records are written in that same order, so the slice indexes
/// with the patch index.
fn resolve_fluent_zones(
    raw: &PolyMeshRaw,
    overrides: &HashMap<String, String>,
) -> Result<Vec<FluentZone>> {
    raw.patches
        .iter()
        .map(|p| fluent_zone_of(&p.name, p.kind, overrides))
        .collect()
}

fn run(args: &[String]) -> Result<()> {
    // ---- flags and positionals --------------------------------------------
    let mut positional: Vec<&String> = Vec::new();
    // A `-type` repeated for the same patch: the last one wins, the way a
    // repeated flag reads everywhere else.
    let mut overrides: HashMap<String, String> = HashMap::new();
    // `-fluent <out.msh>` and the `-fluentType` overrides that travel with
    // it, parsed with the same last-one-wins rule.
    let mut fluent_out: Option<&String> = None;
    let mut fluent_overrides: HashMap<String, String> = HashMap::new();
    let mut i = 1usize;
    while i < args.len() {
        match args[i].as_str() {
            "-type" => {
                i += 1;
                let Some(v) = args.get(i) else {
                    usage();
                    return Err(Error::Config(
                        "-type needs a <patchName>=<type> argument".to_string(),
                    ));
                };
                let (name, t) = parse_type_override(v)?;
                overrides.insert(name, t);
            }
            "-fluent" => {
                i += 1;
                let Some(v) = args.get(i) else {
                    usage();
                    return Err(Error::Config("-fluent needs an <out.msh> path".to_string()));
                };
                fluent_out = Some(v);
            }
            "-fluentType" => {
                i += 1;
                let Some(v) = args.get(i) else {
                    usage();
                    return Err(Error::Config(
                        "-fluentType needs a <patchName>=<zone> argument".to_string(),
                    ));
                };
                let (name, zone) = parse_fluent_zone_override(v)?;
                fluent_overrides.insert(name, zone);
            }
            _ => positional.push(&args[i]),
        }
        i += 1;
    }

    if positional.len() != 2 {
        usage();
        return Err(Error::Config(
            "convert_mesh: expected exactly <in.msh> <outCaseDir>".to_string(),
        ));
    }

    let case_dir = Path::new(positional[1].as_str());
    let dir = case_dir.join("constant").join("polyMesh");

    // `write_poly_mesh_raw` creates what is missing and truncates what is
    // not, so the guard has to come before it is called.
    if dir.exists() {
        return Err(Error::Config(format!(
            "convert_mesh: '{}' already holds a polyMesh; refusing to overwrite \
             it - convert into a fresh directory",
            dir.display()
        )));
    }

    let mut raw = read_msh(positional[0].as_str())?;
    // Types before the write, so the boundary file on disk carries what the
    // summary below prints.
    let rules = apply_patch_types(&mut raw, &overrides);
    write_poly_mesh_raw(&dir, &raw)?;

    // polyMesh never stores a cell count - derive it the way
    // `build_host_mesh` does.
    let n_cells = n_cells_of(&raw);

    println!(
        "[convert] {}: {} cells, {} points, {} faces ({} internal, {} boundary)",
        dir.display(),
        n_cells,
        raw.points.len(),
        raw.faces.len(),
        raw.neighbour.len(),
        raw.faces.len() - raw.neighbour.len()
    );
    // The Fluent mesh is written from the SAME fixed-up mesh, so the zone
    // defaults read the patch types just applied - a `-type ground=wall`
    // steers the Fluent zone the way it steers the solvers. It is written
    // after the polyMesh line above reports it, before the per-patch lines
    // both files share.
    if let Some(fluent_path) = fluent_out {
        let zones = resolve_fluent_zones(&raw, &fluent_overrides)?;
        write_fluent_mesh(Path::new(fluent_path), positional[0].as_str(), &raw, &zones)?;
        println!("[convert] fluent mesh: {fluent_path}");
    }

    for (p, rule) in raw.patches.iter().zip(rules) {
        println!(
            "[convert] patch '{}' ({}, {}): {} face(s)",
            p.name,
            p.type_name,
            rule.as_str(),
            p.size
        );
    }

    Ok(())
}

/// `max(owner, neighbour) + 1` - polyMesh never stores a cell count, and
/// `io/polymesh.rs`'s `n_cells_of` derives it the same way. Repeated here
/// because a binary cannot see that private copy.
fn n_cells_of(raw: &PolyMeshRaw) -> Label {
    raw.owner
        .iter()
        .chain(raw.neighbour.iter())
        .copied()
        .max()
        .unwrap_or(-1)
        + 1
}

/// Fluent zone ids for the patches start at 10: 1 is the fluid cell zone
/// and 2 the interior face zone every file declares.
const FLUENT_FIRST_PATCH_ZONE: usize = 10;

/// Cell element types per line - the reference writer wraps the `(12 ...)`
/// type list at 40 numbers a line, and Fluent is happy with exactly that.
const FLUENT_CELL_TYPES_PER_LINE: usize = 40;

/// Coordinates go out at 9 significant digits, C `%g` style - the same
/// decimal the reference writer's `%.9g` produces, so the two writers'
/// files are byte-identical apart from the leading `(0 ...)` comment. That
/// is display precision, not the polyMesh writer's 17: a Fluent mesh is
/// never read back by ofgpu.
const FLUENT_POINT_DIGITS: usize = 9;

/// The whole Fluent mesh file, built in memory and handed back - the way
/// `io/polymesh.rs` writes every polyMesh file - so the tests can hold the
/// layout byte for byte; [`write_fluent_mesh`] is the one who puts it on
/// disk. Everything the reader of such a file expects: declarations first,
/// then nodes, internal faces, one face block per patch, the per-cell
/// element types, then the zone names - every integer hexadecimal and
/// 1-based, coordinates decimal.
fn fluent_mesh_text(source: &str, raw: &PolyMeshRaw, zones: &[FluentZone]) -> Result<String> {
    check_fluent_mesh(raw)?;

    let n_if = raw.neighbour.len();
    let n_cells = n_cells_of(raw) as usize;
    let mut out = String::new();

    out.push_str(&format!(
        "(0 \"Fluent mesh written by ofgpu-convert-mesh from {source}\")\n"
    ));
    out.push_str("(0 \"Dimension:\")\n(2 3)\n(0 \"Grid dimensions:\")\n");
    out.push_str(&format!("(10 (0 1 {:x} 0 3))\n", raw.points.len()));
    out.push_str(&format!("(12 (0 1 {n_cells:x} 0 0))\n"));
    out.push_str(&format!("(13 (0 1 {:x} 0 0))\n", raw.faces.len()));

    // Nodes, one line each, in polyMesh point order.
    out.push_str(&format!("(10 (1 1 {:x} 1 3)(\n", raw.points.len()));
    for p in &raw.points {
        out.push_str(&format!(
            "{} {} {}\n",
            fmt_g_prec(p.x, FLUENT_POINT_DIGITS),
            fmt_g_prec(p.y, FLUENT_POINT_DIGITS),
            fmt_g_prec(p.z, FLUENT_POINT_DIGITS)
        ));
    }
    out.push_str("))\n");

    // Internal faces: nodes in polyMesh order, c0 = neighbour, c1 = owner -
    // which is what puts Fluent's face normal (right-hand rule through the
    // nodes) into c0.
    out.push_str(&format!("(13 (2 1 {n_if:x} 2 0)(\n"));
    for f in 0..n_if {
        let fv = &raw.faces[f];
        out.push_str(&format!(
            "3 {:x} {:x} {:x} {:x} {:x}\n",
            fv[0] + 1,
            fv[1] + 1,
            fv[2] + 1,
            raw.neighbour[f] + 1,
            raw.owner[f] + 1
        ));
    }
    out.push_str("))\n");

    // One face block per patch: the nodes REVERSED, c0 = owner, c1 = 0.
    // first/last are the patch's faces in the GLOBAL numbering - polyMesh
    // order, internal faces first - so startFace and startFace+nFaces, 1-based.
    for (pi, patch) in raw.patches.iter().enumerate() {
        out.push_str(&format!(
            "(13 ({:x} {:x} {:x} {:x} 0)(\n",
            FLUENT_FIRST_PATCH_ZONE + pi,
            n_if + patch.start + 1,
            n_if + patch.start + patch.size,
            zones[pi].type_no
        ));
        for k in 0..patch.size {
            let f = n_if + patch.start + k;
            let fv = &raw.faces[f];
            out.push_str(&format!(
                "3 {:x} {:x} {:x} {:x} 0\n",
                fv[2] + 1,
                fv[1] + 1,
                fv[0] + 1,
                raw.owner[f] + 1
            ));
        }
        out.push_str("))\n");
    }

    // The cell zone: one element type per cell, 2 = tetrahedron.
    out.push_str(&format!("(12 (1 1 {n_cells:x} 1 0)(\n"));
    let mut left = n_cells;
    while left > 0 {
        let take = left.min(FLUENT_CELL_TYPES_PER_LINE);
        out.push_str(&vec!["2"; take].join(" "));
        out.push('\n');
        left -= take;
    }
    out.push_str("))\n");

    // Zone names - decimal ids in the `(39 ...)` records.
    out.push_str("(39 (1 fluid fluid)())\n(39 (2 interior interior)())\n");
    for (pi, patch) in raw.patches.iter().enumerate() {
        out.push_str(&format!(
            "(39 ({} {} {})())\n",
            FLUENT_FIRST_PATCH_ZONE + pi,
            zones[pi].zone,
            patch.name
        ));
    }

    Ok(out)
}

/// `fluent_mesh_text` onto disk, named on failure like every other write in
/// the crate.
fn write_fluent_mesh(
    path: &Path,
    source: &str,
    raw: &PolyMeshRaw,
    zones: &[FluentZone],
) -> Result<()> {
    fs::write(path, fluent_mesh_text(source, raw, zones)?.as_bytes()).path(path)
}

/// The checks the Fluent writer runs before writing, all naming their
/// finding. The structural ones `build_host_mesh` would catch on the way to
/// the solver have no loader behind them here - unchecked, they surface as
/// an index panic mid-file instead of a message. The tet-only contract is
/// this writer's own: a quadrilateral face, or a cell whose face count is
/// not four, needs Fluent's mixed-element sections, which it does not write.
fn check_fluent_mesh(raw: &PolyMeshRaw) -> Result<()> {
    let n_faces = raw.faces.len();
    let n_if = raw.neighbour.len();

    if raw.owner.len() != n_faces {
        return Err(Error::Mesh(format!(
            "fluent mesh: owner has {} entries but there are {n_faces} faces",
            raw.owner.len()
        )));
    }
    if n_if > n_faces {
        return Err(Error::Mesh(format!(
            "fluent mesh: {n_if} internal faces out of {n_faces} total"
        )));
    }
    let n_bf = n_faces - n_if;
    for p in &raw.patches {
        if p.start + p.size > n_bf {
            return Err(Error::Mesh(format!(
                "fluent mesh: patch '{}' covers boundary faces {}..{} but there \
                 are only {n_bf}",
                p.name,
                p.start,
                p.start + p.size
            )));
        }
    }

    // Tetrahedral meshes only: every face a triangle over labels in range,
    // every cell exactly four faces (one from each of its own faces - a
    // face shared by two cells is counted for both, boundary faces once).
    let n_points = raw.points.len();
    for (f, fv) in raw.faces.iter().enumerate() {
        if fv.len() != 3 {
            return Err(Error::Mesh(format!(
                "fluent mesh: face {f} has {} nodes, not 3 - this writer covers \
                 tetrahedral meshes only (every face must be a triangle)",
                fv.len()
            )));
        }
        for &v in fv {
            if v < 0 || v as usize >= n_points {
                return Err(Error::Mesh(format!(
                    "fluent mesh: face {f} refers to point {v} but there are \
                     {n_points} points"
                )));
            }
        }
    }
    let mut faces_per_cell = vec![0u32; n_cells_of(raw) as usize];
    for (f, &o) in raw.owner.iter().enumerate() {
        if o < 0 {
            return Err(Error::Mesh(format!(
                "fluent mesh: face {f} has negative owner {o}"
            )));
        }
        faces_per_cell[o as usize] += 1;
    }
    for (f, &n) in raw.neighbour.iter().enumerate() {
        if n < 0 {
            return Err(Error::Mesh(format!(
                "fluent mesh: face {f} has negative neighbour {n}"
            )));
        }
        faces_per_cell[n as usize] += 1;
    }
    for (c, k) in faces_per_cell.iter().enumerate() {
        if *k != 4 {
            return Err(Error::Mesh(format!(
                "fluent mesh: cell {c} has {k} faces, not 4 - this writer covers \
                 tetrahedral meshes only (every cell must be a tetrahedron)"
            )));
        }
    }

    Ok(())
}

/// C's `%.*g` with `sig` significant digits - the same formatter
/// `io/polymesh.rs`'s private `fmt_g_prec` is, repeated here because a
/// binary cannot see it.
fn fmt_g_prec(v: Scalar, sig: usize) -> String {
    let x = v as f64;
    if x == 0.0 {
        // printf keeps the sign of a negative zero; so does this.
        return if x.is_sign_negative() { "-0".to_string() } else { "0".to_string() };
    }
    if !x.is_finite() {
        return format!("{x}");
    }

    // The decimal exponent comes from the formatter rather than from `log10`,
    // which is off by one at exact powers of ten on some libm builds.
    let sci = format!("{:.*e}", sig - 1, x);
    let (mant, exp) = match sci.split_once('e') {
        Some((m, e)) => (m, e.parse::<i32>().unwrap_or(0)),
        None => (sci.as_str(), 0),
    };

    // printf switches to the exponent style below 1e-4 and at or above 10^sig.
    if exp < -4 || exp >= sig as i32 {
        format!(
            "{}e{}{:02}",
            trim_trailing_zeros(mant),
            if exp < 0 { '-' } else { '+' },
            exp.abs()
        )
    } else {
        let dec = (sig as i32 - 1 - exp).max(0) as usize;
        trim_trailing_zeros(&format!("{:.*}", dec, x))
    }
}

fn trim_trailing_zeros(s: &str) -> String {
    if s.contains('.') {
        s.trim_end_matches('0').trim_end_matches('.').to_string()
    } else {
        s.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ofgpu::Vec3;
    use ofgpu::mesh::PatchInfo;

    /// The four patches every check here starts from, each the bare `patch`
    /// `read_msh` would have written.
    fn raw_with_patches(names: &[&str]) -> PolyMeshRaw {
        PolyMeshRaw {
            patches: names
                .iter()
                .map(|n| PatchInfo {
                    name: (*n).to_string(),
                    type_name: "patch".to_string(),
                    kind: PatchKind::Generic,
                    start: 0,
                    size: 1,
                    nbr_patch: None,
                })
                .collect(),
            ..PolyMeshRaw::default()
        }
    }

    #[test]
    fn convention_and_override_set_the_patch_type() {
        let mut raw = raw_with_patches(&["wall_ground", "top", "emptyFront", "pool_a"]);

        // No override: the name alone decides, and `pool_a` keeps `patch`.
        let rules = apply_patch_types(&mut raw, &HashMap::new());
        let types: Vec<&str> = raw.patches.iter().map(|p| p.type_name.as_str()).collect();
        assert_eq!(types, ["wall", "patch", "empty", "patch"]);
        let kinds: Vec<PatchKind> = raw.patches.iter().map(|p| p.kind).collect();
        assert_eq!(
            kinds,
            [
                PatchKind::Wall,
                PatchKind::Generic,
                PatchKind::Empty,
                PatchKind::Generic,
            ]
        );
        let rules: Vec<&str> = rules.iter().map(|r| r.as_str()).collect();
        assert_eq!(rules, ["convention", "default", "convention", "default"]);

        // `-type top=wall` wins over the name having no opinion, and leaves
        // every other patch exactly where the convention put it.
        let mut overrides = HashMap::new();
        overrides.insert("top".to_string(), "wall".to_string());
        let rules = apply_patch_types(&mut raw, &overrides);
        assert_eq!(raw.patches[1].type_name, "wall");
        assert_eq!(raw.patches[1].kind, PatchKind::Wall);
        assert_eq!(rules[1], TypeRule::Override);
        assert_eq!(raw.patches[0].type_name, "wall");
        assert_eq!(raw.patches[2].type_name, "empty");
    }

    // ---- the Fluent writer -------------------------------------------------

    /// One tetrahedron: 4 points, all 4 faces boundary on one patch, no
    /// internal faces. The patch carries the `wall` type the naming
    /// convention gives a name starting with `wall`, so the zone defaults
    /// are exercised at their first branch.
    fn one_tet() -> PolyMeshRaw {
        let mut raw = raw_with_patches(&["walls"]);
        raw.patches[0].size = 4;
        raw.patches[0].type_name = "wall".to_string();
        raw.patches[0].kind = PatchKind::Wall;
        raw.points = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
        ];
        // Outward from cell 0, the way `read_msh` winds them.
        raw.faces = vec![vec![0, 2, 1], vec![0, 1, 3], vec![0, 3, 2], vec![1, 2, 3]];
        raw.owner = vec![0; 4];
        raw
    }

    /// Two tets sharing face (0, 2, 1): one internal face owned by cell 0
    /// (the first cell to touch it), three boundary faces per patch on
    /// `east` and `top` - so the name-driven default (`east` ->
    /// velocity-inlet) and the fall-through (`top` -> pressure-outlet) both
    /// fire.
    fn two_tet() -> PolyMeshRaw {
        let mut raw = raw_with_patches(&["east", "top"]);
        raw.patches[0].size = 3;
        raw.patches[1].start = 3;
        raw.patches[1].size = 3;
        raw.points = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(0.0, 0.0, -1.0),
        ];
        raw.faces = vec![
            vec![0, 2, 1], // internal, owner 0 / neighbour 1
            vec![0, 1, 3], // east, owner 0
            vec![0, 3, 2],
            vec![1, 2, 3],
            vec![0, 1, 4], // top, owner 1
            vec![0, 4, 2],
            vec![1, 2, 4],
        ];
        raw.owner = vec![0, 0, 0, 0, 1, 1, 1];
        raw.neighbour = vec![1];
        raw
    }

    /// The whole file, byte for byte: the exact section headers, the four
    /// boundary lines with their nodes REVERSED and `c0 = 1, c1 = 0`, and
    /// the one-cell `(12 ...)` block holding a single `2`.
    #[test]
    fn a_single_tet_writes_the_exact_fluent_layout() -> Result<()> {
        let raw = one_tet();
        let text = fluent_mesh_text("tet.msh", &raw, &resolve_fluent_zones(&raw, &HashMap::new())?)?;

        assert_eq!(
            text,
            r#"(0 "Fluent mesh written by ofgpu-convert-mesh from tet.msh")
(0 "Dimension:")
(2 3)
(0 "Grid dimensions:")
(10 (0 1 4 0 3))
(12 (0 1 1 0 0))
(13 (0 1 4 0 0))
(10 (1 1 4 1 3)(
0 0 0
1 0 0
0 1 0
0 0 1
))
(13 (2 1 0 2 0)(
))
(13 (a 1 4 3 0)(
3 2 3 1 1 0
3 4 2 1 1 0
3 3 4 1 1 0
3 4 3 2 1 0
))
(12 (1 1 1 1 0)(
2
))
(39 (1 fluid fluid)())
(39 (2 interior interior)())
(39 (10 wall walls)())
"#
        );

        Ok(())
    }

    /// The internal face carries the polyMesh node order with `c0 =
    /// neighbour+1, c1 = owner+1`; the two patch blocks carry the right
    /// first/last hex indices in the global face numbering; and
    /// `-fluentType top=wall` moves the `(13 ...)` type number and the
    /// `(39 ...)` zone together, leaving `east`'s name-driven default alone.
    #[test]
    fn two_tets_share_a_face_and_fluent_type_overrides_rezone() -> Result<()> {
        let raw = two_tet();
        let text = fluent_mesh_text("two.msh", &raw, &resolve_fluent_zones(&raw, &HashMap::new())?)?;

        assert_eq!(
            text,
            r#"(0 "Fluent mesh written by ofgpu-convert-mesh from two.msh")
(0 "Dimension:")
(2 3)
(0 "Grid dimensions:")
(10 (0 1 5 0 3))
(12 (0 1 2 0 0))
(13 (0 1 7 0 0))
(10 (1 1 5 1 3)(
0 0 0
1 0 0
0 1 0
0 0 1
0 0 -1
))
(13 (2 1 1 2 0)(
3 1 3 2 2 1
))
(13 (a 2 4 a 0)(
3 4 2 1 1 0
3 3 4 1 1 0
3 4 3 2 1 0
))
(13 (b 5 7 5 0)(
3 5 2 1 2 0
3 3 5 1 2 0
3 5 3 2 2 0
))
(12 (1 1 2 1 0)(
2 2
))
(39 (1 fluid fluid)())
(39 (2 interior interior)())
(39 (10 velocity-inlet east)())
(39 (11 pressure-outlet top)())
"#
        );

        // `-fluentType top=wall`: the (13 ...) type number AND the (39 ...)
        // zone move together.
        let mut overrides = HashMap::new();
        overrides.insert("top".to_string(), "wall".to_string());
        let text = fluent_mesh_text("two.msh", &raw, &resolve_fluent_zones(&raw, &overrides)?)?;
        assert!(text.contains("(13 (b 5 7 3 0)("), "{text}");
        assert!(text.contains("(39 (11 wall top)())"), "{text}");
        assert!(!text.contains("pressure-outlet top"), "{text}");
        // `east` keeps its name-driven default.
        assert!(text.contains("(13 (a 2 4 a 0)("), "{text}");
        assert!(text.contains("(39 (10 velocity-inlet east)())"), "{text}");

        Ok(())
    }

    /// A `-fluentType` naming a zone Fluent does not know is refused at
    /// parse time, accepted zones listed - the same refusal a bad `-type`
    /// gets.
    #[test]
    fn fluent_type_override_with_an_unknown_zone_names_the_accepted_ones() {
        let msg = match parse_fluent_zone_override("top=nonsense") {
            Err(e) => e.to_string(),
            Ok(_) => panic!("an unknown Fluent zone must be refused"),
        };
        assert!(msg.contains("top"), "{msg}");
        assert!(msg.contains("nonsense"), "{msg}");
        for zone in ACCEPTED_FLUENT_ZONES {
            assert!(msg.contains(zone), "{msg}");
        }

        // No '=' at all is refused too, by the flag's own name.
        let msg = match parse_fluent_zone_override("top") {
            Err(e) => e.to_string(),
            Ok(_) => panic!("a -fluentType without '=' must be refused"),
        };
        assert!(msg.contains("-fluentType"), "{msg}");
    }

    /// The tet-only contract, both halves, each naming its finding: a
    /// quadrilateral face, and a cell with five triangular faces (a
    /// pyramid, not a tet).
    #[test]
    fn a_mesh_that_is_not_tetrahedral_is_refused_by_name() -> Result<()> {
        // A quadrilateral face: named by face index, before any cell is
        // even counted.
        let mut raw = one_tet();
        raw.faces[0] = vec![0, 1, 2, 3];
        let zones = resolve_fluent_zones(&raw, &HashMap::new())?;
        let msg = match fluent_mesh_text("quad.msh", &raw, &zones) {
            Err(e) => e.to_string(),
            Ok(_) => panic!("a quadrilateral face must be refused"),
        };
        assert!(msg.contains("face 0"), "{msg}");
        assert!(msg.contains("not 3"), "{msg}");
        assert!(msg.contains("tetrahedral"), "{msg}");

        // Triangular faces, but five of them on one cell.
        let mut raw = one_tet();
        raw.faces.push(vec![0, 1, 2]);
        raw.owner.push(0);
        raw.patches[0].size += 1;
        let zones = resolve_fluent_zones(&raw, &HashMap::new())?;
        let msg = match fluent_mesh_text("pyr.msh", &raw, &zones) {
            Err(e) => e.to_string(),
            Ok(_) => panic!("a five-faced cell must be refused"),
        };
        assert!(msg.contains("cell 0 has 5 faces"), "{msg}");
        assert!(msg.contains("not 4"), "{msg}");
        assert!(msg.contains("tetrahedral"), "{msg}");

        Ok(())
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
