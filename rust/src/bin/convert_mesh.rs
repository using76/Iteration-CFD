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
//!                     [-keepRegions]
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
//! in the layout of the **ANSYS FLUENT 12.0 User's Guide, Appendix B "Mesh
//! File Format", B.3.7 "Faces"**
//! (<https://www.afs.enea.it/project/neptunius/docs/fluent/html/ug/node1471.htm>),
//! whose orientation rule reads: *"if you curl the fingers of your right
//! hand in the order of the nodes, your thumb will point toward c1."* Read
//! literally, that is the polyMesh order (a face's own node order points
//! from the owner to the neighbour), and the first version of this writer
//! wrote it untouched. Fluent 2022 R2's own mesh check on a 7.8 M-cell site
//! mesh written that way reported every face left-handed and every cell
//! with a negative volume; the file with every face's node order REVERSED
//! - the convention OpenFOAM's foamMeshToFluent uses, thumb toward c0 and
//! boundary normals pointing into the cell - is what Fluent accepts. So the
//! nodes go out REVERSED on every face, c0 = owner and c1 = neighbour on an
//! internal face, c1 = 0 on a boundary one ("if a face has a cell only on
//! one side, then either c0 or c1 is zero"). Every index hexadecimal and
//! 1-based, one face block per patch with zone ids from 10, and one cell
//! element type per cell (2 = tetrahedron). Tetrahedral meshes are all it
//! covers: a mesh with a non-triangular face, or a cell whose face count is
//! not four, is refused by name rather than written as something Fluent
//! would quietly misread.
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
//! # Sealed cell regions
//!
//! A tetrahedral generator can seal pockets of cells under buildings: whole
//! clusters whose every face is internal to the cluster or a boundary face of
//! the surrounding wall. Nothing joins a pocket to the rest of the mesh, so
//! with `p` zeroGradient on that wall the pocket's block of the pressure
//! matrix is singular - the pressure there drifts to 1e16, poisons the mean
//! the linear solver normalises residuals over, and from then on the run
//! marches against a dead pressure equation reporting ZERO iterations. The
//! converter counts the cell regions (`mesh::geometry::cell_regions`) and, by
//! default, keeps the largest and drops the rest - their cells, their internal
//! faces and their boundary faces, the kept cells renumbered and the patch
//! order and the upper-triangular face order kept - and says so:
//!
//! ```text
//! regions: 9; dropped 8 sealed region(s), 24 cell(s), 77 face(s) (19 internal, 58 on wall_ground_land)
//! kept region: 343466 cells
//! ```
//!
//! The drop is refused once it stops being pocket-sized: more than 1 % of the
//! cells, or a largest region under ten times the second, is not a pocket but
//! a second domain, and that is the operator's decision, not a default's.
//! `-keepRegions` is the way to record it: keep every region, convert the mesh
//! as it stands, print only the `regions:` line. The Fluent writer sees the
//! SAME mesh either way, because both writers share the one in-memory
//! polyMesh. A converted mesh that still carries more than one region says so
//! at load - the solver's own `regions:` line is the thing to read when a
//! pressure solve reports zero iterations.
//!
//! Provenance: ORIGINAL - the command-line front end to `io/msh.rs` and
//! `io/polymesh.rs`; the reader and the writer are covered by those files'
//! own headers. This one is argument parsing, the type convention and
//! overrides, the overwrite guard, the printed summary, and the Fluent mesh
//! writer - an implementation of a documented file format: the **ANSYS
//! FLUENT 12.0 User's Guide, Appendix B "Mesh File Format", B.3.7 "Faces"**
//! (`afs.enea.it/project/neptunius/docs/fluent/html/ug/node1471.htm`) is the
//! specification here, not any program's source.
//! No GPL-licensed source was consulted.

use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::process::ExitCode;

use ofgpu::error::IoContext;
use ofgpu::io::msh::read_msh;
use ofgpu::io::polymesh::{PolyMeshRaw, build_host_mesh, write_poly_mesh_raw};
use ofgpu::mesh::PatchKind;
use ofgpu::mesh::geometry::{cell_regions, region_sizes_text};
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
         [-fluent <out.msh>] [-fluentType <patchName>=<zone>]... [-keepRegions]"
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
    // `-keepRegions`: count the cell regions, keep every one, decide nothing.
    let mut keep_regions = false;
    let mut i = 1usize;
    while i < args.len() {
        match args[i].as_str() {
            "-keepRegions" => keep_regions = true,
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
    // The region pass before either write: the polyMesh and the Fluent mesh
    // must be the same kept mesh, and a refusal here must write nothing.
    handle_cell_regions(&mut raw, keep_regions)?;
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

// ==========================================================================
//  Sealed cell regions
// ==========================================================================

/// A drop stops being pocket-sized - and becomes the operator's decision -
/// above either of these. 1 % of the cells is a generous ceiling for what a
/// mesher leaves under buildings (the ammonia site's eight pockets were
/// 0.007 %); a largest region under ten times the second means the mesh is
/// two domains of comparable size, which no default should collapse.
const DROP_MAX_CELL_FRACTION_PCT: usize = 1;
const DROP_MIN_REGION_RATIO: usize = 10;

/// Count the mesh's cell regions and, unless `keep_regions`, drop every one
/// but the largest - the pockets a tetrahedral generator seals under
/// buildings, whose every face is internal to the pocket or a boundary face
/// of the wall around it. Must run before either write, so that the polyMesh
/// and the Fluent mesh are the SAME kept mesh and a refusal leaves the case
/// directory untouched.
///
/// A mesh of one region passes through untouched. A mesh of several is either
/// repaired (the default), refused when the drop stops being pocket-sized
/// (a second domain is the operator's call), or kept whole under
/// `-keepRegions` - which then only prints the `regions:` line.
fn handle_cell_regions(raw: &mut PolyMeshRaw, keep_regions: bool) -> Result<()> {
    // `build_host_mesh` fills the addressing `cell_regions` walks, and as a
    // side effect validates it: owner < neighbour, patches in range. A mesh
    // it refuses would have been refused by the loader minutes later, with
    // the files already on disk.
    let host = build_host_mesh(raw)?;
    let (n_regions, region_of) = cell_regions(&host);
    let n_cells = n_cells_of(raw) as usize;

    let mut sizes = vec![0usize; n_regions];
    for &r in &region_of {
        sizes[r as usize] += 1;
    }
    // Region labels ordered by size, largest first; ties keep label order,
    // so "the largest region" is deterministic on a mesh cut in equal halves.
    let mut by_size: Vec<u32> = (0..n_regions as u32).collect();
    by_size.sort_unstable_by(|a, b| sizes[*b as usize].cmp(&sizes[*a as usize]));

    if n_regions <= 1 {
        if keep_regions {
            println!("regions: {n_regions}");
        }
        return Ok(());
    }
    if keep_regions {
        // Largest first, the way the loader's own `regions:` line reads.
        let mut desc = sizes.clone();
        desc.sort_unstable_by(|a, b| b.cmp(a));
        println!("regions: {n_regions} {}", region_sizes_text(&desc));
        return Ok(());
    }

    let largest = by_size[0] as usize;
    let second = sizes[by_size[1] as usize];
    let dropped_cells = n_cells - sizes[largest];

    if dropped_cells * 100 > n_cells * DROP_MAX_CELL_FRACTION_PCT
        || sizes[largest] < DROP_MIN_REGION_RATIO * second
    {
        return Err(Error::Config(format!(
            "convert_mesh: the mesh holds {n_regions} cell regions and keeping \
             only the largest would drop {dropped_cells} of {n_cells} cells \
             (the largest is {:.1}x the second); that is not a sealed pocket \
             but a second domain - pass -keepRegions to convert the mesh as \
             it stands, or split it into one mesh per domain first",
            sizes[largest] as f64 / second as f64
        )));
    }

    let (dropped_internal, dropped_boundary) = keep_region(raw, &region_of, largest as u32);
    let dropped_faces = dropped_internal + dropped_boundary.iter().map(|(_, k)| k).sum::<usize>();

    println!(
        "{}",
        dropped_regions_message(
            n_regions,
            dropped_cells,
            dropped_faces,
            dropped_internal,
            &dropped_boundary
        )
    );
    println!("kept region: {} cells", sizes[largest]);

    Ok(())
}

/// The `regions:` line a drop prints - the region count, then what went away
/// in cells and faces, the faces split into internal and per patch, so the
/// operator can see at a glance which wall the pockets sat under.
fn dropped_regions_message(
    n_regions: usize,
    dropped_cells: usize,
    dropped_faces: usize,
    dropped_internal: usize,
    dropped_boundary: &[(String, usize)],
) -> String {
    let mut where_: Vec<String> = vec![format!("{dropped_internal} internal")];
    where_.extend(
        dropped_boundary
            .iter()
            .map(|(name, k)| format!("{k} on {name}")),
    );

    format!(
        "regions: {n_regions}; dropped {} sealed region(s), {dropped_cells} \
         cell(s), {dropped_faces} face(s) ({})",
        n_regions - 1,
        where_.join(", ")
    )
}

/// Keep one region's cells and their faces; drop the rest, in place.
///
/// Kept cells renumber to `0..n_kept` in their old order, which is what keeps
/// every invariant the writers and the solver assume: a filtered subsequence
/// of faces sorted by (owner, neighbour) is still sorted, and an
/// order-preserving renumbering preserves every comparison - so the
/// upper-triangular face order survives untouched. The kept internal faces
/// stay a prefix and the kept boundary faces stay patch-contiguous in patch
/// order, so the patch table needs only its `start`/`size` recounted; a patch
/// whose every face sat on dropped pockets stays, empty, in its place. Points
/// are left alone - an unused point is harmless, and renumbering them would
/// rewrite every face for nothing.
///
/// Because a region is a connected component, no face of a kept cell touches
/// a dropped one: a kept cell's every face is kept, so nothing about the kept
/// cells changes but their numbering.
///
/// Returns the dropped face counts: internal faces, then boundary faces per
/// patch in patch order, patches with nothing dropped left out.
fn keep_region(
    raw: &mut PolyMeshRaw,
    region_of: &[u32],
    keep: u32,
) -> (usize, Vec<(String, usize)>) {
    let n_if = raw.neighbour.len();
    let n_faces = raw.faces.len();
    let in_region = |c: Label| region_of[c as usize] == keep;

    // Per-patch survival, counted while the addressing still names the old
    // cells, and the starts the kept faces will sit at.
    let mut dropped_per_patch = vec![0usize; raw.patches.len()];
    let mut kept_start = vec![0usize; raw.patches.len()];
    let mut start = 0usize;
    for (p, pi) in raw.patches.iter().enumerate() {
        kept_start[p] = start;
        for bf in pi.start..pi.start + pi.size {
            if !in_region(raw.owner[n_if + bf]) {
                dropped_per_patch[p] += 1;
            }
        }
        start += pi.size - dropped_per_patch[p];
    }

    // Internal faces first, compacted in place: the write cursor never
    // passes the read cursor, so each swap only shifts an already-kept face
    // out of the way, to be picked up again when the cursor reaches it.
    let mut w = 0usize;
    for f in 0..n_if {
        if in_region(raw.owner[f]) {
            raw.faces.swap(w, f);
            raw.owner.swap(w, f);
            raw.neighbour.swap(w, f);
            w += 1;
        }
    }
    let n_if_kept = w;
    let dropped_internal = n_if - n_if_kept;

    // Then the boundary faces, down to just behind them.
    for bf in n_if..n_faces {
        if in_region(raw.owner[bf]) {
            raw.faces.swap(w, bf);
            raw.owner.swap(w, bf);
            w += 1;
        }
    }

    raw.faces.truncate(w);
    raw.owner.truncate(w);
    raw.neighbour.truncate(n_if_kept);

    // The kept cells, renumbered in their old order.
    let mut new_cell = vec![-1 as Label; region_of.len()];
    let mut next: Label = 0;
    for (c, &r) in region_of.iter().enumerate() {
        if r == keep {
            new_cell[c] = next;
            next += 1;
        }
    }
    for o in raw.owner.iter_mut() {
        *o = new_cell[*o as usize];
    }
    for n in raw.neighbour.iter_mut() {
        *n = new_cell[*n as usize];
    }

    for (p, pi) in raw.patches.iter_mut().enumerate() {
        pi.start = kept_start[p];
        pi.size -= dropped_per_patch[p];
    }

    let dropped_boundary: Vec<(String, usize)> = raw
        .patches
        .iter()
        .zip(&dropped_per_patch)
        .filter(|(_, d)| **d > 0)
        .map(|(pi, d)| (pi.name.clone(), *d))
        .collect();

    (dropped_internal, dropped_boundary)
}

/// Fluent zone ids for the patches start at 10: 1 is the fluid cell zone
/// and 2 the interior face zone every file declares.
const FLUENT_FIRST_PATCH_ZONE: usize = 10;

/// Cell element types per line: the `(12 ...)` type list is wrapped at 40
/// numbers a line, a width Fluent reads without complaint.
const FLUENT_CELL_TYPES_PER_LINE: usize = 40;

/// Coordinates go out at 9 significant digits, C `%g` style (the shortest
/// form that survives a single-precision round trip in Fluent). That is
/// display precision, not the polyMesh writer's 17: a Fluent mesh is never
/// read back by ofgpu.
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

    // Internal faces: nodes in REVERSED polyMesh order, c0 = owner and c1 =
    // neighbour. A polyMesh face's own order points from the owner to the
    // neighbour; Fluent's mesh check calls a face written that way
    // left-handed (see the module comment), so the right-hand normal Fluent
    // wants points from the neighbour to the owner - toward c0.
    out.push_str(&format!("(13 (2 1 {n_if:x} 2 0)(\n"));
    for f in 0..n_if {
        let fv = &raw.faces[f];
        out.push_str(&format!(
            "3 {:x} {:x} {:x} {:x} {:x}\n",
            fv[2] + 1,
            fv[1] + 1,
            fv[0] + 1,
            raw.owner[f] + 1,
            raw.neighbour[f] + 1
        ));
    }
    out.push_str("))\n");

    // One face block per patch: the same REVERSED node order, c0 = owner,
    // c1 = 0 - "if a face has a cell only on one side, then either c0 or c1
    // is zero"; the stored polyMesh order points out of the domain, so the
    // reversed one points into the cell, as foamMeshToFluent writes it.
    // first/last are the patch's faces in the GLOBAL numbering - polyMesh
    // order, internal faces first - so startFace and startFace+nFaces,
    // 1-based.
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
    use ofgpu::io::polymesh::read_poly_mesh;
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
    /// fire. Every face is wound outward from its owner - toward the
    /// neighbour on the internal one - the way `read_msh` winds them, which
    /// is the invariant the ANSYS orientation rule leans on.
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
            vec![0, 4, 1], // top, owner 1
            vec![0, 2, 4],
            vec![1, 4, 2],
        ];
        raw.owner = vec![0, 0, 0, 0, 1, 1, 1];
        raw.neighbour = vec![1];
        raw
    }

    /// The whole file, byte for byte: the exact section headers, the four
    /// boundary lines in the polyMesh node order with `c0 = 1, c1 = 0`, and
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

    /// The internal face carries the REVERSED polyMesh node order with `c0 =
    /// owner+1, c1 = neighbour+1`; the two patch blocks carry the right
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
3 2 3 1 1 2
))
(13 (a 2 4 a 0)(
3 4 2 1 1 0
3 3 4 1 1 0
3 4 3 2 1 0
))
(13 (b 5 7 5 0)(
3 2 5 1 2 0
3 5 3 1 2 0
3 3 5 2 2 0
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

    /// The orientation Fluent accepts, checked against the GEOMETRY rather
    /// than against a transcript: each face line the file carries, its
    /// nodes taken in the written order right-handed, must produce a normal
    /// that points toward the cell the line names c0 - out of the neighbour
    /// into the owner on an internal face, and into the owner (into the
    /// domain) on a boundary one whose c1 is 0. This is the inverse of
    /// B.3.7's sentence and what foamMeshToFluent writes; Fluent's own mesh
    /// check called the literal reading left-handed on every face.
    #[test]
    fn every_written_faces_right_hand_normal_points_toward_c0() -> Result<()> {
        let raw = two_tet();
        let text = fluent_mesh_text("geo.msh", &raw, &resolve_fluent_zones(&raw, &HashMap::new())?)?;

        // Each cell's centroid, from the points its own faces touch - on a
        // tet that is all four vertices, so the mean of them is the
        // centroid.
        let n_cells = n_cells_of(&raw) as usize;
        let mut verts: Vec<Vec<usize>> = vec![Vec::new(); n_cells];
        for (f, fv) in raw.faces.iter().enumerate() {
            let cells = std::iter::once(raw.owner[f]).chain(raw.neighbour.get(f).copied());
            for c in cells {
                for &v in fv {
                    let c = c as usize;
                    if !verts[c].contains(&(v as usize)) {
                        verts[c].push(v as usize);
                    }
                }
            }
        }
        let centroid = |c: usize| -> Vec3 {
            verts[c]
                .iter()
                .fold(Vec3::new(0.0, 0.0, 0.0), |a, &v| a + raw.points[v])
                / verts[c].len() as Scalar
        };

        // Every face line, from whichever `(13 ...)` block wrote it:
        // `3 n0 n1 n2 c0 c1`, hexadecimal, 1-based.
        let mut lines = Vec::new();
        for line in text.lines() {
            let t: Vec<&str> = line.split_whitespace().collect();
            if t.len() != 6 || t[0] != "3" {
                continue;
            }
            let mut v = [0i64; 5];
            for (k, s) in t[1..].iter().enumerate() {
                v[k] = match i64::from_str_radix(s, 16) {
                    Ok(x) => x,
                    Err(_) => panic!("a face line must be hexadecimal: '{line}'"),
                };
            }
            lines.push(v);
        }
        assert_eq!(
            lines.len(),
            raw.faces.len(),
            "every face must be written exactly once"
        );

        let point = |n: i64| raw.points[(n - 1) as usize];
        for &[n0, n1, n2, c0, c1] in &lines {
            let (a, b, c) = (point(n0), point(n1), point(n2));
            // The right-hand rule through the WRITTEN node order.
            let normal = (b - a).cross(c - a);
            let face_centre = (a + b + c) / 3.0;
            if c1 == 0 {
                // Boundary: toward c0 - into the owner, into the domain.
                let outward = face_centre - centroid(c0 as usize - 1);
                assert!(
                    normal.dot(outward) < 0.0,
                    "boundary face {n0} {n1} {n2} (c0 {c0}): its right-hand normal \
                     points out of the domain, not into the owner"
                );
            } else {
                // Internal: toward the cell the line names c0 - the owner.
                let to_c1 = centroid(c1 as usize - 1) - centroid(c0 as usize - 1);
                assert!(
                    normal.dot(to_c1) < 0.0,
                    "internal face {n0} {n1} {n2} (c0 {c0}, c1 {c1}): its right-hand \
                     normal does not point toward c0"
                );
            }
        }

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

    // ---- the sealed cell regions ------------------------------------------

    /// A scratch directory for the round trips, the way `io/polymesh.rs`'s
    /// tests make theirs.
    fn scratch(tag: &str) -> std::path::PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let mut d = std::env::temp_dir();
        d.push(format!(
            "ofgpu-convert-mesh-{tag}-{}-{nanos}",
            std::process::id()
        ));
        fs::create_dir_all(&d).unwrap();
        d
    }

    /// `two_tet` plus a third tet 10 m to +x whose four faces are ALL
    /// boundary on a patch of their own: a sealed pocket beside the two-cell
    /// region the mesh is really made of - 11 faces, 1 of them internal,
    /// 3 cells in 2 regions.
    fn two_tet_with_a_sealed_third() -> PolyMeshRaw {
        let mut raw = two_tet();
        raw.patches.push(PatchInfo {
            name: "walls".to_string(),
            type_name: "wall".to_string(),
            kind: PatchKind::Wall,
            start: 6,
            size: 4,
            nbr_patch: None,
        });
        raw.points.extend([
            Vec3::new(10.0, 0.0, 0.0),
            Vec3::new(11.0, 0.0, 0.0),
            Vec3::new(10.0, 1.0, 0.0),
            Vec3::new(10.0, 0.0, 1.0),
        ]);
        // Points 5..8, wound outward from cell 2 - `one_tet`'s face list
        // shifted by the five points the two-cell region already holds - the
        // way `read_msh` winds them.
        raw.faces.extend([
            vec![5, 7, 6],
            vec![5, 6, 8],
            vec![5, 8, 7],
            vec![6, 7, 8],
        ]);
        raw.owner.extend([2; 4]);
        raw
    }

    /// An `nx x ny x 1` slab of hexahedra with cell `(px, py)` sealed off:
    /// every internal face touching it moved to boundary on a patch of its
    /// own, owned by the pocket cell - which is exactly how a sealed pocket
    /// looks in a polyMesh, the surrounding mesh closed by the pocket's own
    /// faces. Polygonal faces are fine: the drop works on the polyMesh, not
    /// on the tet-only Fluent writer.
    ///
    /// Cells `nx*ny`, internal faces `2 nx ny - nx - ny`, boundary faces
    /// `2 nx ny + 2 nx + 2 ny`, plus the pocket's reclassified faces.
    fn hex_slab_with_pocket(nx: usize, ny: usize, px: usize, py: usize) -> PolyMeshRaw {
        let pt = |i: usize, j: usize, k: usize| -> Label {
            (i + (nx + 1) * (j + (ny + 1) * k)) as Label
        };
        let cell = |i: usize, j: usize| -> Label { (i + nx * j) as Label };

        let mut points = Vec::with_capacity(2 * (nx + 1) * (ny + 1));
        for k in 0..2 {
            for j in 0..=ny {
                for i in 0..=nx {
                    points.push(Vec3::new(
                        i as Scalar,
                        j as Scalar,
                        k as Scalar,
                    ));
                }
            }
        }

        // Internal faces, generated cell by cell, which sorts them by
        // (owner, neighbour).
        let mut internal: Vec<(Label, Label, Vec<Label>)> = Vec::new();
        // Boundary faces, one bucket per patch, so the patch order of the
        // written boundary file is the bucket order.
        const XMIN: usize = 0;
        const XMAX: usize = 1;
        const YMIN: usize = 2;
        const YMAX: usize = 3;
        const ZMIN: usize = 4;
        const ZMAX: usize = 5;
        const POCKET: usize = 6;
        let names = [
            "xmin", "xmax", "ymin", "ymax", "zmin", "zmax", "wall_pocket",
        ];
        let mut buckets: Vec<Vec<(Label, Vec<Label>)>> = vec![Vec::new(); names.len()];

        for j in 0..ny {
            for i in 0..nx {
                buckets[ZMIN].push((
                    cell(i, j),
                    vec![
                        pt(i, j, 0),
                        pt(i + 1, j, 0),
                        pt(i + 1, j + 1, 0),
                        pt(i, j + 1, 0),
                    ],
                ));
                buckets[ZMAX].push((
                    cell(i, j),
                    vec![
                        pt(i, j, 1),
                        pt(i, j + 1, 1),
                        pt(i + 1, j + 1, 1),
                        pt(i + 1, j, 1),
                    ],
                ));
                if i == 0 {
                    buckets[XMIN].push((
                        cell(i, j),
                        vec![
                            pt(i, j, 0),
                            pt(i, j, 1),
                            pt(i, j + 1, 1),
                            pt(i, j + 1, 0),
                        ],
                    ));
                }
                if i == nx - 1 {
                    buckets[XMAX].push((
                        cell(i, j),
                        vec![
                            pt(i + 1, j, 0),
                            pt(i + 1, j + 1, 0),
                            pt(i + 1, j + 1, 1),
                            pt(i + 1, j, 1),
                        ],
                    ));
                }
                if j == 0 {
                    buckets[YMIN].push((
                        cell(i, j),
                        vec![
                            pt(i, j, 0),
                            pt(i + 1, j, 0),
                            pt(i + 1, j, 1),
                            pt(i, j, 1),
                        ],
                    ));
                }
                if j == ny - 1 {
                    buckets[YMAX].push((
                        cell(i, j),
                        vec![
                            pt(i, j + 1, 0),
                            pt(i, j + 1, 1),
                            pt(i + 1, j + 1, 1),
                            pt(i + 1, j + 1, 0),
                        ],
                    ));
                }

                if i + 1 < nx {
                    internal.push((
                        cell(i, j),
                        cell(i + 1, j),
                        vec![
                            pt(i + 1, j, 0),
                            pt(i + 1, j, 1),
                            pt(i + 1, j + 1, 1),
                            pt(i + 1, j + 1, 0),
                        ],
                    ));
                }
                if j + 1 < ny {
                    internal.push((
                        cell(i, j),
                        cell(i, j + 1),
                        vec![
                            pt(i, j + 1, 0),
                            pt(i, j + 1, 1),
                            pt(i + 1, j + 1, 1),
                            pt(i + 1, j + 1, 0),
                        ],
                    ));
                }
            }
        }

        // Seal (px, py): its internal faces become boundary faces of its own
        // patch, owned by the pocket cell.
        let pocket = cell(px, py);
        let pocket_faces: Vec<Vec<Label>> = internal
            .iter()
            .filter(|(o, n, _)| *o == pocket || *n == pocket)
            .map(|(_, _, fv)| fv.clone())
            .collect();
        internal.retain(|(o, n, _)| *o != pocket && *n != pocket);
        for fv in pocket_faces {
            buckets[POCKET].push((pocket, fv));
        }

        let n_if = internal.len();
        let mut raw = raw_with_patches(&[]);
        raw.points = points;
        raw.faces = Vec::with_capacity(n_if + buckets.iter().map(|b| b.len()).sum::<usize>());
        raw.owner = Vec::with_capacity(raw.faces.capacity());
        raw.neighbour = Vec::with_capacity(n_if);
        for (o, n, fv) in &internal {
            raw.faces.push(fv.clone());
            raw.owner.push(*o);
            raw.neighbour.push(*n);
        }
        let mut start = 0usize;
        raw.patches = names
            .iter()
            .zip(&buckets)
            .map(|(name, b)| {
                let pi = PatchInfo {
                    name: (*name).to_string(),
                    type_name: "patch".to_string(),
                    kind: PatchKind::Generic,
                    start,
                    size: b.len(),
                    nbr_patch: None,
                };
                start += b.len();
                for (o, fv) in b {
                    raw.faces.push(fv.clone());
                    raw.owner.push(*o);
                }
                pi
            })
            .collect();
        raw
    }

    /// The pocket is its own region; the default keeps the two-cell region,
    /// drops the pocket, and the polyMesh that lands on disk carries the
    /// reduced counts.
    #[test]
    fn the_default_drops_a_sealed_pocket_and_the_written_polymesh_shows_it() -> Result<()> {
        // The regions are what the count says they are before anything moves.
        let probe = two_tet_with_a_sealed_third();
        let host = build_host_mesh(&probe)?;
        let (n_regions, region_of) = cell_regions(&host);
        assert_eq!(n_regions, 2);
        assert_eq!(region_of, vec![0, 0, 1]);

        let mut raw = probe;
        let (dropped_internal, dropped_boundary) = keep_region(&mut raw, &region_of, 0);
        assert_eq!(dropped_internal, 0, "the pocket adds no internal faces");
        assert_eq!(dropped_boundary, vec![("walls".to_string(), 4)]);

        // 7 faces survive: 1 internal + 3 east + 3 top; cells 0 and 1 keep
        // their numbers, the pocket's 2 is gone.
        assert_eq!(raw.neighbour.len(), 1);
        assert_eq!(raw.faces.len(), 7);
        assert_eq!(raw.owner.len(), 7);
        assert_eq!(n_cells_of(&raw), 2);
        assert_eq!(raw.neighbour, vec![1]);
        // The patches keep their order; the pocket's own is empty in place.
        let names: Vec<&str> = raw.patches.iter().map(|p| p.name.as_str()).collect();
        assert_eq!(names, ["east", "top", "walls"]);
        assert_eq!(
            raw.patches[2].start, 6,
            "walls starts after the six kept boundary faces"
        );
        assert_eq!(raw.patches[2].size, 0);

        // And the polyMesh that lands on disk is the reduced one.
        let dir = scratch("drop-pocket");
        write_poly_mesh_raw(&dir, &raw)?;
        let back = read_poly_mesh(&dir)?;
        assert_eq!(back.neighbour.len(), 1);
        assert_eq!(back.faces.len(), 7);
        assert_eq!(n_cells_of(&back), 2);
        assert_eq!(back.patches.len(), 3);
        assert_eq!((back.patches[2].start, back.patches[2].size), (6, 0));

        Ok(())
    }

    /// A 12 x 12 slab with one sealed cell clears both refusal thresholds
    /// (0.7 % of the cells, largest 143x the second), so the default path
    /// really drops it - and the counts come out as the arithmetic says.
    #[test]
    fn a_pocket_in_a_big_mesh_is_dropped_by_default() -> Result<()> {
        let (nx, ny) = (12usize, 12usize);
        let mut raw = hex_slab_with_pocket(nx, ny, 1, 1);

        // 144 cells; 264 - 4 internal faces, four of them moved to boundary
        // to seal the pocket; 336 + 4 boundary faces, four on wall_pocket.
        assert_eq!(n_cells_of(&raw), (nx * ny) as Label);
        assert_eq!(raw.neighbour.len(), 2 * nx * ny - nx - ny - 4);
        assert_eq!(
            raw.faces.len(),
            2 * nx * ny - nx - ny - 4 + 2 * nx * ny + 2 * nx + 2 * ny + 4
        );

        handle_cell_regions(&mut raw, false)?;

        assert_eq!(n_cells_of(&raw), (nx * ny - 1) as Label);
        assert_eq!(raw.neighbour.len(), 2 * nx * ny - nx - ny - 4);
        // The pocket's six boundary faces go with it: four on wall_pocket,
        // its own zmin and zmax. The four sealing faces were already boundary
        // when they moved, so the internal count loses exactly the four that
        // joined the pocket to its neighbours, and the slab's own 336 boundary
        // faces lose the pocket's two.
        assert_eq!(raw.faces.len(), 2 * nx * ny - nx - ny - 4 + 2 * nx * ny + 2 * nx + 2 * ny - 2);
        assert_eq!(raw.faces.len(), raw.owner.len());
        // Every patch survives in order; only the pocket's own is empty, and
        // it starts where the sum of the kept faces before it says.
        let sizes: Vec<usize> = raw.patches.iter().map(|p| p.size).collect();
        assert_eq!(
            sizes,
            [nx, nx, ny, ny, nx * ny - 1, nx * ny - 1, 0],
            "the pocket's own zmin/zmax faces are dropped with it"
        );
        let wall_pocket = raw.patches.last().unwrap();
        assert_eq!(wall_pocket.start, 2 * nx + 2 * ny + 2 * nx * ny - 2);

        Ok(())
    }

    /// `-keepRegions`: everything stays exactly as it arrived, however many
    /// regions the mesh holds.
    #[test]
    fn keep_regions_keeps_every_region_and_touches_nothing() -> Result<()> {
        let mut raw = two_tet_with_a_sealed_third();
        let before = raw.clone();
        handle_cell_regions(&mut raw, true)?;
        assert_eq!(raw.faces, before.faces);
        assert_eq!(raw.owner, before.owner);
        assert_eq!(raw.neighbour, before.neighbour);
        // `PatchInfo` carries no `PartialEq`; the fields that the drop would
        // rewrite are the ones worth comparing.
        let patch_id = |p: &PatchInfo| (p.name.clone(), p.start, p.size);
        let same: Vec<_> = raw
            .patches
            .iter()
            .zip(&before.patches)
            .map(|(a, b)| (patch_id(a), patch_id(b)))
            .collect();
        assert!(same.iter().all(|(a, b)| a == b));
        assert_eq!(raw.patches.len(), before.patches.len());
        Ok(())
    }

    /// A pocket that is a third of the mesh fails both thresholds - that is
    /// a second domain, and the refusal names it and the way out.
    #[test]
    fn a_second_region_that_is_not_pocket_sized_is_refused() {
        let mut raw = two_tet_with_a_sealed_third();
        let msg = match handle_cell_regions(&mut raw, false) {
            Err(e) => e.to_string(),
            Ok(()) => panic!("dropping 1 of 3 cells (a third of the mesh) must be refused"),
        };
        assert!(msg.contains("2 cell regions"), "{msg}");
        assert!(msg.contains("second domain"), "{msg}");
        assert!(msg.contains("-keepRegions"), "{msg}");
        // A refusal decides nothing: the mesh is untouched for the next run.
        assert_eq!(raw.faces.len(), 11);
        assert_eq!(n_cells_of(&raw), 3);
    }

    /// The `regions:` line, spelled out: counts first, then the faces split
    /// into internal and per patch.
    #[test]
    fn the_drop_message_splits_the_faces_into_internal_and_per_patch() {
        let msg = dropped_regions_message(
            9,
            24,
            77,
            19,
            &[("wall_ground_land".to_string(), 58)],
        );
        assert_eq!(
            msg,
            "regions: 9; dropped 8 sealed region(s), 24 cell(s), 77 face(s) \
             (19 internal, 58 on wall_ground_land)"
        );

        // No boundary face dropped - the line does not end in a dangling
        // comma.
        assert_eq!(
            dropped_regions_message(2, 1, 3, 3, &[]),
            "regions: 2; dropped 1 sealed region(s), 1 cell(s), 3 face(s) \
             (3 internal)"
        );
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
