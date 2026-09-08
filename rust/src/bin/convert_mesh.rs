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
//! A directory that already holds a polyMesh is refused rather than
//! overwritten: the files on disk may be the only copy of a mesh some
//! pre-processing chain produced, and nothing about the command line says
//! the operator expected them to be replaced.
//!
//! Provenance: ORIGINAL - the command-line front end to `io/msh.rs` and
//! `io/polymesh.rs`; the reader and the writer are covered by those files'
//! own headers. This one is argument parsing, the type convention and
//! overrides, the overwrite guard and the printed summary. No GPL-licensed
//! source was consulted.

use std::collections::HashMap;
use std::path::Path;
use std::process::ExitCode;

use ofgpu::io::msh::read_msh;
use ofgpu::io::polymesh::{PolyMeshRaw, write_poly_mesh_raw};
use ofgpu::mesh::PatchKind;
use ofgpu::{Error, Result};

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
        "usage: ofgpu-convert-mesh <in.msh> <outCaseDir> [-type <patchName>=<type>]..."
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

fn run(args: &[String]) -> Result<()> {
    // ---- flags and positionals --------------------------------------------
    let mut positional: Vec<&String> = Vec::new();
    // A `-type` repeated for the same patch: the last one wins, the way a
    // repeated flag reads everywhere else.
    let mut overrides: HashMap<String, String> = HashMap::new();
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
    let n_cells = raw
        .owner
        .iter()
        .chain(raw.neighbour.iter())
        .copied()
        .max()
        .unwrap_or(-1)
        + 1;

    println!(
        "[convert] {}: {} cells, {} points, {} faces ({} internal, {} boundary)",
        dir.display(),
        n_cells,
        raw.points.len(),
        raw.faces.len(),
        raw.neighbour.len(),
        raw.faces.len() - raw.neighbour.len()
    );
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

#[cfg(test)]
mod tests {
    use super::*;
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
