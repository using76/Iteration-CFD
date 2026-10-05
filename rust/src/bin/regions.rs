// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! `ofgpu-regions` - the region layout of SPEC-LIT 97, from the command
//! line: split one polyMesh carrying `cellZones` into one polyMesh per
//! region plus a `regions.json`, check a layout against its rules R1-R6
//! and §47.4's own pairing, and list what a manifest names. A layout's
//! meshes carry no physics; a CASE composes the layout through
//! `"mesh": {"regions": ...}` (R8), which is `ofgpu-cht`'s business.

use std::path::Path;
use std::process::ExitCode;

use ofgpu::cht::{PairingTolerances, RegionInput, RegionKind, ThermalMesh};
use ofgpu::error::Result;
use ofgpu::io::polymesh::{read_cell_zones, read_poly_mesh};
use ofgpu::io::regions::{load, read_manifest, split_by_zones, write_layout, RegionsManifest};
use ofgpu::mesh::geometry::cell_regions;
use ofgpu::Error;

const USAGE: &str = "\
ofgpu-regions split <polyMeshDir> <outDir> [-fluid <zone>]
ofgpu-regions check <regions.json>
ofgpu-regions list <regions.json>

The region layout of SPEC-LIT 97: one polyMesh per region and a regions.json
naming the regions and their conformal interface patch pairs.";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    match run(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("\nofgpu-regions: {e}");
            ExitCode::from(1)
        }
    }
}

fn run(args: &[String]) -> Result<()> {
    let Some(cmd) = args.get(1).map(|s| s.as_str()) else {
        eprintln!("{USAGE}");
        return Err(Error::Config("a subcommand is required".to_string()));
    };
    // The one flag is `-fluid <zone>`; everything else is a positional.
    let mut fluid: Option<String> = None;
    let mut pos: Vec<String> = Vec::new();
    let mut i = 2usize;
    while i < args.len() {
        match args[i].as_str() {
            "-fluid" => {
                if fluid.is_some() {
                    return Err(Error::Config("-fluid is given twice".to_string()));
                }
                let Some(v) = args.get(i + 1) else {
                    return Err(Error::Config("-fluid needs a zone name".to_string()));
                };
                fluid = Some(v.clone());
                i += 1;
            }
            other => pos.push(other.to_string()),
        }
        i += 1;
    }
    match (cmd, pos.len()) {
        ("split", 2) => cmd_split(Path::new(&pos[0]), Path::new(&pos[1]), fluid.as_deref()),
        ("check", 1) => cmd_check(Path::new(&pos[0])),
        ("list", 1) => cmd_list(Path::new(&pos[0])),
        _ => {
            eprintln!("{USAGE}");
            Err(Error::Config(format!(
                "expected `split <polyMeshDir> <outDir> [-fluid <zone>]`, \
                 `check <regions.json>` or `list <regions.json>`, got `{cmd}` \
                 with {} positional argument(s)",
                pos.len()
            )))
        }
    }
}

/// `split <polyMeshDir> <outDir> [-fluid <zone>]`: ONE polyMesh carrying
/// `cellZones` becomes the layout - one standalone polyMesh per zone (R7)
/// plus `regions.json`. `-fluid` moves that zone to `regions[0]` BEFORE the
/// split, so the fluid block keeps SPEC-LIT §47.4's numbering; with no
/// `-fluid`, every region is `solid`. An existing `regions.json` is refused
/// by `write_layout` - a layout is never overwritten in place.
fn cmd_split(poly_dir: &Path, out_dir: &Path, fluid: Option<&str>) -> Result<()> {
    let raw = read_poly_mesh(poly_dir)?;
    let mut zones = read_cell_zones(poly_dir)?;
    if let Some(name) = fluid {
        let i = zones.iter().position(|(z, _)| z == name)
            .ok_or_else(|| Error::Config(format!("-fluid {name}: no such zone")))?;
        let z = zones.remove(i);
        zones.insert(0, z);
    }
    let (regions, ifaces) = split_by_zones(&raw, &zones)?;
    let kinds: Vec<RegionKind> = zones.iter().map(|(name, _)| {
        if fluid == Some(name.as_str()) { RegionKind::Fluid } else { RegionKind::Solid }
    }).collect();
    let source = serde_json::json!({
        "tool": "ofgpu-regions",
        "version": env!("CARGO_PKG_VERSION"),
        "geometry": poly_dir.display().to_string(),
        "config": fluid.map(|f| format!("-fluid {f}")).unwrap_or_default(),
    });
    write_layout(out_dir, &regions, &kinds, &ifaces, Some(source))?;
    println!("ofgpu-regions | split '{}' -> {} region(s), {} interface(s), {}",
        poly_dir.display(), regions.len(), ifaces.len(), out_dir.display());
    for (i, (name, raw)) in regions.iter().enumerate() {
        let cells = raw.owner.iter().chain(raw.neighbour.iter()).copied()
            .max().map_or(0, |m| m as usize + 1);
        let kind = if kinds[i] == RegionKind::Fluid { "fluid" } else { "solid" };
        println!("  region '{name}': kind {kind} cells {} faces {} patches {}",
            cells, raw.faces.len(), raw.patches.len());
    }
    for si in &ifaces {
        println!("  {} <-> {}: {} faces", si.patch_a, si.patch_b, si.n_faces);
    }
    Ok(())
}

/// `check <regions.json>`: the two verdicts. First the layout's own rules -
/// R1-R6 with the index-pairing check of the one dimensionless `tolerance`
/// - then SPEC-LIT §47.4's own pairing, run with its default tolerances
/// exactly as the solve would. Either failing exits 1, and the message
/// names WHICH verdict failed.
fn cmd_check(manifest: &Path) -> Result<()> {
    let layout =
        load(manifest).map_err(|e| Error::Config(format!("layout R1-R6: FAIL - {e}")))?;
    println!("layout R1-R6: PASS");
    for pc in &layout.pairing {
        println!("  {}: {} faces, worst centroid/sqrt(area) = {:#e}, area = {:#e}, normal+1 = {:#e}",
            pc.name, pc.n_faces, pc.worst_centroid, pc.worst_area, pc.worst_normal);
    }
    for lr in &layout.regions {
        let (n, _) = cell_regions(&lr.mesh);
        println!("  region '{}': {} connected cell region(s)", lr.name, n);
    }
    let inputs: Vec<RegionInput> = layout.regions.iter()
        .map(|r| RegionInput { name: r.name.clone(), kind: r.kind, mesh: &r.mesh })
        .collect();
    let mesh = ThermalMesh::build(&inputs, &layout.interfaces, PairingTolerances::default())
        .map_err(|e| Error::Config(format!("solver pairing (SPEC-LIT 47.4): FAIL - {e}")))?;
    println!("solver pairing (SPEC-LIT 47.4):");
    let rep = &mesh.report;
    println!("  {} pair(s): worst centroid = {:#e}, area = {:#e}, normal+1 = {:#e}, non-orth = {:.2} deg, total area = {:#e}",
        rep.n_pairs, rep.worst_centroid, rep.worst_area, rep.worst_normal,
        rep.non_orth_deg(), rep.total_area);
    Ok(())
}

/// `list <regions.json>`: the manifest alone - no mesh is read; the cheap
/// look at a layout's names before any of it is loaded.
fn cmd_list(manifest: &Path) -> Result<()> {
    let m: RegionsManifest = read_manifest(manifest)?;
    println!("{} | {} region(s), {} interface(s)",
        manifest.display(), m.regions.len(), m.interfaces.len());
    for r in &m.regions {
        println!("  region '{}': kind {}, polyMesh {}, material {}",
            r.name, r.kind, r.poly_mesh, r.material.as_deref().unwrap_or("-"));
    }
    for (i, e) in m.interfaces.iter().enumerate() {
        let faces = e.faces.map(|n| n.to_string()).unwrap_or_else(|| "?".to_string());
        println!("  interface {i}: '{}' <-> '{}' ({} <-> {}): {faces} faces, tolerance {:#e}",
            e.regions[0], e.regions[1], e.patches[0], e.patches[1], e.tolerance);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ofgpu::blockgen::{raw_mesh, BlockSpec, GradedAxis};
    use ofgpu::io::polymesh::write_poly_mesh_raw;

    /// The bin test's own fixture: the same 4x4x8 union block
    /// `io::regions`' tests split, rebuilt here field for field. A `[[bin]]`
    /// target links `ofgpu` compiled WITHOUT `cfg(test)`, so the library
    /// test helper's fixture does not exist here - and the two must stay
    /// the same block: they differ, the split's cell counts differ.
    fn union_block() -> BlockSpec {
        BlockSpec {
            x: GradedAxis { lo: 0.0, hi: 1.0, n: 4, expansion: 1.0, two_sided: false },
            y: GradedAxis { lo: 0.0, hi: 1.0, n: 4, expansion: 1.0, two_sided: false },
            z: GradedAxis { lo: 0.0, hi: 2.0, n: 8, expansion: 1.0, two_sided: false },
            patch_name: ["xmin", "xmax", "ymin", "ymax", "zmin", "zmax"].map(String::from),
            patch_type: ["patch"; 6].map(String::from),
            windows: Vec::new(),
            cyclic: Vec::new(),
        }
    }

    #[test]
    fn usage_names_the_three_subcommands() {
        for want in [
            "split <polyMeshDir> <outDir>",
            "check <regions.json>",
            "list <regions.json>",
            "-fluid",
        ] {
            assert!(USAGE.contains(want), "USAGE must name {want}");
        }
    }

    #[test]
    #[cfg_attr(feature = "single", ignore = "fails at f32: SPEC-LIT 112.3")]
    fn split_check_and_list_run_on_a_two_zone_block_on_disk() {
        let d = std::env::temp_dir()
            .join(format!("ofgpu-regions-bin-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        let parent = d.join("parent");
        write_poly_mesh_raw(&parent.join("polyMesh"), &raw_mesh(&union_block()).unwrap()).unwrap();
        // The cellZones file, compact form, by hand: lower holds cells
        // 0..63, upper 64..127 - lower first, so regions[0] is the bottom.
        let ls: Vec<String> = (0..128).map(|c| c.to_string()).collect();
        let zone = |name: &str, ls: &[String]| {
            format!(
                "{name}\n{{\n    type cellZone;\n    cellLabels      List<label> {}({});\n}}",
                ls.len(),
                ls.join(" ")
            )
        };
        let cz = format!(
            "FoamFile {{ version 2.0; format ascii; class regIOobject; \
             location \"constant/polyMesh\"; object cellZones; }}\n2\n(\n{}\n{}\n)\n",
            zone("lower", &ls[..64]), zone("upper", &ls[64..])
        );
        std::fs::write(parent.join("polyMesh").join("cellZones"), cz).unwrap();
        let out = d.join("out");
        let a = |list: &[&str]| list.iter().map(|s| s.to_string()).collect::<Vec<String>>();
        run(&a(&["ofgpu-regions", "split", parent.to_str().unwrap(), out.to_str().unwrap(), "-fluid", "lower"]))
            .expect("split runs");
        let rp = out.join("regions.json");
        let m = read_manifest(&rp).expect("manifest reads back");
        let names: Vec<&str> = m.regions.iter().map(|r| r.name.as_str()).collect();
        assert_eq!(names, ["lower", "upper"], "the split's regions, in file order");
        assert_eq!(m.interfaces.len(), 1);
        assert_eq!(m.interfaces[0].faces, Some(16), "16 shared faces");
        run(&a(&["ofgpu-regions", "check", rp.to_str().unwrap()])).expect("check runs");
        run(&a(&["ofgpu-regions", "list", rp.to_str().unwrap()])).expect("list runs");
        let e = run(&a(&["ofgpu-regions", "split", parent.to_str().unwrap(), out.to_str().unwrap()]))
            .expect_err("a second split into the same out is refused");
        assert!(e.to_string().contains("regions.json"), "{e}");
        let _ = std::fs::remove_dir_all(&d);
    }
}
