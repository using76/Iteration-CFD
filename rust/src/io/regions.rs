// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

//! The region layout of `docs/10-fsi-solid-mesh-plan.md` section C
//! (SPEC-LIT §97): a `regions.json` manifest naming one complete polyMesh
//! per region, with conformal interfaces carried as index-paired patch
//! pairs.
//!
//! Provenance: ORIGINAL. The manifest format is this project's own (the
//! plan's §C); the `cellZones` reader (`io::polymesh::read_cell_zones`)
//! reads a published, uncopyrightable case-format FILE FORMAT learned from
//! the shape of the files themselves; the splitter and the layout writer
//! are original and carry no numerics. Written from SPEC-LIT §31.1, §47.4,
//! §47.14 and §97. OpenFOAM's `splitMeshRegions`, `cellZones` and `polyMesh`
//! classes are GPL and were not opened.
//! No GPL-licensed source was consulted.
//!
//! R8 - a case naming the manifest - lives in `crate::io::case_cht`
//! (SPEC-LIT §97).

use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::cht::{InterfaceRequest, RegionKind};
use crate::error::{Error, IoContext, Result};
use crate::io::case_json::parse_jsonc_file;
use crate::io::polymesh::{
    build_host_mesh, check_patch_name, read_poly_mesh, write_poly_mesh_raw, PolyMeshRaw,
};
use crate::mesh::{HostMesh, PatchInfo, PatchKind};
use crate::{Label, Scalar, Vec3};

// ==========================================================================
//  The manifest (SPEC-LIT §97: the layout as the contract)
// ==========================================================================

/// `regions.json`: the layout manifest of the plan's §C. Every field is
/// refused if it is not the one declared (`deny_unknown_fields`), so a
/// mistyped key is an error rather than a setting silently ignored
/// (SPEC-LIT §13.4).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RegionsManifest {
    pub version: u32,
    #[serde(default = "metres")]
    pub units: String,
    /// In the producer's order; the CASE's own `regions[]` order fixes the
    /// concatenated numbering (R5's statement of whose numbering it is).
    pub regions: Vec<RegionEntry>,
    #[serde(default)]
    pub interfaces: Vec<InterfaceEntry>,
    /// Data for a human, never read back (SPEC-LIT §13.4: kept verbatim,
    /// not a setting).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<serde_json::Value>,
}

/// One region of a layout. R4: `kind` is `"fluid"` or `"solid"`; R6: the
/// path is relative to the manifest's own directory.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RegionEntry {
    pub name: String,
    pub kind: String,
    #[serde(rename = "polyMesh")]
    pub poly_mesh: String,
    /// Optional on a solid: the material BINDS in the case's own `material`
    /// block; a fluid region carrying one is refused (R4).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub material: Option<String>,
}

/// One interface of a layout (R2/R3). `tolerance` is ONE dimensionless
/// number used for all three R2 inequalities - M4's reading of the layout,
/// and now this reader's; the solver's own four-number
/// `PairingTolerances::default()` check still runs at solve time.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InterfaceEntry {
    pub regions: [String; 2],
    /// `["<a>_to_<b>", "<b>_to_<a>"]` (R3) - refused in any other form.
    pub patches: [String; 2],
    /// Optional; when present it must equal both patches' face counts.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub faces: Option<usize>,
    #[serde(default = "default_tolerance")]
    pub tolerance: f64,
}

fn metres() -> String {
    "m".to_string()
}

/// The one dimensionless tolerance for all three R2 inequalities (M4's
/// default).
fn default_tolerance() -> f64 {
    1e-9
}

fn kind_of(s: &str) -> Option<RegionKind> {
    match s {
        "fluid" => Some(RegionKind::Fluid),
        "solid" => Some(RegionKind::Solid),
        _ => None,
    }
}

fn kind_str(k: RegionKind) -> &'static str {
    match k {
        RegionKind::Fluid => "fluid",
        RegionKind::Solid => "solid",
    }
}

// ==========================================================================
//  Reading a manifest, and what is refused by name
// ==========================================================================

/// Parse and fully validate a `regions.json`. Every refusal names the
/// offending entry and the rule it breaks: `version` other than 1; `units`
/// other than metres (the tool that made the mesh scales it); a region
/// name declared twice or unwritable as one boundary-file token; a `kind`
/// that is not `fluid`/`solid`, or a fluid carrying a `material` (R4);
/// more than one fluid, or a fluid that is not `regions[0]` (SPEC-LIT
/// §47.4's numbering); an interface naming one region twice or an
/// undeclared region, patches that are not the `<a>_to_<b>` pair (R3), or
/// a `tolerance` that is not `> 0`.
pub fn read_manifest(path: &Path) -> Result<RegionsManifest> {
    let m: RegionsManifest = parse_jsonc_file(path)?;
    let what = path.display();
    if m.version != 1 {
        return Err(Error::Config(format!(
            "{what}: version {} - this reader reads version 1 only",
            m.version
        )));
    }
    if m.units != "m" {
        return Err(Error::Config(format!(
            "{what}: units '{}' - a layout's meshes are in metres; scale the \
             mesh by the tool that made it and write units \"m\"",
            m.units
        )));
    }
    for (i, r) in m.regions.iter().enumerate() {
        check_patch_name(&r.name, "region")?;
        if m.regions[..i].iter().any(|o| o.name == r.name) {
            return Err(Error::Config(format!(
                "{what}: region '{}' is declared twice", r.name
            )));
        }
        if kind_of(&r.kind).is_none() {
            return Err(Error::Config(format!(
                "{what}: region '{}' has kind '{}' - kind is \"fluid\" or \
                 \"solid\" (R4)",
                r.name, r.kind
            )));
        }
        if r.kind == "fluid" && r.material.is_some() {
            return Err(Error::Config(format!(
                "{what}: region '{}' is a fluid and carries material '{}' - \
                 only a solid region carries a material (R4)",
                r.name,
                r.material.as_deref().unwrap_or("")
            )));
        }
    }
    let fluids: Vec<&RegionEntry> =
        m.regions.iter().filter(|r| r.kind == "fluid").collect();
    if fluids.len() > 1 {
        return Err(Error::Config(format!(
            "{what}: regions '{}' and '{}' are both fluid - at most one \
             fluid region (SPEC-LIT 47.4)",
            fluids[0].name, fluids[1].name
        )));
    }
    if let Some(f) = m.regions.iter().position(|r| r.kind == "fluid") {
        if f != 0 {
            return Err(Error::Config(format!(
                "{what}: the fluid region '{}' is regions[{f}] - it must be \
                 regions[0] so the fluid block keeps its cell and \
                 boundary-face numbering (SPEC-LIT 47.4)",
                m.regions[f].name
            )));
        }
    }
    for (i, e) in m.interfaces.iter().enumerate() {
        if e.regions[0] == e.regions[1] {
            return Err(Error::Config(format!(
                "{what}: interface {i} names region '{}' on both sides",
                e.regions[0]
            )));
        }
        for name in &e.regions {
            if !m.regions.iter().any(|r| &r.name == name) {
                return Err(Error::Config(format!(
                    "{what}: interface {i} names region '{name}', which the \
                     manifest does not declare"
                )));
            }
        }
        let expect = format!(
            "[\"{}_to_{}\", \"{}_to_{}\"]",
            e.regions[0], e.regions[1], e.regions[1], e.regions[0]
        );
        if e.patches[0] != format!("{}_to_{}", e.regions[0], e.regions[1])
            || e.patches[1] != format!("{}_to_{}", e.regions[1], e.regions[0])
        {
            return Err(Error::Config(format!(
                "{what}: interface {i} has patches {:?} - the layout's names \
                 are {expect} (R3)",
                e.patches
            )));
        }
        if !(e.tolerance > 0.0) {
            return Err(Error::Config(format!(
                "{what}: interface {i} has tolerance {} - it must be > 0",
                e.tolerance
            )));
        }
    }
    Ok(m)
}

/// R6: a layout's polyMesh path is relative to the manifest's directory;
/// an absolute path, or one that climbs out with `..`, is refused.
fn region_poly_dir(dir: &Path, p: &str) -> Result<PathBuf> {
    let rel = Path::new(p);
    let climbs = rel
        .components()
        .any(|c| c == std::path::Component::ParentDir);
    if rel.is_absolute() || climbs {
        return Err(Error::Config(format!(
            "polyMesh path '{p}' is absolute or climbs out with '..' - a \
             layout's paths are relative to regions.json's directory (R6)"
        )));
    }
    Ok(dir.join(rel))
}

/// R1's one added check: a region's internal faces are ordered by owner
/// then neighbour, so the split and SPEC-LIT §47.4's concatenation keep
/// the LDU invariant without a re-sort. Everything else R1 asks -
/// `owner[f] < neighbour[f]`, the counts, the patches - is what
/// [`build_host_mesh`] already refuses. `what` names the region.
pub fn check_r1(raw: &PolyMeshRaw, what: &str) -> Result<()> {
    let n_if = raw.neighbour.len();
    let mut prev: Option<(Label, Label)> = None;
    for f in 0..n_if {
        let on = (raw.owner[f], raw.neighbour[f]);
        if let Some(p) = prev {
            if on < p {
                return Err(Error::Mesh(format!(
                    "{what}: internal face {f} has (owner, neighbour) = \
                     ({}, {}), which sorts before face {}'s ({}, {}) - a \
                     region's polyMesh keeps its internal faces ordered by \
                     owner then neighbour (R1)",
                    on.0, on.1, f - 1, p.0, p.1
                )));
            }
        }
        prev = Some(on);
    }
    Ok(())
}

/// R2's three numbers for one interface, all dimensionless: the worst
/// `|Cf_a - Cf_b| / sqrt(|Sf_a|)`, the worst `||Sf_a| - |Sf_b|| / |Sf_a|`,
/// and the worst `n_a . n_b + 1`. `ofgpu-regions check` prints them (SPEC-LIT
/// §97); they are the same three inequalities M4's `regions_check.py` reads
/// the manifest's one `tolerance` against.
#[derive(Debug, Clone)]
pub struct PairingCheck {
    pub name: String,
    pub n_faces: usize,
    pub worst_centroid: Scalar,
    pub worst_area: Scalar,
    pub worst_normal: Scalar,
}

/// R2 and R3, checked by index: both patches exist in their region's mesh,
/// both have `type patch` (R3), the face counts agree (and equal `faces`
/// when given), and for every `k` the three inequalities of the manifest's
/// one dimensionless `tolerance` hold. Returns one [`InterfaceRequest`]
/// (`r_c = 0`: perfect contact, SPEC-LIT §47.2) and one [`PairingCheck`]
/// per interface. SPEC-LIT §47.4's centroid-hash pairing stays where it
/// already is - `ThermalMesh::couple`, at solve time; this is the
/// index-pairing check in front of it.
pub fn check_layout(
    m: &RegionsManifest,
    regions: &[LoadedRegion],
) -> Result<(Vec<InterfaceRequest>, Vec<PairingCheck>)> {
    let mut reqs = Vec::new();
    let mut pairing = Vec::new();
    for (i, e) in m.interfaces.iter().enumerate() {
        let what = format!("interface {i} ('{}' <-> '{}')", e.regions[0], e.regions[1]);
        let mut sides = [0usize; 2];
        for s in 0..2 {
            sides[s] = regions
                .iter()
                .position(|r| r.name == e.regions[s])
                .ok_or_else(|| {
                    Error::Config(format!(
                        "{what}: region '{}' is not in the manifest",
                        e.regions[s]
                    ))
                })?;
        }
        let (ia, ib) = (sides[0], sides[1]);
        let (ra, rb) = (&regions[ia], &regions[ib]);
        let pa = patch_of(ra, &e.patches[0], &what)?;
        let pb = patch_of(rb, &e.patches[1], &what)?;
        if pa.type_name != "patch" || pb.type_name != "patch" {
            return Err(Error::Config(format!(
                "{what}: patch '{}' has type '{}' and '{}' has type '{}' - \
                 an interface patch has type \"patch\" (R3)",
                pa.name, pa.type_name, pb.name, pb.type_name
            )));
        }
        if pa.size != pb.size || e.faces.is_some_and(|n| n != pa.size) {
            return Err(Error::Config(format!(
                "{what}: {} faces, {} faces, manifest says {:?} - the two \
                 patches of an interface carry the same number of faces, \
                 and the manifest's `faces`, when present, says what both \
                 carry (R2). Non-conformal (AMI) interfaces are not \
                 implemented (SPEC-LIT 47.4)",
                pa.size, pb.size, e.faces
            )));
        }
        let tol = e.tolerance;
        let (mut wc, mut wa, mut wn) = (0.0f64, 0.0f64, 0.0f64);
        for k in 0..pa.size {
            let bfa = pa.start + k;
            let bfb = pb.start + k;
            let scale = ra.mesh.b_mag_sf[bfa].sqrt();
            let d = (ra.mesh.b_cf[bfa] - rb.mesh.b_cf[bfb]).mag() / scale;
            let ea = (ra.mesh.b_mag_sf[bfa] - rb.mesh.b_mag_sf[bfb]).abs()
                / ra.mesh.b_mag_sf[bfa];
            let dn = ra.mesh.b_sf[bfa].normalised().dot(rb.mesh.b_sf[bfb].normalised())
                + 1.0;
            wc = wc.max(d);
            wa = wa.max(ea);
            wn = wn.max(dn);
            if !(d <= tol) {
                return Err(Error::Config(format!(
                    "{what} face {k}: |Cf_A - Cf_B|/sqrt(|Sf|) = {d}, \
                     tolerance {tol} - the k-th face of one patch is not \
                     the k-th face of the other (R2); SPEC-LIT 47.4 needs \
                     conformal, matched patches"
                )));
            }
            if !(ea <= tol) {
                return Err(Error::Config(format!(
                    "{what} face {k}: the two areas differ by {ea} \
                     relative, tolerance {tol} (R2)"
                )));
            }
            if !(dn <= tol) {
                return Err(Error::Config(format!(
                    "{what} face {k}: n_A . n_B + 1 = {dn}, tolerance {tol} \
                     - the two faces are the same face seen from opposite \
                     sides, so their outward normals are opposed (R2)"
                )));
            }
        }
        reqs.push(InterfaceRequest::new(ia, &e.patches[0], ib, &e.patches[1], 0.0));
        pairing.push(PairingCheck {
            name: format!("{} <-> {}", e.regions[0], e.regions[1]),
            n_faces: pa.size,
            worst_centroid: wc,
            worst_area: wa,
            worst_normal: wn,
        });
    }
    Ok((reqs, pairing))
}

/// The patch of one loaded region a manifest interface names; the refusal
/// lists the patch names the region actually carries.
fn patch_of(r: &LoadedRegion, name: &str, what: &str) -> Result<PatchInfo> {
    r.mesh
        .patches
        .iter()
        .find(|p| p.name == name)
        .cloned()
        .ok_or_else(|| {
            let names: Vec<&str> =
                r.mesh.patches.iter().map(|p| p.name.as_str()).collect();
            Error::Config(format!(
                "{what}: patch '{name}' does not exist in region '{}' - its \
                 patches are {names:?} (R2)",
                r.name
            ))
        })
}

// ==========================================================================
//  Loading a layout from disk
// ==========================================================================

/// One region of a layout, read and built.
#[derive(Debug)]
pub struct LoadedRegion {
    pub name: String,
    pub kind: RegionKind,
    /// The manifest's optional material name. Nothing here reads it: the
    /// material BINDS in the case's own `material` block (SPEC-LIT §97).
    pub material: Option<String>,
    pub raw: PolyMeshRaw,
    pub mesh: HostMesh,
}

/// A whole layout: the manifest, its regions loaded, and R2 checked.
#[derive(Debug)]
pub struct LoadedLayout {
    pub dir: PathBuf,
    pub manifest: RegionsManifest,
    pub regions: Vec<LoadedRegion>,
    pub interfaces: Vec<InterfaceRequest>,
    pub pairing: Vec<PairingCheck>,
}

/// Read a layout: the manifest, then (R6) each region's polyMesh joined to
/// the manifest's directory, read through [`read_poly_mesh`] and
/// [`build_host_mesh`] with [`check_r1`] in front, then (R2/R3)
/// [`check_layout`]. A region's numbering is its own (R5).
pub fn load(manifest_path: &Path) -> Result<LoadedLayout> {
    let manifest = read_manifest(manifest_path)?;
    let dir = manifest_path
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .to_path_buf();
    let mut regions = Vec::with_capacity(manifest.regions.len());
    for (i, e) in manifest.regions.iter().enumerate() {
        let pm = region_poly_dir(&dir, &e.poly_mesh)?;
        let raw = read_poly_mesh(&pm).map_err(|err| {
            Error::Mesh(format!("regions[{}] ('{}'): {err}", i, e.name))
        })?;
        check_r1(&raw, &format!("regions[{i}] ('{}')", e.name))?;
        let mesh = build_host_mesh(&raw)?;
        regions.push(LoadedRegion {
            name: e.name.clone(),
            kind: kind_of(&e.kind).expect("read_manifest checked kind (R4)"),
            material: e.material.clone(),
            raw,
            mesh,
        });
    }
    let (interfaces, pairing) = check_layout(&manifest, &regions)?;
    Ok(LoadedLayout {
        dir,
        manifest,
        regions,
        interfaces,
        pairing,
    })
}

// ==========================================================================
//  The split (R7)
// ==========================================================================

/// Reverse a face's vertex loop keeping the FIRST vertex: `[f[0], f[n-1],
/// ..., f[1]]`. The normal flips - and this reversal is what a hex mesher's
/// own outward loop is, which is why the split's geometry stays bitwise the
/// sub-block's: blockgen's `zMin` quad `[A, D, C, B]` (`boundary_quad` slot
/// 4) is exactly the internal `+z` quad `[A, B, C, D]` reversed this way,
/// so `face_geometry`'s fan about the vertex average runs in the same
/// order. A plain reversal `[D, C, B, A]` gives the same normal and a
/// last-bit-different centroid.
pub fn reverse_face(f: &[Label]) -> Vec<Label> {
    let mut out = Vec::with_capacity(f.len());
    if let Some(&first) = f.first() {
        out.push(first);
    }
    out.extend(f[1..].iter().rev().copied());
    out
}

/// One interface a split produced: the pair of region INDICES and the pair
/// of patch names, ordered `region_a < region_b` (R2: the k-th face of
/// `patch_a` is the k-th face of `patch_b` with opposite winding).
#[derive(Debug, Clone)]
pub struct SplitInterface {
    pub region_a: usize,
    pub patch_a: String,
    pub region_b: usize,
    pub patch_b: String,
    pub n_faces: usize,
}

/// R7: turn ONE polyMesh plus a `cellZones` list into the layout - one
/// standalone [`PolyMeshRaw`] per zone, in zone (file) order, plus the
/// interface pairs. The rules, all checked before anything is written:
///
/// * every zone name passes [`check_patch_name`]; no zone is empty; every
///   label is a cell of the mesh; every cell is in EXACTLY one zone;
/// * no cyclic patch in the parent (SPEC-LIT §31.1 declares a pair once and
///   matches it then, and the k-th-to-k-th ordering of
///   `mesh::geometry::cyclic_pairing` is a property of the declared patch -
///   a split renumbers and re-partitions the faces, so neither survives
///   it; split before the cyclic is declared);
/// * a zone's cells are renumbered by rank in ASCENDING global order (R5),
///   which keeps `owner < neighbour` and the parent's order;
/// * an internal face with both cells in one zone stays internal to it,
///   STABLE-sorted by `(owner, neighbour)` (R1); one across two zones
///   becomes ONE interface face on each side - the owner's zone keeps the
///   loop, the neighbour's zone gets [`reverse_face`], and the k-th face of
///   BOTH patches is the k-th such parent face in ascending order (R2).
///   Names are `<zone>_to_<other>` (R3);
/// * a boundary face goes to its owner's zone, keeping its patch; a patch
///   with no face left in a zone is DROPPED there (an empty patch would be
///   a name the case claims for nothing). Patch order is the parent's
///   survivors in parent order, then the interface patches in ascending
///   other-zone index (*DESIGN*: nothing reads two patches of one region
///   by position);
/// * each region keeps only the points it uses, renumbered by FIRST USE
///   walking its faces in output order; coordinates copied unchanged.
pub fn split_by_zones(
    raw: &PolyMeshRaw,
    zones: &[(String, Vec<Label>)],
) -> Result<(Vec<(String, PolyMeshRaw)>, Vec<SplitInterface>)> {
    let n = zones.len();
    let n_cells = raw
        .owner
        .iter()
        .chain(raw.neighbour.iter())
        .copied()
        .max()
        .map_or(0, |m| m as usize + 1);
    for (name, labels) in zones {
        check_patch_name(name, "cellZone")?;
        if labels.is_empty() {
            return Err(Error::Mesh(format!(
                "split_by_zones: zone '{name}' is empty - a zone with no \
                 cells is not a region"
            )));
        }
    }
    let mut zone_of = vec![usize::MAX; n_cells];
    for (zi, (name, labels)) in zones.iter().enumerate() {
        for &l in labels {
            if l < 0 || l as usize >= n_cells {
                return Err(Error::Mesh(format!(
                    "split_by_zones: zone '{name}' labels cell {l}, but the \
                     mesh has {n_cells} cells"
                )));
            }
            let c = l as usize;
            if zone_of[c] != usize::MAX {
                return Err(Error::Mesh(format!(
                    "split_by_zones: cell {c} is in two zones, '{}' and \
                     '{name}'",
                    zones[zone_of[c]].0
                )));
            }
            zone_of[c] = zi;
        }
    }
    let unassigned = zone_of.iter().filter(|&&z| z == usize::MAX).count();
    if unassigned > 0 {
        let first = zone_of.iter().position(|&z| z == usize::MAX).unwrap();
        return Err(Error::Mesh(format!(
            "split_by_zones: {unassigned} cells are in no zone, starting \
             with cell {first} - `ofgpu-regions split` needs a cellZones \
             entry for every cell"
        )));
    }
    for p in &raw.patches {
        if p.kind == PatchKind::Cyclic {
            return Err(Error::Mesh(format!(
                "split_by_zones: the parent mesh carries cyclic patch '{}'. \
                 A cyclic pair is declared once and matched then (SPEC-LIT \
                 31.1: nearest transformed centroid, a bijection, |Sf| equal \
                 with Sf opposed), and its faces couple by position within \
                 the patch (mesh::geometry::cyclic_pairing); a split \
                 renumbers and re-partitions them, so neither survives it - \
                 split before the cyclic is declared",
                p.name
            )));
        }
    }
    let mut order: Vec<Vec<u32>> = vec![Vec::new(); n];
    for (zi, (_, labels)) in zones.iter().enumerate() {
        let mut v: Vec<u32> = labels.iter().map(|&l| l as u32).collect();
        v.sort_unstable();
        order[zi] = v;
    }
    let mut local = vec![0u32; n_cells];
    for (_zi, cells) in order.iter().enumerate() {
        for (rank, &c) in cells.iter().enumerate() {
            local[c as usize] = rank as u32;
        }
    }
    let n_if = raw.neighbour.len();
    let mut z_internal: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut i_faces: Vec<Vec<usize>> = vec![Vec::new(); n * n];
    for f in 0..n_if {
        let zo = zone_of[raw.owner[f] as usize];
        let zn = zone_of[raw.neighbour[f] as usize];
        if zo == zn {
            z_internal[zo].push(f);
        } else {
            i_faces[zo * n + zn].push(f);
            i_faces[zn * n + zo].push(f);
        }
    }
    for zl in z_internal.iter_mut() {
        zl.sort_by_key(|&f| (local[raw.owner[f] as usize], local[raw.neighbour[f] as usize]));
    }
    let mut bnd: Vec<Vec<Vec<usize>>> = vec![vec![Vec::new(); n]; raw.patches.len()];
    for (pi, p) in raw.patches.iter().enumerate() {
        for k in 0..p.size {
            let f = n_if + p.start + k;
            bnd[pi][zone_of[raw.owner[f] as usize]].push(f);
        }
    }
    let mut split: Vec<SplitInterface> = Vec::new();
    for zo in 0..n {
        for zn in (zo + 1)..n {
            let fs = &i_faces[zo * n + zn];
            if !fs.is_empty() {
                split.push(SplitInterface {
                    region_a: zo,
                    patch_a: format!("{}_to_{}", zones[zo].0, zones[zn].0),
                    region_b: zn,
                    patch_b: format!("{}_to_{}", zones[zn].0, zones[zo].0),
                    n_faces: fs.len(),
                });
            }
        }
    }
    let mk_patch = |name: String, tn: &str, kind: PatchKind, start: usize, size: usize| {
        PatchInfo { name, type_name: tn.to_string(), kind, start, size, nbr_patch: None }
    };
    let mut out: Vec<(String, PolyMeshRaw)> = Vec::with_capacity(n);
    for zi in 0..n {
        let mut faces: Vec<Vec<Label>> = Vec::new();
        let mut owner: Vec<Label> = Vec::new();
        let mut neighbour: Vec<Label> = Vec::new();
        let mut patches: Vec<PatchInfo> = Vec::new();
        for &f in &z_internal[zi] {
            faces.push(raw.faces[f].clone());
            owner.push(local[raw.owner[f] as usize] as Label);
            neighbour.push(local[raw.neighbour[f] as usize] as Label);
        }
        for (pi, p) in raw.patches.iter().enumerate() {
            let fs = &bnd[pi][zi];
            if fs.is_empty() {
                continue;
            }
            let start = faces.len() - z_internal[zi].len();
            for &f in fs {
                faces.push(raw.faces[f].clone());
                owner.push(local[raw.owner[f] as usize] as Label);
            }
            patches.push(mk_patch(p.name.clone(), &p.type_name, p.kind, start, fs.len()));
        }
        for zj in 0..n {
            if zj == zi {
                continue;
            }
            let fs = &i_faces[zi * n + zj];
            if fs.is_empty() {
                continue;
            }
            let start = faces.len() - z_internal[zi].len();
            for &f in fs {
                // zi is the owner's side or the neighbour's side of this
                // parent face, per face: the owner's side keeps the loop
                // (outward from it already), the neighbour's side reverses
                // it and takes the neighbour cell (R2: ordered identically).
                let o = raw.owner[f] as usize;
                let nb = raw.neighbour[f] as usize;
                if zone_of[o] == zi {
                    faces.push(raw.faces[f].clone());
                    owner.push(local[o] as Label);
                } else {
                    faces.push(reverse_face(&raw.faces[f]));
                    owner.push(local[nb] as Label);
                }
            }
            let name = format!("{}_to_{}", zones[zi].0, zones[zj].0);
            patches.push(mk_patch(name, "patch", PatchKind::Generic, start, fs.len()));
        }
        let mut pmap = vec![-1 as Label; raw.points.len()];
        let mut points: Vec<Vec3> = Vec::new();
        for face in faces.iter_mut() {
            for v in face.iter_mut() {
                let old = *v as usize;
                if pmap[old] < 0 {
                    pmap[old] = points.len() as Label;
                    points.push(raw.points[old]);
                }
                *v = pmap[old];
            }
        }
        out.push((
            zones[zi].0.clone(),
            PolyMeshRaw { points, faces, owner, neighbour, patches },
        ));
    }
    Ok((out, split))
}

// ==========================================================================
//  Writing the layout
// ==========================================================================

/// Write a layout as §C's tree: `<out>/<region>/polyMesh/...` through
/// [`write_poly_mesh_raw`], and `<out>/regions.json` in field order. An
/// existing `<out>/regions.json` is refused here, not only in the driver.
pub fn write_layout(
    out_dir: &Path,
    regions: &[(String, PolyMeshRaw)],
    kinds: &[RegionKind],
    interfaces: &[SplitInterface],
    source: Option<serde_json::Value>,
) -> Result<RegionsManifest> {
    if kinds.len() != regions.len() {
        return Err(Error::Config(format!(
            "write_layout: {} regions but {} kinds",
            regions.len(),
            kinds.len()
        )));
    }
    let manifest_path = out_dir.join("regions.json");
    if manifest_path.exists() {
        return Err(Error::Config(format!(
            "{} already exists - refusing to overwrite an existing layout; \
             write to a fresh directory",
            manifest_path.display()
        )));
    }
    let mut entries = Vec::with_capacity(regions.len());
    for (i, (name, raw)) in regions.iter().enumerate() {
        check_patch_name(name, "region")?;
        write_poly_mesh_raw(&out_dir.join(name).join("polyMesh"), raw)?;
        entries.push(RegionEntry {
            name: name.clone(),
            kind: kind_str(kinds[i]).to_string(),
            poly_mesh: format!("{name}/polyMesh"),
            material: None,
        });
    }
    let ifaces = interfaces
        .iter()
        .map(|si| InterfaceEntry {
            regions: [regions[si.region_a].0.clone(), regions[si.region_b].0.clone()],
            patches: [si.patch_a.clone(), si.patch_b.clone()],
            faces: Some(si.n_faces),
            tolerance: default_tolerance(),
        })
        .collect();
    let manifest = RegionsManifest {
        version: 1,
        units: metres(),
        regions: entries,
        interfaces: ifaces,
        source,
    };
    let text = serde_json::to_string_pretty(&manifest)
        .map_err(|e| Error::Config(format!("regions.json: {e}")))
        .map(|mut t| {
            t.push('\n');
            t
        })?;
    fs::write(&manifest_path, text).path(&manifest_path)?;
    Ok(manifest)
}

#[cfg(test)]
mod tests;
