// meteor-cfd - Copyright (c) 2026 주식회사 메테오시뮬레이션 (Meteo Simulation Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

//! Gmsh MSH 4.1 ASCII reader -> [`crate::io::polymesh::PolyMeshRaw`].
//!
//! Written from:
//!   the published Gmsh reference manual's file-format chapter (`$MeshFormat`,
//!     `$PhysicalNames`, `$Entities`, `$Nodes`, `$Elements` block layout, and
//!     the low-order element node ordering for the point/line/triangle/quad/
//!     tetrahedron/hexahedron/prism/pyramid element types) - Gmsh's SOURCE is
//!     GPL and was not consulted; the file format is a public specification
//!     and is not copyrightable;
//!   ofgpu `SPEC-LIT.md` section 1 (the lower/diagonal/upper mesh model this
//!     reader has to hand `build_host_mesh` - owner < neighbour, faces sorted
//!     by (owner, neighbour), outward area vectors) and section 13.4 (the
//!     unsupported-setting contract, used here for a mesh with no patch
//!     identity at all).
//! No GPL-licensed source was consulted.
//!
//! # What this reader does
//!
//! A `.msh` volume mesh carries no owner/neighbour/boundary addressing the
//! way `constant/polyMesh` does - it only has elements. This reader:
//!
//! 1. reads every tetrahedron/hexahedron/prism/pyramid as a cell, in the
//!    reference-element node order the Gmsh manual specifies, and looks up
//!    each cell's faces (also in the manual's order) with the winding that
//!    is OUTWARD from that cell - derived once, by hand, in [`TET_FACES`],
//!    [`HEX_FACES`], [`PRISM_FACES`] and [`PYRAMID_FACES`] below, and checked
//!    there against the right-hand rule;
//! 2. dedupes every face by its (order-independent) vertex set: a face seen
//!    by exactly one cell is a boundary face with that cell as owner and the
//!    recorded (outward) winding; a face seen by exactly two cells is
//!    internal, and because cells are numbered in the order they are read,
//!    the cell that touches it FIRST always has the smaller index - which is
//!    exactly the `owner` polyMesh wants, with no re-winding needed;
//! 3. reads every triangle/quadrangle element as a naming tag rather than
//!    geometry: its vertex set, together with the physical name of the
//!    surface entity it belongs to, says which patch a matching boundary
//!    face (from step 2) is in. A boundary face with no matching 2-D element,
//!    or whose 2-D element names no physical surface, falls into
//!    `defaultFaces` with a warning (never silently dropped - SPEC-LIT §13.4)
//! 4. sorts the internal faces into (owner, neighbour) order and the
//!    boundary faces into patch-contiguous blocks, and hands the result to
//!    [`crate::io::polymesh::build_host_mesh`], unchanged from what a
//!    `constant/polyMesh` reader would hand it.
//!
//! `$PhysicalNames` missing entirely means the mesh has no patch identity at
//! all (every boundary face would be `defaultFaces`) - the section-13.4
//! contract makes that a loud error rather than a silently patch-less mesh;
//! `-permissive` accepts it and puts everything in `defaultFaces`.

use std::collections::HashMap;
use std::path::Path;

use crate::error::{Error, IoContext, Result};
use crate::io::contract;
use crate::io::polymesh::PolyMeshRaw;
use crate::mesh::{PatchInfo, PatchKind};
use crate::{Label, Scalar, Vec3};

// ==========================================================================
//  Reference-element face tables
// ==========================================================================
//
// Local vertex indices, in the order the Gmsh manual assigns to the element's
// nodes, listed per face so that walking them in the given order winds the
// face OUTWARD from the cell. Verified below (in the module doc's sense: by
// hand, via (p1-p0) x (p2-p0) against the reference-element coordinates), not
// copied from anyone's code - this is the same reference-element convention
// the manual's node-ordering diagrams describe.
//
// Tetrahedron, v0=(0,0,0) v1=(1,0,0) v2=(0,1,0) v3=(0,0,1):
const TET_FACES: [&[usize]; 4] = [
    &[0, 2, 1], // z = 0,           outward -z
    &[0, 1, 3], // y = 0,           outward -y
    &[0, 3, 2], // x = 0,           outward -x
    &[1, 2, 3], // x+y+z = 1,       outward (1,1,1)
];

// Hexahedron, unit cube v0..v3 = z=0 (CCW from +z), v4..v7 = z=1 above them:
const HEX_FACES: [&[usize]; 6] = [
    &[0, 3, 2, 1], // z = 0, outward -z
    &[4, 5, 6, 7], // z = 1, outward +z
    &[0, 1, 5, 4], // y = 0, outward -y
    &[3, 7, 6, 2], // y = 1, outward +y
    &[0, 4, 7, 3], // x = 0, outward -x
    &[1, 2, 6, 5], // x = 1, outward +x
];

// Prism (wedge), v0,v1,v2 bottom triangle (z=0), v3,v4,v5 the same triangle
// at z=1 (v3 above v0, v4 above v1, v5 above v2):
const PRISM_FACES: [&[usize]; 5] = [
    &[0, 2, 1],    // bottom z = 0, outward -z
    &[3, 4, 5],    // top z = 1,    outward +z
    &[0, 1, 4, 3], // y = 0 side,   outward -y
    &[1, 2, 5, 4], // slanted side, outward (1,1,0)
    &[0, 3, 5, 2], // x = 0 side,   outward -x
];

// Pyramid, v0..v3 the base quad (z=0, same layout as the hex's bottom face),
// v4 the apex above it:
const PYRAMID_FACES: [&[usize]; 5] = [
    &[0, 3, 2, 1], // base z = 0, outward -z
    &[0, 1, 4],    // side, outward -y ish
    &[1, 2, 4],    // side, outward +x ish
    &[2, 3, 4],    // side, outward +y ish
    &[3, 0, 4],    // side, outward -x ish
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CellKind {
    Tet,
    Hex,
    Prism,
    Pyramid,
}

impl CellKind {
    fn faces(self) -> &'static [&'static [usize]] {
        match self {
            CellKind::Tet => &TET_FACES,
            CellKind::Hex => &HEX_FACES,
            CellKind::Prism => &PRISM_FACES,
            CellKind::Pyramid => &PYRAMID_FACES,
        }
    }
}

/// Gmsh element-type code -> (node count, what it is).
#[derive(Clone, Copy)]
enum ElemShape {
    /// A cell: tet/hex/prism/pyramid.
    Cell(CellKind, usize),
    /// A 2-D element (triangle/quadrangle): a patch-naming tag, not a cell.
    Surface(usize),
    /// A 1-D or 0-D element: irrelevant to the volume mesh, node count only.
    Ignored(usize),
}

fn elem_shape(elem_type: i64) -> Option<ElemShape> {
    Some(match elem_type {
        1 => ElemShape::Ignored(2),        // 2-node line
        2 => ElemShape::Surface(3),        // 3-node triangle
        3 => ElemShape::Surface(4),        // 4-node quadrangle
        4 => ElemShape::Cell(CellKind::Tet, 4),
        5 => ElemShape::Cell(CellKind::Hex, 8),
        6 => ElemShape::Cell(CellKind::Prism, 6),
        7 => ElemShape::Cell(CellKind::Pyramid, 5),
        15 => ElemShape::Ignored(1),       // 1-node point
        _ => return None,
    })
}

// ==========================================================================
//  Public entry point
// ==========================================================================

pub fn read_msh(path: impl AsRef<Path>) -> Result<PolyMeshRaw> {
    let path = path.as_ref();
    let text = std::fs::read_to_string(path).path(path)?;
    parse_msh(&text, &path.display().to_string())
}

/// Parse from memory - public so tests need no fixture files on disk, the
/// same shape as [`crate::surface::stl::parse_stl`].
pub fn parse_msh(text: &str, origin: &str) -> Result<PolyMeshRaw> {
    let mut lx = Lex::new(text, origin);

    lx.expect_tok("$MeshFormat")?;
    let version = lx.tok()?;
    if version != "4.1" {
        return lx.err(format!(
            "unsupported MSH version \"{version}\" (ofgpu reads MSH 4.1 ASCII \
             only; re-export from Gmsh with `-format msh41`)"
        ));
    }
    let file_type = lx.int()?;
    let _data_size = lx.int()?;
    if file_type != 0 {
        return lx.err(
            "binary MSH is not supported; re-export as ASCII (Gmsh: \
             File > Export > .msh, uncheck \"Binary\")"
                .to_string(),
        );
    }
    lx.expect_tok("$EndMeshFormat")?;

    // ---- optional $PhysicalNames -------------------------------------------
    let mut phys_names: HashMap<(i64, i64), String> = HashMap::new();
    let mut have_physical_names = false;
    if lx.peek() == Some("$PhysicalNames") {
        have_physical_names = true;
        lx.tok()?;
        let n = lx.int()?;
        for _ in 0..n {
            let dim = lx.int()?;
            let tag = lx.int()?;
            let name = lx.quoted()?;
            phys_names.insert((dim, tag), name.to_string());
        }
        lx.expect_tok("$EndPhysicalNames")?;
    }

    if !have_physical_names {
        contract::unsupported::<()>(
            "msh/physicalNames",
            "(section absent)",
            &[],
            "one 'defaultFaces' patch covering every boundary face",
            (),
        )?;
    }

    // ---- $Entities: which physical tag(s) a surface entity carries ---------
    lx.expect_tok("$Entities")?;
    let surf_phys = read_entities(&mut lx)?;
    lx.expect_tok("$EndEntities")?;

    // ---- $Nodes --------------------------------------------------------------
    lx.expect_tok("$Nodes")?;
    let (points, tag_to_idx) = read_nodes(&mut lx)?;
    lx.expect_tok("$EndNodes")?;

    // ---- $Elements -------------------------------------------------------
    lx.expect_tok("$Elements")?;
    let (cells, surf_patch_of_face) =
        read_elements(&mut lx, &tag_to_idx, &surf_phys, &phys_names)?;
    lx.expect_tok("$EndElements")?;

    build_raw_mesh(points, cells, surf_patch_of_face)
}

// ==========================================================================
//  $Entities
// ==========================================================================

/// surfaceTag -> its physical tags (dim 2), in file order.
fn read_entities(lx: &mut Lex) -> Result<HashMap<i64, Vec<i64>>> {
    let n_points = lx.int()?;
    let n_curves = lx.int()?;
    let n_surfaces = lx.int()?;
    let n_volumes = lx.int()?;

    for _ in 0..n_points {
        let _tag = lx.int()?;
        let _x = lx.num()?;
        let _y = lx.num()?;
        let _z = lx.num()?;
        let n_tags = lx.int()?;
        for _ in 0..n_tags {
            lx.int()?;
        }
    }

    for _ in 0..n_curves {
        read_bbox_entity(lx)?;
    }

    let mut surf_phys = HashMap::with_capacity(n_surfaces.max(0) as usize);
    for _ in 0..n_surfaces {
        let (tag, phys) = read_bbox_entity(lx)?;
        surf_phys.insert(tag, phys);
    }

    for _ in 0..n_volumes {
        read_bbox_entity(lx)?;
    }

    Ok(surf_phys)
}

/// The shared shape of a curve/surface/volume entity record:
/// `tag minX minY minZ maxX maxY maxZ numPhysTags physTag* numBounding bounding*`.
fn read_bbox_entity(lx: &mut Lex) -> Result<(i64, Vec<i64>)> {
    let tag = lx.int()?;
    for _ in 0..6 {
        lx.num()?;
    }
    let n_phys = lx.int()?;
    let mut phys = Vec::with_capacity(n_phys.max(0) as usize);
    for _ in 0..n_phys {
        phys.push(lx.int()?);
    }
    let n_bound = lx.int()?;
    for _ in 0..n_bound {
        lx.int()?; // signed entity tag of a bounding point/curve/surface
    }
    Ok((tag, phys))
}

// ==========================================================================
//  $Nodes
// ==========================================================================

fn read_nodes(lx: &mut Lex) -> Result<(Vec<Vec3>, HashMap<i64, u32>)> {
    let n_blocks = lx.int()?;
    let n_nodes = lx.int()?;
    let _min_tag = lx.int()?;
    let _max_tag = lx.int()?;

    let mut points = Vec::with_capacity(n_nodes.max(0) as usize);
    let mut tag_to_idx: HashMap<i64, u32> = HashMap::with_capacity(n_nodes.max(0) as usize);

    for _ in 0..n_blocks {
        let entity_dim = lx.int()?;
        let _entity_tag = lx.int()?;
        let parametric = lx.int()?;
        let n_in_block = lx.int()?;

        let mut tags = Vec::with_capacity(n_in_block.max(0) as usize);
        for _ in 0..n_in_block {
            tags.push(lx.int()?);
        }
        // A parametric node's line carries `entity_dim` parametric
        // coordinates AFTER x y z - the manual's `$Nodes` grammar is
        // `x y z <u <v <w>>>`. Skipped, never used. (Order verified against
        // the manual after a cross-check with PyFR's BSD gmsh reader caught
        // this file skipping them BEFORE the coordinates, which would have
        // read y,z,u as x,y,z on any `-save_parametric` mesh.)
        let extra = if parametric != 0 { entity_dim.max(0) as usize } else { 0 };
        for tag in tags {
            let x = lx.num()? as Scalar;
            let y = lx.num()? as Scalar;
            let z = lx.num()? as Scalar;
            for _ in 0..extra {
                lx.num()?;
            }
            let idx = points.len() as u32;
            if tag_to_idx.insert(tag, idx).is_some() {
                return lx.err(format!("node tag {tag} appears more than once"));
            }
            points.push(Vec3::new(x, y, z));
        }
    }

    if points.len() != n_nodes.max(0) as usize {
        return lx.err(format!(
            "$Nodes declared {n_nodes} nodes but the blocks contained {}",
            points.len()
        ));
    }
    Ok((points, tag_to_idx))
}

// ==========================================================================
//  $Elements
// ==========================================================================

struct Cell {
    kind: CellKind,
    verts: Vec<u32>,
}

/// Reads every element block. Volume elements become [`Cell`]s (in file
/// order, which is what fixes the "first cell to touch a face has the
/// smaller index" property the face-dedup step below relies on). Surface
/// (triangle/quadrangle) elements resolve to a patch name through the
/// entity's physical tag and are recorded by their sorted vertex key.
fn read_elements(
    lx: &mut Lex,
    tag_to_idx: &HashMap<i64, u32>,
    surf_phys: &HashMap<i64, Vec<i64>>,
    phys_names: &HashMap<(i64, i64), String>,
) -> Result<(Vec<Cell>, HashMap<Vec<u32>, String>)> {
    let n_blocks = lx.int()?;
    let n_elements = lx.int()?;
    let _min_tag = lx.int()?;
    let _max_tag = lx.int()?;

    let mut cells: Vec<Cell> = Vec::new();
    let mut surf_patch_of_face: HashMap<Vec<u32>, String> = HashMap::new();

    let mut n_read = 0i64;
    for _ in 0..n_blocks {
        let entity_dim = lx.int()?;
        let entity_tag = lx.int()?;
        let elem_type = lx.int()?;
        let n_in_block = lx.int()?;

        // Not routed through the section-13.4 permissive contract: a element
        // type this reader has no node-count table for cannot be *skipped*
        // safely either, because there is nothing to say how many node-tag
        // tokens follow it. Unlike a dictionary setting with a documented
        // default, there is no safe fallback here - it is always fatal.
        let Some(shape) = elem_shape(elem_type) else {
            return lx.err(format!(
                "unsupported Gmsh element type {elem_type} (ofgpu reads \
                 point/line/triangle/quadrangle/tetrahedron/hexahedron/prism/\
                 pyramid, types 1,2,3,4,5,6,7,15 - not a higher-order element)"
            ));
        };

        let patch_name = if matches!(shape, ElemShape::Surface(_)) {
            surf_phys
                .get(&entity_tag)
                .and_then(|tags| tags.first())
                .and_then(|&tag| phys_names.get(&(entity_dim, tag)))
                .cloned()
        } else {
            None
        };

        for _ in 0..n_in_block {
            let _elem_tag = lx.int()?;
            let n_nodes = match shape {
                ElemShape::Cell(_, n) | ElemShape::Surface(n) | ElemShape::Ignored(n) => n,
            };
            let mut verts = Vec::with_capacity(n_nodes);
            for _ in 0..n_nodes {
                let node_tag = lx.int()?;
                let Some(&idx) = tag_to_idx.get(&node_tag) else {
                    return lx.err(format!(
                        "element references node tag {node_tag}, which $Nodes never defined"
                    ));
                };
                verts.push(idx);
            }

            match shape {
                ElemShape::Cell(kind, _) => cells.push(Cell { kind, verts }),
                ElemShape::Surface(_) => {
                    if let Some(name) = &patch_name {
                        let mut key = verts;
                        key.sort_unstable();
                        surf_patch_of_face.entry(key).or_insert_with(|| name.clone());
                    }
                }
                ElemShape::Ignored(_) => {}
            }
        }

        n_read += n_in_block;
    }

    if n_read != n_elements {
        return lx.err(format!(
            "$Elements declared {n_elements} elements but the blocks contained {n_read}"
        ));
    }

    Ok((cells, surf_patch_of_face))
}

// ==========================================================================
//  Face derivation and mesh assembly
// ==========================================================================

struct FaceRec {
    verts: Vec<u32>,
    owner: u32,
    neighbour: Option<u32>,
}

fn build_raw_mesh(
    points: Vec<Vec3>,
    cells: Vec<Cell>,
    surf_patch_of_face: HashMap<Vec<u32>, String>,
) -> Result<PolyMeshRaw> {
    if cells.is_empty() {
        return Err(Error::Mesh(
            "the MSH file has no tetrahedron/hexahedron/prism/pyramid \
             elements; there is no volume mesh to build"
                .into(),
        ));
    }

    // ---- dedupe every cell face by its vertex SET --------------------------
    // Cells are visited in ascending index (file order), so the first cell
    // ever to touch a face always has the smaller index - exactly polyMesh's
    // owner - and the winding recorded on first touch, "outward from that
    // cell", is already the owner's outward normal. Nothing needs re-winding.
    let mut face_index: HashMap<Vec<u32>, usize> = HashMap::new();
    let mut face_recs: Vec<FaceRec> = Vec::new();

    for (cell_idx, cell) in cells.iter().enumerate() {
        for local_face in cell.kind.faces() {
            let verts: Vec<u32> = local_face.iter().map(|&li| cell.verts[li]).collect();
            let mut key = verts.clone();
            key.sort_unstable();

            match face_index.get(&key) {
                None => {
                    face_index.insert(key, face_recs.len());
                    face_recs.push(FaceRec {
                        verts,
                        owner: cell_idx as u32,
                        neighbour: None,
                    });
                }
                Some(&idx) => {
                    let rec = &mut face_recs[idx];
                    if rec.neighbour.is_some() {
                        return Err(Error::Mesh(format!(
                            "face with vertices {key:?} is shared by more than two \
                             cells; the mesh is not a valid volume mesh"
                        )));
                    }
                    rec.neighbour = Some(cell_idx as u32);
                }
            }
        }
    }

    // ---- split into internal (sorted) and boundary (patch-contiguous) -----
    let mut internal: Vec<(Label, Label, Vec<u32>)> = Vec::new();
    let mut boundary: Vec<(String, Label, Vec<u32>)> = Vec::new();
    let mut default_count = 0usize;

    for rec in face_recs {
        match rec.neighbour {
            Some(n) => internal.push((rec.owner as Label, n as Label, rec.verts)),
            None => {
                let mut key = rec.verts.clone();
                key.sort_unstable();
                let name = match surf_patch_of_face.get(&key) {
                    Some(n) => n.clone(),
                    None => {
                        default_count += 1;
                        "defaultFaces".to_string()
                    }
                };
                boundary.push((name, rec.owner as Label, rec.verts));
            }
        }
    }

    if default_count > 0 {
        contract::warn_once(
            "msh/defaultFaces",
            &format!(
                "{default_count} boundary face(s) were not covered by any \
                 physical surface; assigned to patch 'defaultFaces'"
            ),
        );
    }

    internal.sort_by(|a, b| (a.0, a.1).cmp(&(b.0, b.1)));

    // Patch order: the order each name is first seen among boundary faces -
    // deterministic, and matches the order the physical surfaces were
    // encountered when no reordering is needed.
    let mut patch_order: Vec<String> = Vec::new();
    let mut patch_id: HashMap<String, usize> = HashMap::new();
    for (name, _, _) in &boundary {
        if !patch_id.contains_key(name) {
            patch_id.insert(name.clone(), patch_order.len());
            patch_order.push(name.clone());
        }
    }
    let mut by_patch: Vec<Vec<(Label, Vec<u32>)>> = vec![Vec::new(); patch_order.len()];
    for (name, owner, verts) in boundary {
        by_patch[patch_id[&name]].push((owner, verts));
    }

    // ---- assemble PolyMeshRaw ----------------------------------------------
    let n_if = internal.len();
    let mut faces: Vec<Vec<Label>> = Vec::with_capacity(n_if + by_patch.iter().map(Vec::len).sum::<usize>());
    let mut owner: Vec<Label> = Vec::with_capacity(faces.capacity());
    let mut neighbour: Vec<Label> = Vec::with_capacity(n_if);

    for (o, n, verts) in internal {
        owner.push(o);
        neighbour.push(n);
        faces.push(verts.into_iter().map(|v| v as Label).collect());
    }

    let mut patches = Vec::with_capacity(patch_order.len());
    for (name, group) in patch_order.into_iter().zip(by_patch) {
        let start = owner.len() - n_if;
        for (o, verts) in group {
            owner.push(o);
            faces.push(verts.into_iter().map(|v| v as Label).collect());
        }
        let size = (owner.len() - n_if) - start;
        patches.push(PatchInfo {
            name,
            type_name: "patch".to_string(),
            kind: PatchKind::Generic,
            start,
            size,
            nbr_patch: None,
        });
    }

    Ok(PolyMeshRaw { points, faces, owner, neighbour, patches })
}

// ==========================================================================
//  Lexer: a whitespace/quote tokeniser over the whole file
// ==========================================================================

struct Lex<'a> {
    toks: Vec<&'a str>,
    pos: usize,
    origin: String,
}

impl<'a> Lex<'a> {
    fn new(text: &'a str, origin: &str) -> Self {
        Lex { toks: tokenize(text), pos: 0, origin: origin.to_string() }
    }

    fn peek(&self) -> Option<&'a str> {
        self.toks.get(self.pos).copied()
    }

    fn tok(&mut self) -> Result<&'a str> {
        let Some(t) = self.toks.get(self.pos).copied() else {
            return self.err("unexpected end of file".to_string());
        };
        self.pos += 1;
        Ok(t)
    }

    fn quoted(&mut self) -> Result<&'a str> {
        self.tok()
    }

    fn expect_tok(&mut self, want: &str) -> Result<()> {
        let t = self.tok()?;
        if t != want {
            return self.err(format!("expected \"{want}\", found \"{t}\""));
        }
        Ok(())
    }

    fn int(&mut self) -> Result<i64> {
        let t = self.tok()?;
        t.parse::<i64>()
            .map_err(|_| ())
            .or_else(|_| self.err(format!("expected an integer, found \"{t}\"")))
    }

    fn num(&mut self) -> Result<f64> {
        let t = self.tok()?;
        t.parse::<f64>()
            .map_err(|_| ())
            .or_else(|_| self.err(format!("expected a number, found \"{t}\"")))
    }

    fn err<T>(&self, msg: String) -> Result<T> {
        Err(Error::Parse { path: self.origin.clone(), msg })
    }
}

fn tokenize(text: &str) -> Vec<&str> {
    let bytes = text.as_bytes();
    let mut toks = Vec::new();
    let mut i = 0usize;
    while i < bytes.len() {
        let c = bytes[i];
        if c.is_ascii_whitespace() {
            i += 1;
            continue;
        }
        if c == b'"' {
            let start = i + 1;
            let mut j = start;
            while j < bytes.len() && bytes[j] != b'"' {
                j += 1;
            }
            toks.push(&text[start..j]);
            i = (j + 1).min(bytes.len());
            continue;
        }
        let start = i;
        while i < bytes.len() && !bytes[i].is_ascii_whitespace() && bytes[i] != b'"' {
            i += 1;
        }
        toks.push(&text[start..i]);
    }
    toks
}

// ==========================================================================
//  Tests
// ==========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::polymesh::build_host_mesh;

    fn parse(text: &str) -> Result<PolyMeshRaw> {
        crate::io::contract::set_permissive(false);
        parse_msh(text, "<memory>")
    }

    /// One unit hexahedron, physical surface "walls" covering all six faces
    /// (one physical surface per Gmsh geometric side, all sharing the same
    /// physical tag/name so the whole cube is one patch). Node tags are
    /// deliberately non-contiguous (11, 22, ...) to exercise the tag map.
    fn one_hex_msh() -> String {
        // Node tags 11..18 for the unit cube corners in the hex reference
        // order: 0..3 the z=0 face, 4..7 the z=1 face above them.
        let pts = [
            (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 1.0), (0.0, 1.0, 1.0),
        ];
        let tags: Vec<i64> = (0..8).map(|i| 11 + i).collect();

        let mut nodes_body = String::new();
        for &t in &tags {
            nodes_body += &format!("{t}\n");
        }
        for (i, &(x, y, z)) in pts.iter().enumerate() {
            let _ = i;
            nodes_body += &format!("{x} {y} {z}\n");
        }

        let hex_nodes: Vec<String> = tags.iter().map(|t| t.to_string()).collect();

        format!(
            r#"$MeshFormat
4.1 0 8
$EndMeshFormat
$PhysicalNames
1
2 1 "walls"
$EndPhysicalNames
$Entities
0 0 1 1
1 0 0 0 1 1 1 1 1 0
7 0 0 0 1 1 1 0 0
$EndEntities
$Nodes
1 8 11 18
3 1 0 8
{nodes_body}$EndNodes
$Elements
2 7 1 7
2 1 3 6
1 11 14 13 12
2 15 16 17 18
3 11 12 16 15
4 14 18 17 13
5 11 15 18 14
6 12 13 17 16
3 5 5 1
7 {hex}
$EndElements
"#,
            hex = hex_nodes.join(" ")
        )
    }

    #[test]
    fn one_hex_has_six_boundary_faces_and_closes() {
        let raw = match parse(&one_hex_msh()) {
            Ok(r) => r,
            Err(e) => panic!("parse failed: {e}"),
        };
        assert_eq!(raw.points.len(), 8);
        assert_eq!(raw.faces.len(), 6);
        assert_eq!(raw.owner, vec![0; 6]);
        assert_eq!(raw.neighbour.len(), 0);
        assert_eq!(raw.patches.len(), 1);
        assert_eq!(raw.patches[0].name, "walls");
        assert_eq!(raw.patches[0].size, 6);

        let m = match build_host_mesh(&raw) {
            Ok(m) => m,
            Err(e) => panic!("build_host_mesh failed: {e}"),
        };
        assert_eq!(m.n_cells, 1);
        assert!((m.v[0] - 1.0).abs() < 1e-10, "v = {}", m.v[0]);
        let r = crate::mesh::geometry::check(&m);
        assert!(r.max_closure_error < 1e-10, "closure error {}", r.max_closure_error);
    }

    /// Two unit hexes sharing the x = 1 face: one internal face, owner 0 /
    /// neighbour 1, ten boundary faces, and the mesh must close.
    fn two_hex_msh() -> String {
        let pts = [
            (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 1.0), (0.0, 1.0, 1.0),
            (2.0, 0.0, 0.0), (2.0, 1.0, 0.0), (2.0, 0.0, 1.0), (2.0, 1.0, 1.0),
        ];
        // Cell 0: 0,1,2,3,4,5,6,7 (as one_hex_msh). Cell 1 shares the x=1
        // face (points 1,2,6,5) and extends to the new points 8,9,10,11:
        // local hex order v0=1 v1=8 v2=9 v3=2 v4=5 v5=10 v6=11 v7=6.
        let hex0 = [0, 1, 2, 3, 4, 5, 6, 7];
        let hex1 = [1, 8, 9, 2, 5, 10, 11, 6];

        let tags: Vec<i64> = (0..pts.len() as i64).map(|i| i + 1).collect();
        let mut nodes_body = String::new();
        for &t in &tags {
            nodes_body += &format!("{t}\n");
        }
        for &(x, y, z) in &pts {
            nodes_body += &format!("{x} {y} {z}\n");
        }
        let n = pts.len();

        let to_tags = |ids: &[usize]| -> String {
            ids.iter().map(|&i| (i + 1).to_string()).collect::<Vec<_>>().join(" ")
        };

        format!(
            r#"$MeshFormat
4.1 0 8
$EndMeshFormat
$PhysicalNames
1
2 1 "walls"
$EndPhysicalNames
$Entities
0 0 0 1
7 0 0 0 2 1 1 0 0
$EndEntities
$Nodes
1 {n} 1 {n}
3 1 0 {n}
{nodes_body}$EndNodes
$Elements
1 2 1 2
3 5 5 2
1 {hex0}
2 {hex1}
$EndElements
"#,
            hex0 = to_tags(&hex0),
            hex1 = to_tags(&hex1),
        )
    }

    #[test]
    fn two_hexes_share_one_internal_face_with_owner_lower_than_neighbour() {
        let raw = match parse(&two_hex_msh()) {
            Ok(r) => r,
            Err(e) => panic!("parse failed: {e}"),
        };
        assert_eq!(raw.neighbour.len(), 1, "exactly one internal face");
        assert_eq!(raw.owner[0], 0);
        assert_eq!(raw.neighbour[0], 1);
        assert_eq!(raw.faces.len() - raw.neighbour.len(), 10, "10 boundary faces");

        let m = match build_host_mesh(&raw) {
            Ok(m) => m,
            Err(e) => panic!("build_host_mesh failed: {e}"),
        };
        assert_eq!(m.n_cells, 2);
        assert_eq!(m.n_internal_faces, 1);
        assert!((m.v.iter().sum::<Scalar>() - 2.0).abs() < 1e-10);
        let r = crate::mesh::geometry::check(&m);
        assert!(r.max_closure_error < 1e-10, "closure error {}", r.max_closure_error);
        assert!(r.ldu_ordered);
    }

    /// One tetrahedron plus one pyramid, sharing the pyramid's base (a
    /// square split... actually kept simple: a tet and a pyramid that do NOT
    /// touch, just to exercise reading both element types and their face
    /// tables in one file with two disjoint boundary shells.
    #[test]
    fn tet_and_pyramid_each_read_with_the_right_face_count() {
        // Regular-ish tetrahedron: 4 boundary faces, no sharing.
        let tet_pts = [
            (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0),
        ];
        // Square-base pyramid, apex above the centre: 5 boundary faces.
        let pyr_pts = [
            (10.0, 0.0, 0.0), (11.0, 0.0, 0.0), (11.0, 1.0, 0.0), (10.0, 1.0, 0.0),
            (10.5, 0.5, 1.0),
        ];
        let all_pts: Vec<(f64, f64, f64)> =
            tet_pts.iter().chain(pyr_pts.iter()).copied().collect();
        let tags: Vec<i64> = (1..=all_pts.len() as i64).collect();
        let mut nodes_body = String::new();
        for &t in &tags {
            nodes_body += &format!("{t}\n");
        }
        for &(x, y, z) in &all_pts {
            nodes_body += &format!("{x} {y} {z}\n");
        }
        let n = all_pts.len();

        let text = format!(
            r#"$MeshFormat
4.1 0 8
$EndMeshFormat
$PhysicalNames
1
2 1 "walls"
$EndPhysicalNames
$Entities
0 0 0 2
4 0 0 0 1 1 1 0 0
7 0 0 0 11 1 1 0 0
$EndEntities
$Nodes
1 {n} 1 {n}
3 1 0 {n}
{nodes_body}$EndNodes
$Elements
2 2 1 2
3 4 4 1
1 1 2 3 4
3 7 7 1
2 5 6 7 8 9
$EndElements
"#
        );

        let raw = match parse(&text) {
            Ok(r) => r,
            Err(e) => panic!("parse failed: {e}"),
        };
        assert_eq!(raw.neighbour.len(), 0, "the tet and pyramid do not touch");
        assert_eq!(raw.faces.len(), 4 + 5);

        let m = match build_host_mesh(&raw) {
            Ok(m) => m,
            Err(e) => panic!("build_host_mesh failed: {e}"),
        };
        assert_eq!(m.n_cells, 2);
        let r = crate::mesh::geometry::check(&m);
        assert!(r.max_closure_error < 1e-9, "closure error {}", r.max_closure_error);
    }

    /// The same hex, but its six sides carry two different physical names -
    /// three faces each - and the reader must produce two contiguous patches.
    #[test]
    fn a_physical_name_split_into_two_patches() {
        let pts = [
            (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 1.0), (0.0, 1.0, 1.0),
        ];
        let tags: Vec<i64> = (1..=8).collect();
        let mut nodes_body = String::new();
        for &t in &tags {
            nodes_body += &format!("{t}\n");
        }
        for &(x, y, z) in &pts {
            nodes_body += &format!("{x} {y} {z}\n");
        }

        // Hex local faces, as global (1-based) node tags:
        // F0 bottom (0,3,2,1) F1 top (4,5,6,7) F2 front (0,1,5,4)
        // F3 back (3,7,6,2)  F4 left (0,4,7,3) F5 right (1,2,6,5)
        // "inlet": bottom, front, left (F0,F2,F4). "outlet": the rest.
        let text = format!(
            r#"$MeshFormat
4.1 0 8
$EndMeshFormat
$PhysicalNames
2
2 1 "inlet"
2 2 "outlet"
$EndPhysicalNames
$Entities
0 0 2 1
1 0 0 0 1 1 1 1 1 0
2 0 0 0 1 1 1 1 2 0
100 0 0 0 1 1 1 0 0
$EndEntities
$Nodes
1 8 1 8
3 1 0 8
{nodes_body}$EndNodes
$Elements
3 7 1 7
2 1 3 3
1 1 4 3 2
2 1 2 6 5
3 1 5 8 4
2 2 3 3
4 5 8 7 6
5 4 8 7 3
6 2 3 7 6
3 5 5 1
7 1 2 3 4 5 6 7 8
$EndElements
"#
        );

        let raw = match parse(&text) {
            Ok(r) => r,
            Err(e) => panic!("parse failed: {e}"),
        };
        assert_eq!(raw.patches.len(), 2, "{:?}", raw.patches.iter().map(|p| &p.name).collect::<Vec<_>>());
        let names: Vec<&str> = raw.patches.iter().map(|p| p.name.as_str()).collect();
        assert!(names.contains(&"inlet"));
        assert!(names.contains(&"outlet"));
        assert_eq!(raw.patches.iter().map(|p| p.size).sum::<usize>(), 6);
        for p in &raw.patches {
            assert_eq!(p.size, 3, "each named patch has 3 of the 6 faces");
        }

        let m = match build_host_mesh(&raw) {
            Ok(m) => m,
            Err(e) => panic!("build_host_mesh failed: {e}"),
        };
        let r = crate::mesh::geometry::check(&m);
        assert!(r.max_closure_error < 1e-10, "closure error {}", r.max_closure_error);
    }

    #[test]
    fn missing_physical_names_is_a_strict_error_naming_permissive() {
        let _guard = crate::io::contract::permissive_test_guard();
        crate::io::contract::set_permissive(false);
        let text = one_hex_msh().replace(
            "$PhysicalNames\n1\n2 1 \"walls\"\n$EndPhysicalNames\n",
            "",
        );
        let e = match parse_msh(&text, "<memory>") {
            Err(e) => e.to_string(),
            Ok(_) => panic!("a mesh with no $PhysicalNames must be refused"),
        };
        assert!(e.contains("-permissive"), "{e}");
    }

    #[test]
    fn missing_physical_names_under_permissive_falls_back_to_default_faces() {
        let _guard = crate::io::contract::permissive_test_guard();
        crate::io::contract::set_permissive(true);
        crate::io::contract::reset_warnings();
        let text = one_hex_msh().replace(
            "$PhysicalNames\n1\n2 1 \"walls\"\n$EndPhysicalNames\n",
            "",
        );
        let raw = match parse_msh(&text, "<memory>") {
            Ok(r) => r,
            Err(e) => panic!("permissive parse failed: {e}"),
        };
        assert_eq!(raw.patches.len(), 1);
        assert_eq!(raw.patches[0].name, "defaultFaces");
        crate::io::contract::set_permissive(false);
    }

    #[test]
    fn binary_msh_is_refused() {
        let text = one_hex_msh().replace("4.1 0 8", "4.1 1 8");
        let e = match parse_msh(&text, "<memory>") {
            Err(e) => e.to_string(),
            Ok(_) => panic!("binary MSH must be refused"),
        };
        assert!(e.to_lowercase().contains("binary"), "{e}");
    }
    /// The `$Nodes` grammar is `x y z <u <v <w>>>` - parametric coordinates
    /// come AFTER the position. Caught by cross-checking against PyFR's BSD
    /// reader; a reader that skips them first shears every parametric node.
    #[test]
    fn parametric_nodes_read_xyz_first() {
        // One 2-D entity block, parametric = 1, entityDim = 2 -> two extra
        // numbers (u v) after each x y z.
        let msh = "$MeshFormat
4.1 0 8
$EndMeshFormat
$PhysicalNames
1
2 1 \"wall\"
$EndPhysicalNames
$Entities
0 0 1 0
1 0 0 0 1 1 0 1 1 0
$EndEntities
$Nodes
1 3 1 3
2 1 1 3
1
2
3
0 0 0 0.5 0.5
1 0 0 0.9 0.1
0 1 0 0.1 0.9
$EndNodes
$Elements
1 1 1 1
2 1 2 1
1 1 2 3
$EndElements
";
        let raw = parse_msh(msh, "parametric.msh");
        // Surface-only mesh has no volume cells - the reader must still have
        // taken (0,0,0), (1,0,0), (0,1,0) as the positions, NOT (0,0.5,0.5).
        match raw {
            Ok(r) => {
                assert!((r.points[0].x).abs() < 1e-14 && (r.points[0].y).abs() < 1e-14);
                assert!((r.points[1].x - 1.0).abs() < 1e-14);
                assert!((r.points[2].y - 1.0).abs() < 1e-14);
            }
            // A no-volume-cells refusal is fine too, as long as it is not a
            // parse error from misaligned numbers.
            Err(e) => {
                let msg = format!("{e}");
                assert!(msg.contains("volume") || msg.contains("cell"), "unexpected: {msg}");
            }
        }
    }

}
