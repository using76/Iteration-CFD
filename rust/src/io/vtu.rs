// meteor-cfd - Copyright (c) 2026 주식회사 메테오시뮬레이션 (Meteo Simulation Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

//! VTK XML unstructured-grid (`.vtu`) writer, plus the `.pvd` collection
//! (time series) format.
//!
//! Provenance: encoded directly from Kitware's published "VTK File Formats"
//! XML reference (the `UnstructuredGrid`/`vtkPolyhedron` sections and the
//! `AppendedData`/`header_type` binary-append convention it defines), and
//! from ParaView's documented `.pvd` "Collection" reader format. No VTK, ITK
//! or ParaView source was consulted - only the published format description.
//!
//! Cross-references into `SPEC-LIT.md`:
//! * The cell -> face CSR this writer walks (`cf_offset`/`cf_face`/`cf_own`,
//!   `bcf_offset`/`bcf_face`) is `HostMesh`'s own layout, motivated in
//!   `mesh/topology.rs` (SPEC section 1, `lduAddressing`).
//! * "A cell is just a polyhedron with more faces" (SPEC-LIT.md section
//!   24.5) is the same fact this writer leans on: `VTK_POLYHEDRON` (type 42)
//!   takes an arbitrary face list, so every mesh cell - hexahedron, merged
//!   cut cell, whatever `polyMesh` produced - is emitted the same way with no
//!   special case.
//! * Closure, `sum_f Sf = 0` exactly (SPEC-LIT.md section 24.3), is what
//!   guarantees a face's outward normal is unambiguous from its area vector
//!   alone, which is what the point-reconstruction below relies on.
//!
//! No GPL-licensed source was consulted.
//!
//! # Point reconstruction
//!
//! `HostMesh` is the finite-volume representation: it keeps face **centroids**
//! and **area vectors** (`cf`/`sf`, `b_cf`/`b_sf`), not the original polygon
//! vertices - those are consumed by `mesh/geometry.rs` and not retained (see
//! `mesh.rs`; there is no `points` field on `HostMesh` to fall back to).
//! So each face is re-synthesised as a planar quadrilateral, centred on the
//! face centroid, normal to the face's area vector, sized so the quad's area
//! equals `|Sf|`: a tangent frame `(t1, t2)` is built from the unit normal,
//! and the four corners are `centroid +/- (h*t1) +/- (h*t2)` with
//! `h = sqrt(|Sf|) / 2`. This is a faithful stand-in for area, centroid and
//! outward direction - the three quantities the solver actually carries - but
//! not for the true vertex count or shape of the original polygon; it is a
//! visualisation proxy, *DESIGN*, not a decoded mesh. Every face gets its own
//! four fresh points (nothing is shared with a neighbouring cell's copy of
//! the same physical face), which keeps this module a pure function of
//! `HostMesh`'s public fields and is legal for `VTK_POLYHEDRON`: faces are
//! not required to share vertex ids, only global point ids within `faces`
//! must also appear in that cell's `connectivity` slice, which they do by
//! construction below.

use std::io::Write;
use std::path::Path;

use crate::error::{Error, IoContext, Result};
use crate::io::output_types::{FieldValues, OutputField};
use crate::mesh::HostMesh;
use crate::{Scalar, Vec3};

const VTK_POLYHEDRON: u8 = 42;

// ---------------------------------------------------------------------------
// Face reconstruction
// ---------------------------------------------------------------------------

/// An orthonormal frame `(t1, t2)` spanning the plane perpendicular to unit
/// vector `n`, with `t1 x t2 == n`. Picks whichever coordinate axis is least
/// aligned with `n` as the seed so the cross product never degenerates.
fn tangent_frame(n: Vec3) -> (Vec3, Vec3) {
    let seed = if n.x.abs() <= n.y.abs() && n.x.abs() <= n.z.abs() {
        Vec3::new(1.0, 0.0, 0.0)
    } else if n.y.abs() <= n.z.abs() {
        Vec3::new(0.0, 1.0, 0.0)
    } else {
        Vec3::new(0.0, 0.0, 1.0)
    };
    let t1 = n.cross(seed).normalised();
    let t2 = n.cross(t1);
    (t1, t2)
}

/// The four corners of the synthetic quad for a face with the given centroid
/// and (non-unit) outward area vector, wound so `cross(p1-p0, p3-p0)` points
/// along the outward normal (the VTK polyhedron convention).
fn face_quad(centroid: Vec3, area_vec: Vec3) -> [Vec3; 4] {
    let mag = area_vec.mag();
    if mag <= Scalar::MIN_POSITIVE {
        // Degenerate face (zero area); emit a vanishingly small quad rather
        // than propagate NaNs into the file.
        let eps = 1e-12 as Scalar;
        return [
            centroid + Vec3::new(-eps, -eps, 0.0),
            centroid + Vec3::new(eps, -eps, 0.0),
            centroid + Vec3::new(eps, eps, 0.0),
            centroid + Vec3::new(-eps, eps, 0.0),
        ];
    }
    let n = area_vec * (1.0 / mag);
    let (t1, t2) = tangent_frame(n);
    let h = mag.sqrt() * 0.5;
    [
        centroid - t1 * h - t2 * h,
        centroid + t1 * h - t2 * h,
        centroid + t1 * h + t2 * h,
        centroid - t1 * h + t2 * h,
    ]
}

/// Push one face's four fresh points and its `[numPts, id...]` entry into the
/// running buffers, and record its ids in the cell's `connectivity` slice.
fn emit_face(
    centroid: Vec3,
    area_vec: Vec3,
    points: &mut Vec<Vec3>,
    connectivity: &mut Vec<i64>,
    cell_faces: &mut Vec<i64>,
) {
    let quad = face_quad(centroid, area_vec);
    let base = points.len() as i64;
    points.extend_from_slice(&quad);
    cell_faces.push(4);
    for k in 0..4i64 {
        cell_faces.push(base + k);
        connectivity.push(base + k);
    }
}

/// Everything the XML `<Piece>` and the appended binary blocks are built
/// from: the reconstructed points and the VTK `faces`/`connectivity`
/// arrays.
struct Geometry {
    points: Vec<Vec3>,
    connectivity: Vec<i64>,
    offsets: Vec<i64>,
    types: Vec<u8>,
    faces: Vec<i64>,
    faceoffsets: Vec<i64>,
}

fn build_geometry(m: &HostMesh) -> Geometry {
    let mut points = Vec::new();
    let mut connectivity = Vec::new();
    let mut offsets = Vec::with_capacity(m.n_cells);
    let mut types = Vec::with_capacity(m.n_cells);
    let mut faces = Vec::new();
    let mut faceoffsets = Vec::with_capacity(m.n_cells);

    for c in 0..m.n_cells {
        let mut cell_faces: Vec<i64> = Vec::new();
        let mut n_faces_cell: i64 = 0;

        let cf_start = m.cf_offset[c] as usize;
        let cf_end = m.cf_offset[c + 1] as usize;
        for i in cf_start..cf_end {
            let f = m.cf_face[i] as usize;
            let owns = m.cf_own[i] != 0;
            let area = if owns { m.sf[f] } else { -m.sf[f] };
            emit_face(m.cf[f], area, &mut points, &mut connectivity, &mut cell_faces);
            n_faces_cell += 1;
        }

        let bcf_start = m.bcf_offset[c] as usize;
        let bcf_end = m.bcf_offset[c + 1] as usize;
        for i in bcf_start..bcf_end {
            let bf = m.bcf_face[i] as usize;
            emit_face(m.b_cf[bf], m.b_sf[bf], &mut points, &mut connectivity, &mut cell_faces);
            n_faces_cell += 1;
        }

        faces.push(n_faces_cell);
        faces.extend_from_slice(&cell_faces);
        faceoffsets.push(faces.len() as i64);

        offsets.push(connectivity.len() as i64);
        types.push(VTK_POLYHEDRON);
    }

    Geometry { points, connectivity, offsets, types, faces, faceoffsets }
}

// ---------------------------------------------------------------------------
// Appended-data block assembly
// ---------------------------------------------------------------------------

/// One `AppendedData` block: raw little-endian payload bytes. The `UInt64`
/// byte count prefix (`header_type="UInt64"`) is written separately once the
/// block's final offset within the appended section is known.
struct Block {
    bytes: Vec<u8>,
}

impl Block {
    fn f64(vals: impl IntoIterator<Item = f64>) -> Self {
        let mut bytes = Vec::new();
        for v in vals {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        Self { bytes }
    }
    fn i64(vals: impl IntoIterator<Item = i64>) -> Self {
        let mut bytes = Vec::new();
        for v in vals {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        Self { bytes }
    }
    fn u8(vals: impl IntoIterator<Item = u8>) -> Self {
        Self { bytes: vals.into_iter().collect() }
    }
}

/// Registers appended-data blocks in file order and hands back each one's
/// `offset` attribute (byte offset from the start of the appended section,
/// i.e. from the byte right after the leading `_` marker).
#[derive(Default)]
struct AppendedWriter {
    blocks: Vec<Block>,
    cursor: u64,
}

impl AppendedWriter {
    fn push(&mut self, block: Block) -> u64 {
        let offset = self.cursor;
        self.cursor += 8 + block.bytes.len() as u64; // UInt64 header_type
        self.blocks.push(block);
        offset
    }

    /// Concatenate `[len:u64][bytes]` for every registered block, in order.
    fn into_bytes(self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.cursor as usize);
        for b in self.blocks {
            out.extend_from_slice(&(b.bytes.len() as u64).to_le_bytes());
            out.extend_from_slice(&b.bytes);
        }
        out
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Write one serial `.vtu` (UnstructuredGrid, appended raw binary, little
/// endian, `header_type="UInt64"`). Cells are the mesh's polyhedra
/// (`VTK_POLYHEDRON`, type 42); `fields` become `CellData`, one value per
/// mesh cell.
pub fn write_vtu(
    path: &Path,
    m: &HostMesh,
    fields: &[OutputField],
    time: Option<Scalar>,
) -> Result<()> {
    for f in fields {
        if f.len() != m.n_cells {
            return Err(Error::Field {
                field: f.name.to_string(),
                msg: format!(
                    "has {} value(s), mesh has {} cell(s)",
                    f.len(),
                    m.n_cells
                ),
            });
        }
    }

    let geo = build_geometry(m);
    let n_points = geo.points.len();
    let n_cells = m.n_cells;

    let mut app = AppendedWriter::default();

    let off_points = app.push(Block::f64(
        geo.points.iter().flat_map(|p| [p.x as f64, p.y as f64, p.z as f64]),
    ));
    let off_connectivity = app.push(Block::i64(geo.connectivity.iter().copied()));
    let off_offsets = app.push(Block::i64(geo.offsets.iter().copied()));
    let off_types = app.push(Block::u8(geo.types.iter().copied()));
    let off_faces = app.push(Block::i64(geo.faces.iter().copied()));
    let off_faceoffsets = app.push(Block::i64(geo.faceoffsets.iter().copied()));

    let off_time = time.map(|t| app.push(Block::f64([t as f64])));

    struct FieldMeta {
        name: String,
        n_comp: usize,
        offset: u64,
    }
    let mut field_meta = Vec::with_capacity(fields.len());
    for f in fields {
        let (n_comp, block) = match &f.values {
            FieldValues::Scalar(v) => (1usize, Block::f64(v.iter().map(|&x| x as f64))),
            FieldValues::Vector(v) => (
                3usize,
                Block::f64(v.iter().flat_map(|p| [p.x as f64, p.y as f64, p.z as f64])),
            ),
        };
        let offset = app.push(block);
        field_meta.push(FieldMeta { name: f.name.to_string(), n_comp, offset });
    }

    let mut xml = String::new();
    xml.push_str("<VTKFile type=\"UnstructuredGrid\" version=\"1.0\" byte_order=\"LittleEndian\" header_type=\"UInt64\">\n");
    xml.push_str("  <UnstructuredGrid>\n");
    xml.push_str(&format!(
        "    <Piece NumberOfPoints=\"{n_points}\" NumberOfCells=\"{n_cells}\">\n"
    ));

    if let Some(off) = off_time {
        xml.push_str("      <FieldData>\n");
        xml.push_str(&format!(
            "        <DataArray type=\"Float64\" Name=\"TIME\" NumberOfTuples=\"1\" format=\"appended\" offset=\"{off}\"/>\n"
        ));
        xml.push_str("      </FieldData>\n");
    }

    xml.push_str("      <Points>\n");
    xml.push_str(&format!(
        "        <DataArray type=\"Float64\" NumberOfComponents=\"3\" format=\"appended\" offset=\"{off_points}\"/>\n"
    ));
    xml.push_str("      </Points>\n");

    xml.push_str("      <Cells>\n");
    xml.push_str(&format!(
        "        <DataArray type=\"Int64\" Name=\"connectivity\" format=\"appended\" offset=\"{off_connectivity}\"/>\n"
    ));
    xml.push_str(&format!(
        "        <DataArray type=\"Int64\" Name=\"offsets\" format=\"appended\" offset=\"{off_offsets}\"/>\n"
    ));
    xml.push_str(&format!(
        "        <DataArray type=\"UInt8\" Name=\"types\" format=\"appended\" offset=\"{off_types}\"/>\n"
    ));
    xml.push_str(&format!(
        "        <DataArray type=\"Int64\" Name=\"faces\" format=\"appended\" offset=\"{off_faces}\"/>\n"
    ));
    xml.push_str(&format!(
        "        <DataArray type=\"Int64\" Name=\"faceoffsets\" format=\"appended\" offset=\"{off_faceoffsets}\"/>\n"
    ));
    xml.push_str("      </Cells>\n");

    if !field_meta.is_empty() {
        let scalars = field_meta
            .iter()
            .find(|f| f.n_comp == 1)
            .map(|f| f.name.as_str());
        let vectors = field_meta
            .iter()
            .find(|f| f.n_comp == 3)
            .map(|f| f.name.as_str());
        xml.push_str("      <CellData");
        if let Some(s) = scalars {
            xml.push_str(&format!(" Scalars=\"{s}\""));
        }
        if let Some(v) = vectors {
            xml.push_str(&format!(" Vectors=\"{v}\""));
        }
        xml.push_str(">\n");
        for f in &field_meta {
            xml.push_str(&format!(
                "        <DataArray type=\"Float64\" Name=\"{}\" NumberOfComponents=\"{}\" format=\"appended\" offset=\"{}\"/>\n",
                f.name, f.n_comp, f.offset
            ));
        }
        xml.push_str("      </CellData>\n");
    }

    xml.push_str("    </Piece>\n");
    xml.push_str("  </UnstructuredGrid>\n");
    xml.push_str("  <AppendedData encoding=\"raw\">\n_");

    let appended_bytes = app.into_bytes();

    let mut out = std::fs::File::create(path).path(path)?;
    out.write_all(xml.as_bytes()).path(path)?;
    out.write_all(&appended_bytes).path(path)?;
    out.write_all(b"\n  </AppendedData>\n</VTKFile>\n").path(path)?;

    Ok(())
}

/// Write the live Lagrangian parcels as a serial `.vtp` (PolyData, appended
/// raw binary, little endian, `header_type="UInt64"`) - SPEC-LIT §66.13.
///
/// One VTK vertex per live parcel, in slot order, with the parcel state as
/// `PointData`. `.vtp` rather than `.vtu` because PolyData with a `Verts`
/// section is the conventional VTK form for a particle cloud and is what
/// ParaView's Glyph filter expects; it is the same appended-binary encoding
/// this module already writes, from the same published Kitware "VTK File
/// Formats" reference, so no new format was decoded.
///
/// **Dead slots are not written.** A parcel whose `cell` is negative has left
/// the domain or was never filled; emitting it would put a stale position in
/// the file and make the point count meaningless as a parcel count.
///
/// `uid` is emitted as `Float64` rather than as an integer array, because a
/// 64-bit identity does not survive `Float64` exactly and a reader that
/// silently rounds it would be worse than one that never had it: the
/// low-order half is written as `uidLow` alongside, so the full identity is
/// recoverable from the file. The identity is not a physical quantity - it is
/// there to follow one parcel across output times.
pub fn write_parcels_vtp(
    path: &Path,
    s: &crate::parcels::ParcelSnapshot,
    time: Option<Scalar>,
) -> Result<()> {
    let live = s.live();
    let n = live.len();

    let mut app = AppendedWriter::default();

    let off_points = app.push(Block::f64(
        live.iter()
            .flat_map(|&i| [s.x[i].x as f64, s.x[i].y as f64, s.x[i].z as f64]),
    ));
    let off_connectivity = app.push(Block::i64((0..n as i64).collect::<Vec<_>>()));
    let off_offsets = app.push(Block::i64((1..=n as i64).collect::<Vec<_>>()));
    let off_time = time.map(|t| app.push(Block::f64([t as f64])));

    let off_u = app.push(Block::f64(
        live.iter()
            .flat_map(|&i| [s.u[i].x as f64, s.u[i].y as f64, s.u[i].z as f64]),
    ));
    let off_d = app.push(Block::f64(live.iter().map(|&i| s.d[i] as f64)));
    let off_t = app.push(Block::f64(live.iter().map(|&i| s.temperature[i] as f64)));
    let off_np = app.push(Block::f64(live.iter().map(|&i| s.n_p[i] as f64)));
    let off_cell = app.push(Block::i64(live.iter().map(|&i| i64::from(s.cell[i]))));
    let off_uid_hi = app.push(Block::f64(live.iter().map(|&i| (s.uid[i] >> 32) as f64)));
    let off_uid_lo = app.push(Block::f64(
        live.iter().map(|&i| (s.uid[i] & 0xffff_ffff) as f64),
    ));

    let mut xml = String::new();
    xml.push_str("<VTKFile type=\"PolyData\" version=\"1.0\" byte_order=\"LittleEndian\" header_type=\"UInt64\">\n");
    xml.push_str("  <PolyData>\n");
    xml.push_str(&format!(
        "    <Piece NumberOfPoints=\"{n}\" NumberOfVerts=\"{n}\" NumberOfLines=\"0\" \
         NumberOfStrips=\"0\" NumberOfPolys=\"0\">\n"
    ));

    if let Some(off) = off_time {
        xml.push_str("      <FieldData>\n");
        xml.push_str(&format!(
            "        <DataArray type=\"Float64\" Name=\"TIME\" NumberOfTuples=\"1\" format=\"appended\" offset=\"{off}\"/>\n"
        ));
        xml.push_str("      </FieldData>\n");
    }

    xml.push_str("      <Points>\n");
    xml.push_str(&format!(
        "        <DataArray type=\"Float64\" NumberOfComponents=\"3\" format=\"appended\" offset=\"{off_points}\"/>\n"
    ));
    xml.push_str("      </Points>\n");

    xml.push_str("      <Verts>\n");
    xml.push_str(&format!(
        "        <DataArray type=\"Int64\" Name=\"connectivity\" format=\"appended\" offset=\"{off_connectivity}\"/>\n"
    ));
    xml.push_str(&format!(
        "        <DataArray type=\"Int64\" Name=\"offsets\" format=\"appended\" offset=\"{off_offsets}\"/>\n"
    ));
    xml.push_str("      </Verts>\n");

    xml.push_str("      <PointData Scalars=\"d\" Vectors=\"U\">\n");
    for (name, comps, off) in [
        ("U", 3, off_u),
        ("d", 1, off_d),
        ("T", 1, off_t),
        ("nP", 1, off_np),
        ("uidHigh", 1, off_uid_hi),
        ("uidLow", 1, off_uid_lo),
    ] {
        xml.push_str(&format!(
            "        <DataArray type=\"Float64\" Name=\"{name}\" NumberOfComponents=\"{comps}\" format=\"appended\" offset=\"{off}\"/>\n"
        ));
    }
    xml.push_str(&format!(
        "        <DataArray type=\"Int64\" Name=\"cell\" NumberOfComponents=\"1\" format=\"appended\" offset=\"{off_cell}\"/>\n"
    ));
    xml.push_str("      </PointData>\n");

    xml.push_str("    </Piece>\n");
    xml.push_str("  </PolyData>\n");
    xml.push_str("  <AppendedData encoding=\"raw\">\n_");

    let appended_bytes = app.into_bytes();

    let mut out = std::fs::File::create(path).path(path)?;
    out.write_all(xml.as_bytes()).path(path)?;
    out.write_all(&appended_bytes).path(path)?;
    out.write_all(b"\n  </AppendedData>\n</VTKFile>\n").path(path)?;

    Ok(())
}

/// Write a ParaView `.pvd` "Collection" file listing a `.vtu` per time step.
/// `series` pairs each output time with the path to its `.vtu` (written as
/// given - relative paths are the caller's job if the two files must move
/// together).
pub fn write_pvd(path: &Path, series: &[(Scalar, std::path::PathBuf)]) -> Result<()> {
    let mut xml = String::new();
    xml.push_str("<VTKFile type=\"Collection\" version=\"0.1\" byte_order=\"LittleEndian\">\n");
    xml.push_str("  <Collection>\n");
    for (t, p) in series {
        let file = p.to_string_lossy().replace('\\', "/");
        xml.push_str(&format!(
            "    <DataSet timestep=\"{t}\" part=\"0\" file=\"{file}\"/>\n"
        ));
    }
    xml.push_str("  </Collection>\n");
    xml.push_str("</VTKFile>\n");
    std::fs::write(path, xml.as_bytes()).path(path)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::topology::tests::box_mesh;
    use crate::mesh::HostMesh;

    /// Minimal internal reader: pulls every `format="appended" offset="N"`
    /// DataArray out of the XML preamble (name -> byte offset) and hands back
    /// slices of the appended blob at those offsets, respecting the
    /// `UInt64` length prefix this writer used. Deliberately independent of
    /// the writer's internal `AppendedWriter`/`Block` types, so a bug that
    /// corrupts an offset or a length is not masked by re-using the same
    /// code to check it.
    struct Parsed {
        appended: Vec<u8>,
        offsets: std::collections::HashMap<String, u64>,
        n_points: usize,
        n_cells: usize,
    }

    /// Byte-string search, since the appended section is raw binary and
    /// cannot go through a UTF-8 `String` without `from_utf8_lossy` shifting
    /// offsets (invalid sequences get replaced by a multi-byte replacement
    /// character, changing the byte count).
    fn find_bytes(haystack: &[u8], needle: &[u8]) -> Option<usize> {
        haystack.windows(needle.len()).position(|w| w == needle)
    }

    fn parse(bytes: &[u8]) -> Parsed {
        // Appended section starts right after "encoding=\"raw\">\n_"; find its
        // start in the raw bytes so binary data never passes through a
        // lossy UTF-8 conversion that could shift offsets.
        let marker = b"encoding=\"raw\">\n_";
        let marker_start = find_bytes(bytes, marker).expect("no AppendedData marker");
        let data_start = marker_start + marker.len();
        let end_marker = b"\n  </AppendedData>";
        let data_end =
            find_bytes(&bytes[data_start..], end_marker).expect("no AppendedData end") + data_start;
        let appended = bytes[data_start..data_end].to_vec();

        // The XML preamble (everything before the appended section) is plain
        // ASCII, so it is safe to parse as text.
        let text = std::str::from_utf8(&bytes[..marker_start]).expect("preamble is valid UTF-8");

        let mut offsets = std::collections::HashMap::new();
        for line in text.lines() {
            if !line.contains("format=\"appended\"") {
                continue;
            }
            let key = if let Some(i) = line.find("Name=\"") {
                let rest = &line[i + 6..];
                rest[..rest.find('"').unwrap()].to_string()
            } else if line.contains("Points") || line.contains("connectivity") {
                "points".to_string()
            } else {
                "unnamed".to_string()
            };
            let oi = line.find("offset=\"").unwrap() + 8;
            let rest = &line[oi..];
            let off: u64 = rest[..rest.find('"').unwrap()].parse().unwrap();
            offsets.insert(key, off);
        }

        let n_points = text
            .find("NumberOfPoints=\"")
            .map(|i| {
                let rest = &text[i + 16..];
                rest[..rest.find('"').unwrap()].parse().unwrap()
            })
            .unwrap();
        let n_cells = text
            .find("NumberOfCells=\"")
            .map(|i| {
                let rest = &text[i + 15..];
                rest[..rest.find('"').unwrap()].parse().unwrap()
            })
            .unwrap();

        Parsed { appended, offsets, n_points, n_cells }
    }

    fn block_at(p: &Parsed, offset: u64) -> &[u8] {
        let off = offset as usize;
        let len = u64::from_le_bytes(p.appended[off..off + 8].try_into().unwrap()) as usize;
        &p.appended[off + 8..off + 8 + len]
    }

    fn read_i64_block(p: &Parsed, name: &str) -> Vec<i64> {
        let off = p.offsets[name];
        block_at(p, off)
            .chunks_exact(8)
            .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
            .collect()
    }

    fn mesh_2x1x1() -> HostMesh {
        let (mut m, points, faces) = box_mesh([2, 1, 1], Vec3::new(1.0, 1.0, 1.0));
        m.compute_geometry(&points, &faces).expect("geometry");
        m.build_cell_face_maps();
        m
    }

    #[test]
    fn round_trips_and_offsets_are_consistent() {
        let m = mesh_2x1x1();
        let scalar_vals: Vec<Scalar> = (0..m.n_cells).map(|i| 1.5 * (i as Scalar) + 0.25).collect();
        let fields = vec![OutputField::scalar("p", &scalar_vals)];

        let dir = std::env::temp_dir();
        let path = dir.join("ofgpu_test_2x1x1.vtu");
        write_vtu(&path, &m, &fields, Some(0.5)).expect("write_vtu");

        let bytes = std::fs::read(&path).expect("read back");
        let p = parse(&bytes);

        assert_eq!(p.n_cells, m.n_cells);
        assert!(p.n_points > 0);

        // offsets: NumberOfComponents=3 connectivity length must equal the
        // last "offsets" entry.
        let offsets_arr = read_i64_block(&p, "offsets");
        assert_eq!(offsets_arr.len(), m.n_cells);
        let connectivity = read_i64_block(&p, "connectivity");
        assert_eq!(*offsets_arr.last().unwrap(), connectivity.len() as i64);

        // faces count per cell matches the mesh's own per-cell face count
        // from the cell -> face CSR (internal + boundary).
        let faces_arr = read_i64_block(&p, "faces");
        let faceoffsets_arr = read_i64_block(&p, "faceoffsets");
        assert_eq!(faceoffsets_arr.len(), m.n_cells);

        let mut cursor = 0usize;
        for c in 0..m.n_cells {
            let n_faces_from_file = faces_arr[cursor];
            let expected = (m.cf_offset[c + 1] - m.cf_offset[c]) as i64
                + (m.bcf_offset[c + 1] - m.bcf_offset[c]) as i64;
            assert_eq!(n_faces_from_file, expected, "cell {c} face count");

            // Walk this cell's face records to find the next cursor and
            // check faceoffsets is exact.
            let mut walk = cursor + 1;
            for _ in 0..n_faces_from_file {
                let n_pts = faces_arr[walk] as usize;
                walk += 1 + n_pts;
            }
            assert_eq!(walk as i64, faceoffsets_arr[c], "faceoffsets exact for cell {c}");
            cursor = walk;
        }
        assert_eq!(cursor, faces_arr.len(), "appended faces section length exact");

        // scalar field round-trips bit-exact through our own reader.
        let field_off = p.offsets["p"];
        let raw = block_at(&p, field_off);
        assert_eq!(raw.len(), m.n_cells * 8);
        let read_back: Vec<f64> = raw
            .chunks_exact(8)
            .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
            .collect();
        for (i, &v) in scalar_vals.iter().enumerate() {
            assert_eq!(read_back[i], v as f64, "field value {i} bit-exact");
        }

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn rejects_mismatched_field_length() {
        let m = mesh_2x1x1();
        let too_short = vec![0.0 as Scalar; m.n_cells - 1];
        let fields = vec![OutputField::scalar("bad", &too_short)];
        let path = std::env::temp_dir().join("ofgpu_test_mismatch.vtu");
        let err = write_vtu(&path, &m, &fields, None).unwrap_err();
        match err {
            Error::Field { field, .. } => assert_eq!(field, "bad"),
            other => panic!("expected Error::Field, got {other:?}"),
        }
    }

    #[test]
    fn writes_pvd_collection() {
        let dir = std::env::temp_dir();
        let path = dir.join("ofgpu_test_series.pvd");
        let series = vec![
            (0.0 as Scalar, std::path::PathBuf::from("case_0.vtu")),
            (1.0 as Scalar, std::path::PathBuf::from("case_1.vtu")),
        ];
        write_pvd(&path, &series).expect("write_pvd");
        let text = std::fs::read_to_string(&path).expect("read pvd");
        assert!(text.contains("case_0.vtu"));
        assert!(text.contains("case_1.vtu"));
        assert!(text.contains("Collection"));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn vector_field_has_three_components() {
        let m = mesh_2x1x1();
        let vecs: Vec<Vec3> = (0..m.n_cells)
            .map(|i| Vec3::new(i as Scalar, 2.0 * i as Scalar, -(i as Scalar)))
            .collect();
        let fields = vec![OutputField::vector("U", &vecs)];
        let path = std::env::temp_dir().join("ofgpu_test_vector.vtu");
        write_vtu(&path, &m, &fields, None).expect("write_vtu");
        let bytes = std::fs::read(&path).expect("read back");
        let p = parse(&bytes);
        let off = p.offsets["U"];
        let raw = block_at(&p, off);
        assert_eq!(raw.len(), m.n_cells * 3 * 8);
        let _ = std::fs::remove_file(&path);
    }
}
