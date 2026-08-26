// meteor-cfd - Copyright (c) 2026 주식회사 메테오시뮬레이션 (Meteo Simulation Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

//! The `.mcr` restart format: exact `f64` fields including `phi`, a mesh
//! hash, refuses to load onto a different mesh.
//!
//! Written from:
//!   `docs/05-io-redesign.md` §4.6 ("자체 바이너리" - our own binary format,
//!     a header with version / cell counts / field list / **mesh hash**,
//!     rejected on hash mismatch) and §7/Q5 ("재시작은 한 형식으로 되는가? -
//!     안 됩니다", i.e. a single restart format is not optional - this is the
//!     format). The problem this closes is named directly in §4.6: the
//!     `ofgpu-buoyant` driver does not write `phi` on restart today, so a
//!     restarted run falls back to a non-conservative potential-flow flux
//!     instead of the converged one. This is meteor-cfd's own format, not a
//!     transcription of anything external.
//!
//! No GPL-licensed source was consulted.
//!
//! # Layout (little-endian throughout)
//!
//! ```text
//! magic          [8]   b"MCFDRSTR"
//! version        u32   = 2
//! endian check   u32   = 0x01020304
//! mesh_hash      u64
//! time           f64
//! p0             f64
//! dp0dt          f64   (added at version 2 - SPEC-LIT §25.2/§31.2)
//! n_cells        u64
//! n_internal     u64   (internal faces)
//! n_boundary     u64   (boundary faces)
//! n_fields       u32
//! per field:
//!   name_len     u32
//!   name         [name_len] utf-8
//!   kind         u8    0 = cell scalar, 1 = cell vector, 2 = surface scalar
//!   internal     f64[] (n_cells, n_cells*3 xyz-interleaved, or n_internal)
//!   boundary     f64[] (n_boundary, n_boundary*3 xyz-interleaved, or n_boundary)
//! ```
//!
//! # Mesh hash
//!
//! [`mesh_hash`] is an FNV-1a 64-bit digest over the mesh's LDU addressing
//! (`owner`, `neighbour`), its face/cell geometry, and its patch names -
//! deliberately not just cell/face counts, since two different meshes can
//! share those. One deviation from a literal reading of "hash the points":
//! `HostMesh` does not retain the raw point array after
//! `compute_geometry` folds it into cell and face geometry (see
//! `src/mesh.rs`), so the hash instead covers the geometry that array
//! produced - cell volumes and centres, and face areas - which is at least
//! as sensitive to a changed point position and is what the solver actually
//! reads. This is documented here rather than silently assumed.
//!
//! A mismatch is refused unconditionally (never downgraded, even under
//! `-permissive`): there is no sane substitute for "this data belongs to a
//! different mesh", so this is a hard error naming both hashes, not a
//! section-13.4 `unsupported()`/`unreadable()` case (those exist for case
//! settings with a documented fallback; a restart has none).
//!
//! # Version 2: `dp0dt`
//!
//! SPEC-LIT §31.2's gate for `ofgpu-fire` found that `p0` alone is not
//! enough: `ofgpu::energy::Energy::update_target_divergence` reads
//! `GasState::dp0dt` at a ONE-ITERATION LAG (the value
//! [`crate::energy::GasState::advance_p0`] computed at the END of the
//! previous unit of work) - exactly the segregated lag every other coupling
//! coefficient in that driver already runs at. A `GasState` rebuilt fresh
//! from a checkpoint's `p0` alone starts with `dp0dt = 0`, which is the
//! correct value on a cold start and the WRONG one on a restart of a sealed
//! (§25.2) case with an ongoing heat release: the first pressure solve after
//! resuming would assemble the low-Mach target divergence without the
//! `-dp0dt/(gamma p0)` term the continuous run's own next step carried,
//! producing a first pressure residual that does not match the continuous
//! run's even though every FIELD (`U`, `p`, `T`, the species) was restored
//! bit-exact. `dp0dt` closes that gap. A version-1 file has no such field
//! and is refused by the version check below rather than silently read with
//! a wrong offset - the same reasoning [`mesh_hash`] mismatches are refused
//! by.

use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

use crate::error::{Error, Result, IoContext};
use crate::mesh::HostMesh;

const MAGIC: &[u8; 8] = b"MCFDRSTR";
const VERSION: u32 = 2;
const ENDIAN_CHECK: u32 = 0x0102_0304;

// ==========================================================================
//  Mesh hash
// ==========================================================================

/// FNV-1a 64-bit digest of the parts of a [`HostMesh`] that identify it: the
/// LDU addressing, the geometry that the (unretained) points produced, and
/// the patch names. See the module doc for why cell centres/volumes/face
/// areas stand in for raw points.
pub fn mesh_hash(mesh: &HostMesh) -> u64 {
    const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

    let mut h: u64 = FNV_OFFSET;
    let mut feed = |bytes: &[u8]| {
        for &b in bytes {
            h ^= b as u64;
            h = h.wrapping_mul(FNV_PRIME);
        }
    };

    feed(&(mesh.n_cells as u64).to_le_bytes());
    feed(&(mesh.n_internal_faces as u64).to_le_bytes());
    feed(&(mesh.n_boundary_faces as u64).to_le_bytes());
    feed(&(mesh.n_points as u64).to_le_bytes());

    for &o in &mesh.owner {
        feed(&o.to_le_bytes());
    }
    for &n in &mesh.neighbour {
        feed(&n.to_le_bytes());
    }
    for v in &mesh.v {
        feed(&v.to_le_bytes());
    }
    for c in &mesh.c {
        feed(&c.x.to_le_bytes());
        feed(&c.y.to_le_bytes());
        feed(&c.z.to_le_bytes());
    }
    for sf in &mesh.sf {
        feed(&sf.x.to_le_bytes());
        feed(&sf.y.to_le_bytes());
        feed(&sf.z.to_le_bytes());
    }

    for p in &mesh.patches {
        feed(p.name.as_bytes());
        feed(&[0u8]); // separator, so "ab","c" cannot collide with "a","bc"
        feed(&(p.size as u64).to_le_bytes());
    }

    h
}

// ==========================================================================
//  Data model
// ==========================================================================

/// What a restart field is defined over - the on-disk `kind` byte.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FieldKind {
    /// One `f64` per cell.
    CellScalar,
    /// One `Vec3` (x,y,z interleaved) per cell.
    CellVector,
    /// One `f64` per face - internal faces then boundary faces. This is the
    /// shape `phi` needs.
    SurfaceScalar,
}

impl FieldKind {
    fn to_u8(self) -> u8 {
        match self {
            FieldKind::CellScalar => 0,
            FieldKind::CellVector => 1,
            FieldKind::SurfaceScalar => 2,
        }
    }

    fn from_u8(v: u8, path: &Path) -> Result<Self> {
        match v {
            0 => Ok(FieldKind::CellScalar),
            1 => Ok(FieldKind::CellVector),
            2 => Ok(FieldKind::SurfaceScalar),
            other => Err(Error::Parse {
                path: path.display().to_string(),
                msg: format!("unknown restart field kind byte {other} (expected 0, 1 or 2)"),
            }),
        }
    }

    /// `(internal_len, boundary_len)` for this kind, given the mesh sizes.
    fn expected_lens(self, n_cells: u64, n_internal: u64, n_boundary: u64) -> (u64, u64) {
        match self {
            FieldKind::CellScalar => (n_cells, n_boundary),
            FieldKind::CellVector => (n_cells * 3, n_boundary * 3),
            FieldKind::SurfaceScalar => (n_internal, n_boundary),
        }
    }
}

/// One named field carried by a restart: its raw `f64` internal and boundary
/// arrays, already in the on-disk layout (vectors xyz-interleaved).
#[derive(Debug, Clone)]
pub struct RestartField {
    pub name: String,
    pub kind: FieldKind,
    pub internal: Vec<f64>,
    pub boundary: Vec<f64>,
}

/// Everything a `.mcr` file holds.
#[derive(Debug, Clone)]
pub struct RestartData {
    /// [`mesh_hash`] of the mesh this restart was written from.
    pub mesh_hash: u64,
    pub time: f64,
    pub p0: f64,
    /// Version 2+ only - see the module doc's "Version 2: `dp0dt`" section.
    /// `0.0` for `ofgpu-buoyant`/`ofgpu-vof`, which have no `p0` ODE and
    /// nothing to carry here.
    pub dp0dt: f64,
    pub n_cells: u64,
    pub n_internal: u64,
    pub n_boundary: u64,
    pub fields: Vec<RestartField>,
}

// ==========================================================================
//  Byte-level helpers
// ==========================================================================

fn truncated(path: &Path, what: &str) -> Error {
    Error::Parse {
        path: path.display().to_string(),
        msg: format!("truncated .mcr restart file: could not read {what}"),
    }
}

fn read_u8(r: &mut impl Read, path: &Path, what: &str) -> Result<u8> {
    let mut buf = [0u8; 1];
    r.read_exact(&mut buf).map_err(|_| truncated(path, what))?;
    Ok(buf[0])
}

fn read_u32(r: &mut impl Read, path: &Path, what: &str) -> Result<u32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf).map_err(|_| truncated(path, what))?;
    Ok(u32::from_le_bytes(buf))
}

fn read_u64(r: &mut impl Read, path: &Path, what: &str) -> Result<u64> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf).map_err(|_| truncated(path, what))?;
    Ok(u64::from_le_bytes(buf))
}

fn read_f64(r: &mut impl Read, path: &Path, what: &str) -> Result<f64> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf).map_err(|_| truncated(path, what))?;
    Ok(f64::from_le_bytes(buf))
}

fn read_f64_array(r: &mut impl Read, path: &Path, what: &str, n: u64) -> Result<Vec<f64>> {
    let n = usize::try_from(n).map_err(|_| Error::Parse {
        path: path.display().to_string(),
        msg: format!("{what}: length {n} does not fit this platform's usize"),
    })?;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        out.push(read_f64(r, path, what)?);
    }
    Ok(out)
}

fn read_name(r: &mut impl Read, path: &Path) -> Result<String> {
    let len = read_u32(r, path, "a field name length")?;
    let len = len as usize;
    let mut buf = vec![0u8; len];
    r.read_exact(&mut buf)
        .map_err(|_| truncated(path, "a field name"))?;
    String::from_utf8(buf).map_err(|e| Error::Parse {
        path: path.display().to_string(),
        msg: format!("a field name is not valid utf-8: {e}"),
    })
}

fn write_f64_array(w: &mut impl Write, path: &Path, values: &[f64]) -> Result<()> {
    for v in values {
        w.write_all(&v.to_le_bytes()).path(path)?;
    }
    Ok(())
}

// ==========================================================================
//  Public API
// ==========================================================================

/// Write `data` to `path` in the `.mcr` format described in the module doc.
///
/// Every field's `internal`/`boundary` lengths are checked against its
/// `kind` and `data`'s cell/face counts before anything is written, so a
/// caller mistake produces a named error instead of a file nobody can read
/// back.
pub fn write_restart(path: impl AsRef<Path>, data: &RestartData) -> Result<()> {
    let path = path.as_ref();
    let file = File::create(path).path(path)?;
    let mut w = BufWriter::new(file);

    for field in &data.fields {
        let (want_internal, want_boundary) =
            field.kind.expected_lens(data.n_cells, data.n_internal, data.n_boundary);
        if field.internal.len() as u64 != want_internal
            || field.boundary.len() as u64 != want_boundary
        {
            return Err(Error::Field {
                field: field.name.clone(),
                msg: format!(
                    "internal/boundary length {}/{} does not match {:?} on a mesh of \
                     {} cells / {} internal faces / {} boundary faces (expected {}/{})",
                    field.internal.len(),
                    field.boundary.len(),
                    field.kind,
                    data.n_cells,
                    data.n_internal,
                    data.n_boundary,
                    want_internal,
                    want_boundary,
                ),
            });
        }
    }

    w.write_all(MAGIC).path(path)?;
    w.write_all(&VERSION.to_le_bytes()).path(path)?;
    w.write_all(&ENDIAN_CHECK.to_le_bytes()).path(path)?;
    w.write_all(&data.mesh_hash.to_le_bytes()).path(path)?;
    w.write_all(&data.time.to_le_bytes()).path(path)?;
    w.write_all(&data.p0.to_le_bytes()).path(path)?;
    w.write_all(&data.dp0dt.to_le_bytes()).path(path)?;
    w.write_all(&data.n_cells.to_le_bytes()).path(path)?;
    w.write_all(&data.n_internal.to_le_bytes()).path(path)?;
    w.write_all(&data.n_boundary.to_le_bytes()).path(path)?;
    w.write_all(&(data.fields.len() as u32).to_le_bytes())
        .path(path)?;

    for field in &data.fields {
        let name_bytes = field.name.as_bytes();
        w.write_all(&(name_bytes.len() as u32).to_le_bytes())
            .path(path)?;
        w.write_all(name_bytes).path(path)?;
        w.write_all(&[field.kind.to_u8()]).path(path)?;
        write_f64_array(&mut w, path, &field.internal)?;
        write_f64_array(&mut w, path, &field.boundary)?;
    }

    w.flush().path(path)?;
    Ok(())
}

/// Read a `.mcr` restart from `path`, refusing it unless it was written from
/// a mesh whose [`mesh_hash`] is `expected_hash`.
///
/// A hash mismatch, an unsupported version, or a truncated/corrupt file are
/// all hard errors - never downgraded by `-permissive`, since there is no
/// safe substitute for "this data may belong to a different mesh" or "this
/// file is missing bytes".
pub fn read_restart(path: impl AsRef<Path>, expected_hash: u64) -> Result<RestartData> {
    let path = path.as_ref();
    let file = File::open(path).path(path)?;
    let mut r = BufReader::new(file);

    let mut magic = [0u8; 8];
    r.read_exact(&mut magic)
        .map_err(|_| truncated(path, "the .mcr magic bytes"))?;
    if &magic != MAGIC {
        return Err(Error::Parse {
            path: path.display().to_string(),
            msg: "not a .mcr restart file (bad magic bytes)".to_string(),
        });
    }

    let version = read_u32(&mut r, path, "the format version")?;
    if version != VERSION {
        return Err(Error::Parse {
            path: path.display().to_string(),
            msg: format!(
                "restart format version {version} is not supported (this build reads \
                 version {VERSION} only)"
            ),
        });
    }

    let endian = read_u32(&mut r, path, "the endian check word")?;
    if endian != ENDIAN_CHECK {
        return Err(Error::Parse {
            path: path.display().to_string(),
            msg: format!(
                "the endian check word is 0x{endian:08x}, expected 0x{ENDIAN_CHECK:08x} - \
                 the file is corrupt or was not written by this format"
            ),
        });
    }

    let file_hash = read_u64(&mut r, path, "the mesh hash")?;
    if file_hash != expected_hash {
        return Err(Error::Parse {
            path: path.display().to_string(),
            msg: format!(
                "this restart belongs to a different mesh (file mesh hash \
                 0x{file_hash:016x}, current mesh hash 0x{expected_hash:016x})"
            ),
        });
    }

    let time = read_f64(&mut r, path, "the restart time")?;
    let p0 = read_f64(&mut r, path, "p0")?;
    let dp0dt = read_f64(&mut r, path, "dp0dt")?;
    let n_cells = read_u64(&mut r, path, "n_cells")?;
    let n_internal = read_u64(&mut r, path, "n_internal")?;
    let n_boundary = read_u64(&mut r, path, "n_boundary")?;
    let n_fields = read_u32(&mut r, path, "n_fields")?;

    let mut fields = Vec::with_capacity(n_fields as usize);
    for _ in 0..n_fields {
        let name = read_name(&mut r, path)?;
        let kind = FieldKind::from_u8(read_u8(&mut r, path, "a field kind byte")?, path)?;
        let (internal_len, boundary_len) = kind.expected_lens(n_cells, n_internal, n_boundary);
        let internal = read_f64_array(&mut r, path, "a field's internal values", internal_len)?;
        let boundary = read_f64_array(&mut r, path, "a field's boundary values", boundary_len)?;
        fields.push(RestartField { name, kind, internal, boundary });
    }

    Ok(RestartData {
        mesh_hash: file_hash,
        time,
        p0,
        dp0dt,
        n_cells,
        n_internal,
        n_boundary,
        fields,
    })
}

// ==========================================================================
//  Tests
// ==========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Vec3;
    use crate::Label;
    use crate::mesh::{HostMesh, PatchInfo, PatchKind};
    use std::io::Write;

    /// A tiny two-cell mesh, just enough to exercise `mesh_hash` and give
    /// field arrays realistic lengths (1 internal face, 2 boundary faces).
    fn tiny_mesh() -> HostMesh {
        let mut m = HostMesh::default();
        m.n_cells = 2;
        m.n_internal_faces = 1;
        m.n_boundary_faces = 2;
        m.n_points = 8;
        m.owner = vec![0 as Label];
        m.neighbour = vec![1 as Label];
        m.v = vec![1.0, 1.0];
        m.c = vec![Vec3::new(0.5, 0.5, 0.5), Vec3::new(1.5, 0.5, 0.5)];
        m.sf = vec![Vec3::new(1.0, 0.0, 0.0)];
        m.mag_sf = vec![1.0];
        m.b_face_cells = vec![0, 1];
        m.patches = vec![PatchInfo {
            name: "walls".to_string(),
            type_name: "wall".to_string(),
            kind: PatchKind::Wall,
            start: 0,
            size: 2,
            nbr_patch: None,
        }];
        m
    }

    fn tmp_path(name: &str) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "ofgpu_restart_test_{name}_{}.mcr",
            std::process::id()
        ));
        p
    }

    #[test]
    fn hash_is_deterministic_and_sensitive_to_geometry() {
        let m1 = tiny_mesh();
        let m2 = tiny_mesh();
        assert_eq!(mesh_hash(&m1), mesh_hash(&m2));

        let mut m3 = tiny_mesh();
        m3.c[1].x = 9.0;
        assert_ne!(mesh_hash(&m1), mesh_hash(&m3));

        let mut m4 = tiny_mesh();
        m4.patches[0].name = "inlet".to_string();
        assert_ne!(mesh_hash(&m1), mesh_hash(&m4));
    }

    fn round_trip_case(kind: FieldKind, name: &str) -> (RestartData, RestartData) {
        let mesh = tiny_mesh();
        let hash = mesh_hash(&mesh);
        let (n_internal, n_boundary, n_cells) = (
            mesh.n_internal_faces as u64,
            mesh.n_boundary_faces as u64,
            mesh.n_cells as u64,
        );
        let (ilen, blen) = kind.expected_lens(n_cells, n_internal, n_boundary);
        let internal: Vec<f64> = (0..ilen).map(|i| i as f64 * 1.5 + 0.25).collect();
        let boundary: Vec<f64> = (0..blen).map(|i| -(i as f64) * 2.5 - 0.75).collect();

        let data = RestartData {
            mesh_hash: hash,
            time: 12.5,
            p0: 101325.0,
            dp0dt: -3.5,
            n_cells,
            n_internal,
            n_boundary,
            fields: vec![RestartField {
                name: name.to_string(),
                kind,
                internal,
                boundary,
            }],
        };

        let path = tmp_path(name);
        write_restart(&path, &data).expect("write");
        let back = read_restart(&path, hash).expect("read");
        let _ = std::fs::remove_file(&path);
        (data, back)
    }

    #[test]
    fn round_trip_is_bit_exact_cell_scalar() {
        let (a, b) = round_trip_case(FieldKind::CellScalar, "p");
        assert_eq!(a.time, b.time);
        assert_eq!(a.p0, b.p0);
        assert_eq!(a.dp0dt, b.dp0dt);
        assert_eq!(a.mesh_hash, b.mesh_hash);
        assert_eq!(a.fields[0].internal, b.fields[0].internal);
        assert_eq!(a.fields[0].boundary, b.fields[0].boundary);
        assert_eq!(b.fields[0].kind, FieldKind::CellScalar);
    }

    #[test]
    fn round_trip_is_bit_exact_cell_vector() {
        let (a, b) = round_trip_case(FieldKind::CellVector, "U");
        assert_eq!(a.fields[0].internal, b.fields[0].internal);
        assert_eq!(a.fields[0].boundary, b.fields[0].boundary);
        assert_eq!(b.fields[0].kind, FieldKind::CellVector);
    }

    #[test]
    fn round_trip_is_bit_exact_surface_scalar_phi() {
        let (a, b) = round_trip_case(FieldKind::SurfaceScalar, "phi");
        assert_eq!(a.fields[0].internal, b.fields[0].internal);
        assert_eq!(a.fields[0].boundary, b.fields[0].boundary);
        assert_eq!(b.fields[0].kind, FieldKind::SurfaceScalar);
    }

    #[test]
    fn hash_mismatch_is_refused_naming_both_hashes() {
        let mesh = tiny_mesh();
        let hash = mesh_hash(&mesh);
        let data = RestartData {
            mesh_hash: hash,
            time: 0.0,
            p0: 0.0,
            dp0dt: 0.0,
            n_cells: mesh.n_cells as u64,
            n_internal: mesh.n_internal_faces as u64,
            n_boundary: mesh.n_boundary_faces as u64,
            fields: vec![],
        };
        let path = tmp_path("hash_mismatch");
        write_restart(&path, &data).expect("write");

        let wrong_hash = hash ^ 0xdead_beef;
        let err = read_restart(&path, wrong_hash).unwrap_err();
        let s = err.to_string();
        assert!(s.contains("different mesh"), "{s}");
        assert!(s.contains(&format!("{hash:016x}")), "{s}");
        assert!(s.contains(&format!("{wrong_hash:016x}")), "{s}");

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn truncated_file_is_refused_with_a_clear_error() {
        let mesh = tiny_mesh();
        let hash = mesh_hash(&mesh);
        let data = RestartData {
            mesh_hash: hash,
            time: 1.0,
            p0: 2.0,
            dp0dt: 0.0,
            n_cells: mesh.n_cells as u64,
            n_internal: mesh.n_internal_faces as u64,
            n_boundary: mesh.n_boundary_faces as u64,
            fields: vec![RestartField {
                name: "p".to_string(),
                kind: FieldKind::CellScalar,
                internal: vec![1.0, 2.0],
                boundary: vec![3.0, 4.0],
            }],
        };
        let path = tmp_path("truncated");
        write_restart(&path, &data).expect("write");

        // Chop the file down to just past the header - well before all the
        // field payload has been written.
        let full = std::fs::read(&path).expect("read back");
        let short_path = tmp_path("truncated_short");
        std::fs::File::create(&short_path)
            .unwrap()
            .write_all(&full[..full.len() - 10])
            .unwrap();

        let err = read_restart(&short_path, hash).unwrap_err();
        assert!(err.to_string().contains("truncated"), "{err}");

        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(&short_path);
    }

    #[test]
    fn future_version_is_refused_naming_it() {
        let mesh = tiny_mesh();
        let hash = mesh_hash(&mesh);
        let data = RestartData {
            mesh_hash: hash,
            time: 0.0,
            p0: 0.0,
            dp0dt: 0.0,
            n_cells: mesh.n_cells as u64,
            n_internal: mesh.n_internal_faces as u64,
            n_boundary: mesh.n_boundary_faces as u64,
            fields: vec![],
        };
        let path = tmp_path("future_version");
        write_restart(&path, &data).expect("write");

        // Patch the version field (bytes 8..12) to one past what this build
        // writes (VERSION = 2, so 3 is "future" regardless of when this
        // format is next extended).
        let mut bytes = std::fs::read(&path).expect("read back");
        bytes[8..12].copy_from_slice(&3u32.to_le_bytes());
        std::fs::write(&path, &bytes).expect("rewrite");

        let err = read_restart(&path, hash).unwrap_err();
        let s = err.to_string();
        assert!(s.contains("version 3"), "{s}");

        let _ = std::fs::remove_file(&path);
    }
}
