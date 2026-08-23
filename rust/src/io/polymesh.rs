// meteor-cfd - Copyright (c) 2026 주식회사 메테오시뮬레이션 (Meteo Simulation Co., Ltd.)
// Source-available, not Open Source. Educational use is free; research,
// publication and commercial use require a licence - simul@msimul.com
// See LICENSE at the repository root.

//! `constant/polyMesh` -> [`HostMesh`].
//!
//! Provenance: carried across from this project's own earlier C++ I/O layer
//! when the crate moved to Rust. That C++ was written from the case format as
//! it appears in data files - not from any CFD code's source - and the format
//! itself, not another program, is the specification here. No GPL-licensed
//! source was consulted.
//!
//! Nothing here links against OpenFOAM: a case has to be runnable on a machine
//! that has never had a FOAM installation, so this reader owns the file
//! formats itself.
//!
//! Three things are worth knowing before reading the code.
//!
//! * [`PatchInfo::start`] is an offset into the FLATTENED boundary arrays
//!   (`startFace - nInternalFaces`), never a global face index. Every boundary
//!   kernel indexes with that convention, so the conversion happens once, here.
//! * polyMesh stores the internal faces first and in upper-triangular order
//!   (`owner[f] < neighbour[f]`, sorted by owner then neighbour). That is
//!   exactly `lduAddressing` (SPEC section 1), and every gather kernel assumes
//!   it, so [`build_host_mesh`] refuses a mesh that violates it rather than
//!   assembling a silently mis-addressed matrix.
//! * A cyclic pair is written face for face: face `k` of one patch is the
//!   partner of face `k` of the other. The cell across a couple is therefore a
//!   direct index - no search, no geometric matching.
//!
//! The geometry itself is not computed here; `mesh/geometry.rs` owns it.

use std::path::{Path, PathBuf};

use crate::error::{parse_err, Error, Result};
use crate::io::tokenizer::{self, Tok, Tokenizer};
use crate::mesh::{HostMesh, PatchInfo, PatchKind};
use crate::{Label, Scalar, Vec3};

// ==========================================================================
//  PolyMeshRaw
// ==========================================================================

/// `constant/polyMesh` as it is stored on disk, before any geometry.
#[derive(Debug, Default, Clone)]
pub struct PolyMeshRaw {
    pub points: Vec<Vec3>,
    /// ALL faces, internal first.
    pub faces: Vec<Vec<Label>>,
    /// `[n_faces]`
    pub owner: Vec<Label>,
    /// `[n_internal_faces]`
    pub neighbour: Vec<Label>,
    /// `PatchInfo::start` is the offset into the FLATTENED boundary arrays,
    /// i.e. `startFace - nInternalFaces`.
    pub patches: Vec<PatchInfo>,
}

// ==========================================================================
//  Reading
// ==========================================================================

/// Read a polyMesh.
///
/// `case_dir` may be the case root, the `constant` directory, or the
/// `polyMesh` directory itself; all three are probed, because every caller has
/// a different idea of which one it is holding.
pub fn read_poly_mesh(case_dir: &Path) -> Result<PolyMeshRaw> {
    let dir = find_poly_mesh_dir(case_dir)?;

    let mut raw = PolyMeshRaw {
        points: read_points_file(&dir.join("points"))?,
        faces: read_faces_file(&dir.join("faces"))?,
        owner: read_label_list_file(&dir.join("owner"))?,
        neighbour: read_label_list_file(&dir.join("neighbour"))?,
        patches: Vec::new(),
    };

    let n_faces = raw.faces.len();
    let n_if = raw.neighbour.len();

    if raw.owner.len() != n_faces {
        return Err(Error::Mesh(format!(
            "{}: owner has {} entries but faces has {}",
            dir.display(),
            raw.owner.len(),
            n_faces
        )));
    }
    if n_if > n_faces {
        return Err(Error::Mesh(format!(
            "{}: neighbour has {} entries, more than the {} faces",
            dir.display(),
            n_if,
            n_faces
        )));
    }

    // ---- patches ---------------------------------------------------------
    let bnd = dir.join("boundary");
    let pe = read_boundary_file(&bnd)?;

    let mut patches: Vec<PatchInfo> = Vec::with_capacity(pe.len());
    let mut expected_start = n_if as i64;

    for r in &pe {
        // The boundary faces have to be contiguous and in patch order for
        // `start` to mean anything; polyMesh always writes them that way, so
        // a gap means the mesh, not the reader, is inconsistent.
        if i64::from(r.start_face) != expected_start {
            return parse_err(
                &bnd,
                format!(
                    "patch '{}' starts at face {} but the previous patch ends \
                     at {}. ofgpu needs the boundary faces contiguous and in \
                     patch order, which is how polyMesh always stores them - \
                     the mesh looks inconsistent.",
                    r.name, r.start_face, expected_start
                ),
            );
        }
        if r.n_faces < 0
            || i64::from(r.start_face) + i64::from(r.n_faces) > n_faces as i64
        {
            return parse_err(
                &bnd,
                format!(
                    "patch '{}' runs off the end of the face list ({} + {} > {})",
                    r.name, r.start_face, r.n_faces, n_faces
                ),
            );
        }

        patches.push(PatchInfo {
            name: r.name.clone(),
            type_name: r.type_name.clone(),
            kind: PatchKind::from_type(&r.type_name),
            start: (r.start_face as usize) - n_if,
            size: r.n_faces as usize,
            nbr_patch: None,
        });

        expected_start += i64::from(r.n_faces);
    }

    if expected_start != n_faces as i64 {
        return parse_err(
            &bnd,
            format!(
                "the patches cover faces {}..{} but the mesh has {} faces",
                n_if, expected_start, n_faces
            ),
        );
    }

    // ---- resolve a cyclic's neighbourPatch into an index ------------------
    for k in 0..patches.len() {
        if patches[k].kind != PatchKind::Cyclic {
            continue;
        }

        let nbr = pe[k].neighbour_patch.clone();
        if nbr.is_empty() {
            return parse_err(
                &bnd,
                format!(
                    "cyclic patch '{}' has no neighbourPatch entry",
                    patches[k].name
                ),
            );
        }

        let Some(j) = patches.iter().position(|p| p.name == nbr) else {
            return parse_err(
                &bnd,
                format!(
                    "cyclic patch '{}' names neighbourPatch '{}', which does \
                     not exist",
                    patches[k].name, nbr
                ),
            );
        };
        if patches[j].size != patches[k].size {
            return parse_err(
                &bnd,
                format!(
                    "cyclic patches '{}' and '{}' have different face counts",
                    patches[k].name, nbr
                ),
            );
        }
        patches[k].nbr_patch = Some(j);
    }

    raw.patches = patches;
    Ok(raw)
}

fn find_poly_mesh_dir(case_dir: &Path) -> Result<PathBuf> {
    let tried = [
        case_dir.join("constant").join("polyMesh"),
        case_dir.join("polyMesh"),
        case_dir.to_path_buf(),
    ];

    for d in &tried {
        // A `.gz` counts as found so the user gets the tokeniser's "gunzip the
        // case" message rather than a misleading "there is no polyMesh here".
        if d.join("points").is_file() || d.join("points.gz").is_file() {
            return Ok(d.clone());
        }
    }

    let mut msg = format!(
        "cannot find a polyMesh under '{}'. Tried:",
        case_dir.display()
    );
    for d in &tried {
        msg += &format!("\n    {}", d.join("points").display());
    }
    Err(Error::Mesh(msg))
}

// ==========================================================================
//  buildHostMesh
// ==========================================================================

/// Topology plus the flattened boundary arrays, then the geometry.
///
/// `n_cells` is derived from the addressing (`max(owner, neighbour) + 1`)
/// exactly as OpenFOAM derives it - polyMesh never stores a cell count.
pub fn build_host_mesh(raw: &PolyMeshRaw) -> Result<HostMesh> {
    let n_faces = raw.faces.len();
    let n_if = raw.neighbour.len();

    if raw.owner.len() != n_faces {
        return Err(Error::Mesh(format!(
            "build_host_mesh: owner has {} entries but there are {} faces",
            raw.owner.len(),
            n_faces
        )));
    }
    if n_if > n_faces {
        return Err(Error::Mesh(format!(
            "build_host_mesh: {n_if} internal faces out of {n_faces} total"
        )));
    }
    let n_bf = n_faces - n_if;

    // ---- n_cells = max(owner, neighbour) + 1 -----------------------------
    // Accumulated as i64 so a corrupt label near `Label::MAX` is rejected
    // rather than overflowing the count on the way to being rejected.
    let mut n_cells: i64 = 0;
    for (f, &o) in raw.owner.iter().enumerate() {
        if o < 0 {
            return Err(Error::Mesh(format!(
                "build_host_mesh: face {f} has negative owner {o}"
            )));
        }
        n_cells = n_cells.max(i64::from(o) + 1);
    }
    for (f, &n) in raw.neighbour.iter().enumerate() {
        if n < 0 {
            return Err(Error::Mesh(format!(
                "build_host_mesh: face {f} has negative neighbour {n}"
            )));
        }
        n_cells = n_cells.max(i64::from(n) + 1);
    }
    if n_cells > i64::from(Label::MAX) {
        return Err(Error::Mesh(format!(
            "build_host_mesh: {n_cells} cells does not fit in a 32-bit label"
        )));
    }
    let n_cells = n_cells as usize;

    let mut m = HostMesh {
        n_cells,
        n_internal_faces: n_if,
        n_boundary_faces: n_bf,
        n_points: raw.points.len(),
        ..HostMesh::default()
    };

    // ---- lduAddressing ---------------------------------------------------
    m.owner = raw.owner[..n_if].to_vec();
    m.neighbour = raw.neighbour.clone();

    // SPEC section 1 relies on owner[f] < neighbour[f] (upper-triangular
    // order); every gather kernel would silently mis-address otherwise.
    for f in 0..n_if {
        if m.owner[f] >= m.neighbour[f] {
            return Err(Error::Mesh(format!(
                "build_host_mesh: face {} has owner {} >= neighbour {}; \
                 polyMesh is not in upper-triangular order",
                f, m.owner[f], m.neighbour[f]
            )));
        }
    }

    // ---- flattened boundary ----------------------------------------------
    m.patches = raw.patches.clone();
    m.b_face_cells = raw.owner[n_if..].to_vec();

    m.b_kind = vec![PatchKind::Generic as Label; n_bf];
    m.b_patch = vec![-1; n_bf];
    m.b_nbr_cell = vec![-1; n_bf];
    // A non-coupled boundary weight is 1 (SPEC section 2). Cyclic weights come
    // out of the geometry, so compute_geometry() overwrites those.
    m.b_weights = vec![1.0; n_bf];

    for (p, pi) in raw.patches.iter().enumerate() {
        // A hand-built PolyMeshRaw would panic on the index below rather than
        // read past the end, so the range is checked first.
        if pi.start + pi.size > n_bf {
            return Err(Error::Mesh(format!(
                "build_host_mesh: patch '{}' covers boundary faces {}..{} but \
                 there are only {}",
                pi.name,
                pi.start,
                pi.start + pi.size,
                n_bf
            )));
        }
        for k in 0..pi.size {
            m.b_kind[pi.start + k] = pi.kind as Label;
            m.b_patch[pi.start + k] = p as Label;
        }
    }

    // ---- cyclic pairing --------------------------------------------------
    for pi in raw.patches.iter() {
        if pi.kind != PatchKind::Cyclic {
            continue;
        }
        let Some(n) = pi.nbr_patch else { continue };
        let Some(pn) = raw.patches.get(n) else {
            return Err(Error::Mesh(format!(
                "build_host_mesh: cyclic patch '{}' names patch {} of {}",
                pi.name,
                n,
                raw.patches.len()
            )));
        };
        if pn.size != pi.size {
            return Err(Error::Mesh(format!(
                "build_host_mesh: cyclic patches '{}' and '{}' have different \
                 sizes",
                pi.name, pn.name
            )));
        }
        for k in 0..pi.size {
            m.b_nbr_cell[pi.start + k] = m.b_face_cells[pn.start + k];
        }
    }

    // ---- face vertices ---------------------------------------------------
    // compute_geometry indexes `points` with these. A bad label would panic
    // deep inside the geometry instead of naming the file's actual problem.
    for (f, fv) in raw.faces.iter().enumerate() {
        for &v in fv.iter() {
            if v < 0 || v as usize >= raw.points.len() {
                return Err(Error::Mesh(format!(
                    "build_host_mesh: face {} refers to point {} but there are \
                     {} points",
                    f,
                    v,
                    raw.points.len()
                )));
            }
        }
    }

    // ---- size the geometry arrays, then let mesh::geometry fill them ------
    m.v = vec![0.0; n_cells];
    m.c = vec![Vec3::ZERO; n_cells];

    m.sf = vec![Vec3::ZERO; n_if];
    m.mag_sf = vec![0.0; n_if];
    m.cf = vec![Vec3::ZERO; n_if];
    m.weights = vec![0.0; n_if];
    m.delta_coeffs = vec![0.0; n_if];
    m.non_orth_corr = vec![Vec3::ZERO; n_if];

    m.b_sf = vec![Vec3::ZERO; n_bf];
    m.b_mag_sf = vec![0.0; n_bf];
    m.b_cf = vec![Vec3::ZERO; n_bf];
    m.b_delta_coeffs = vec![0.0; n_bf];
    m.b_y = vec![0.0; n_bf];

    m.compute_geometry(&raw.points, &raw.faces)?;
    m.build_cell_face_maps();

    Ok(m)
}

// ==========================================================================
//  The four files
// ==========================================================================

/// Slurp, refuse a binary file before it is shredded into nonsense tokens,
/// tokenise, and drop the `FoamFile` header - which is what disposes of the
/// `note` entry blockMesh writes into `owner` and `neighbour`.
fn open(path: &Path) -> Result<Tokenizer> {
    let text = tokenizer::slurp(path)?;
    let name = path.display().to_string();
    tokenizer::check_ascii_format(&text, &name)?;

    let mut ts = Tokenizer::new(&text, &name);
    ts.skip_header()?;
    Ok(ts)
}

/// `N ( (x y z) ... )`, or a bare `( ... )` when the writer omitted the count.
fn read_points_file(path: &Path) -> Result<Vec<Vec3>> {
    let mut ts = open(path)?;
    let mut pts = Vec::new();

    if ts.is_punct('(') {
        ts.next()?;
        while !ts.done() && !ts.is_punct(')') {
            pts.push(parse_vec3(&mut ts)?);
        }
        ts.expect_punct(')')?;
    } else {
        let n = ts.expect_label()?;
        if n < 0 {
            return ts.err("negative point count");
        }
        ts.expect_punct('(')?;
        pts.reserve(n as usize);
        for _ in 0..n {
            pts.push(parse_vec3(&mut ts)?);
        }
        ts.expect_punct(')')?;
    }

    ts.check_scan_error()?;
    Ok(pts)
}

/// Both the compact `4(a b c d)` and the long `4 (a b c d)` tokenise the same
/// way - a token that starts with a digit is a number, so the `4` and the `(`
/// always arrive separately - which is why one loop covers both forms.
fn read_faces_file(path: &Path) -> Result<Vec<Vec<Label>>> {
    let mut ts = open(path)?;

    let nf = ts.expect_label()?;
    if nf < 0 {
        return ts.err("negative face count");
    }
    ts.expect_punct('(')?;

    let mut faces: Vec<Vec<Label>> = Vec::with_capacity(nf as usize);
    for _ in 0..nf {
        let np = ts.expect_label()?;
        if np < 0 {
            return ts.err("negative vertex count");
        }
        ts.expect_punct('(')?;
        let mut fv: Vec<Label> = Vec::with_capacity(np as usize);
        for _ in 0..np {
            fv.push(ts.expect_label()?);
        }
        ts.expect_punct(')')?;
        faces.push(fv);
    }

    ts.expect_punct(')')?;
    ts.check_scan_error()?;
    Ok(faces)
}

/// `owner` / `neighbour`: `N ( i i i ... )`, OpenFOAM's all-equal short form
/// `N{i}`, or a bare `( ... )`.
fn read_label_list_file(path: &Path) -> Result<Vec<Label>> {
    let mut ts = open(path)?;
    let mut v = Vec::new();

    if ts.is_punct('(') {
        ts.next()?;
        while !ts.done() && !ts.is_punct(')') {
            v.push(ts.expect_label()?);
        }
        ts.expect_punct(')')?;
        ts.check_scan_error()?;
        return Ok(v);
    }

    let n = ts.expect_label()?;
    if n < 0 {
        return ts.err("negative list size");
    }

    if ts.is_punct('{') {
        ts.next()?;
        let a = ts.expect_label()?;
        ts.expect_punct('}')?;
        ts.check_scan_error()?;
        return Ok(vec![a; n as usize]);
    }

    ts.expect_punct('(')?;
    v.reserve(n as usize);
    for _ in 0..n {
        v.push(ts.expect_label()?);
    }
    ts.expect_punct(')')?;
    ts.check_scan_error()?;
    Ok(v)
}

#[derive(Debug, Default)]
struct RawPatchEntry {
    name: String,
    type_name: String,
    neighbour_patch: String,
    n_faces: Label,
    start_face: Label,
    have_n_faces: bool,
    have_start_face: bool,
}

/// `N ( name { type ...; nFaces N; startFace N; } ... )`.
///
/// Anything that is not `type`, `neighbourPatch`, `nFaces` or `startFace` is
/// skipped, sub-dictionaries included - real boundary files carry `inGroups`,
/// `transform`, `matchTolerance` and friends.
fn read_boundary_file(path: &Path) -> Result<Vec<RawPatchEntry>> {
    let mut ts = open(path)?;

    let np = ts.expect_label()?;
    if np < 0 {
        return ts.err("negative patch count");
    }
    ts.expect_punct('(')?;

    let mut out: Vec<RawPatchEntry> = Vec::with_capacity(np as usize);

    for _ in 0..np {
        let mut e = RawPatchEntry {
            name: ts.expect_word()?,
            ..RawPatchEntry::default()
        };
        ts.expect_punct('{')?;

        while !ts.done() && !ts.is_punct('}') {
            if ts.is_punct(';') {
                ts.next()?;
                continue;
            }
            if ts.peek_at(0).is_some_and(|t| t.is_punct_any()) {
                return ts.err("expected a keyword");
            }

            let k = ts.expect_word()?;
            if ts.is_punct('{') {
                ts.skip_entry()?;
                continue;
            }

            match k.as_str() {
                "type" => e.type_name = gather_raw(&mut ts)?,
                "neighbourPatch" => e.neighbour_patch = gather_raw(&mut ts)?,
                "nFaces" => {
                    e.n_faces = ts.expect_label()?;
                    ts.expect_punct(';')?;
                    e.have_n_faces = true;
                }
                "startFace" => {
                    e.start_face = ts.expect_label()?;
                    ts.expect_punct(';')?;
                    e.have_start_face = true;
                }
                _ => ts.skip_entry()?,
            }
        }
        ts.expect_punct('}')?;

        if !e.have_n_faces || !e.have_start_face {
            return parse_err(
                path,
                format!("patch '{}' is missing nFaces/startFace", e.name),
            );
        }
        out.push(e);
    }

    ts.expect_punct(')')?;
    ts.check_scan_error()?;
    Ok(out)
}

// ==========================================================================
//  Token-level helpers
// ==========================================================================

fn parse_vec3(ts: &mut Tokenizer) -> Result<Vec3> {
    ts.expect_punct('(')?;
    let x = ts.expect_num()? as Scalar;
    let y = ts.expect_num()? as Scalar;
    let z = ts.expect_num()? as Scalar;
    ts.expect_punct(')')?;
    Ok(Vec3::new(x, y, z))
}

/// Capture an entry's tokens verbatim, consuming the `;`.
///
/// The spacing rule is OpenFOAM's, so a value reads back the way it was
/// written (`uniform (0 0 0)`, not `uniform ( 0 0 0 )`) - `type` and
/// `neighbourPatch` are single tokens, but the same helper has to survive the
/// odd `type            cyclic;` written with a trailing group.
fn gather_raw(ts: &mut Tokenizer) -> Result<String> {
    let mut v = String::new();

    while !ts.done() && !ts.is_punct(';') {
        if ts.is_punct('}') {
            return ts.err("missing ';'");
        }
        let Some(t) = ts.next()? else { break };
        append_raw(&mut v, &t);
    }
    if ts.done() {
        return ts.err("expected ';'");
    }
    ts.next()?;
    Ok(v)
}

fn append_raw(v: &mut String, t: &Tok) {
    let opens_group = v.ends_with('(') || v.ends_with('[');
    let closes_group = matches!(t, Tok::Punct(')') | Tok::Punct(']'));
    if !v.is_empty() && !opens_group && !closes_group {
        v.push(' ');
    }
    v.push_str(&t.to_string());
}

// ==========================================================================
//  Tests
// ==========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    // A 2x1x1 box of unit cells. Cell 0 spans x in [0,1], cell 1 x in [1,2],
    // with one internal face at x = 1. y is a cyclic pair, z is `empty`, and a
    // zero-size patch is thrown in because real cases grow them.
    const POINTS: &str = r#"
FoamFile
{
    version     2.0;
    format      ascii;
    class       vectorField;
    location    "constant/polyMesh";
    object      points;
}

12
(
(0 0 0)
(1 0 0)
(2 0 0)
(0 1 0)
(1 1 0)
(2 1 0)
(0 0 1)
(1 0 1)
(2 0 1)
(0 1 1)
(1 1 1)
(2 1 1)
)
"#;

    // Deliberately alternates the compact `4(a b c d)` and the long
    // `4 (a b c d)` forms - both occur in the wild and they have to parse
    // identically. Every face is wound so that its normal points out of its
    // owner, which is what makes the geometry close.
    const FACES: &str = r#"
FoamFile
{
    version     2.0;
    format      ascii;
    class       faceList;
    location    "constant/polyMesh";
    object      faces;
}

11
(
4(1 4 10 7)
4 (0 6 9 3)
4(2 5 11 8)
4 (0 1 7 6)
4(1 2 8 7)
4 (3 9 10 4)
4(4 10 11 5)
4 (0 3 4 1)
4(1 4 5 2)
4 (6 7 10 9)
4(7 8 11 10)
)
"#;

    // The `note` entry has to be ignored; blockMesh always writes one.
    const OWNER: &str = r#"
FoamFile
{
    version     2.0;
    format      ascii;
    class       labelList;
    note        "nPoints:12  nCells:2  nFaces:11  nInternalFaces:1";
    location    "constant/polyMesh";
    object      owner;
}

11
(
0 0 1 0 1 0 1 0 1 0 1
)
"#;

    const NEIGHBOUR: &str = r#"
FoamFile
{
    version     2.0;
    format      ascii;
    class       labelList;
    note        "nPoints:12  nCells:2  nFaces:11  nInternalFaces:1";
    location    "constant/polyMesh";
    object      neighbour;
}

1
(
1
)
"#;

    const BOUNDARY: &str = r#"
FoamFile
{
    version     2.0;
    format      ascii;
    class       polyBoundaryMesh;
    location    "constant/polyMesh";
    object      boundary;
}

6
(
    inlet
    {
        type            patch;
        nFaces          1;
        startFace       1;
    }
    outlet
    {
        type            patch;
        nFaces          1;
        startFace       2;
    }
    cyc_lo
    {
        type            cyclic;
        inGroups        1(cyclic);
        nFaces          2;
        startFace       3;
        matchTolerance  0.0001;
        transform       translational;
        separationVector (0 -1 0);
        neighbourPatch  cyc_hi;
    }
    cyc_hi
    {
        type            cyclic;
        inGroups        1(cyclic);
        nFaces          2;
        startFace       5;
        neighbourPatch  cyc_lo;
    }
    frontAndBack
    {
        type            empty;
        inGroups        1(empty);
        nFaces          4;
        startFace       7;
    }
    unused
    {
        type            patch;
        nFaces          0;
        startFace       11;
    }
)
"#;

    fn scratch(tag: &str) -> PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let mut d = std::env::temp_dir();
        d.push(format!(
            "ofgpu-polymesh-{}-{}-{}",
            tag,
            std::process::id(),
            nanos
        ));
        fs::create_dir_all(&d).unwrap();
        d
    }

    fn write_case(root: &Path, boundary: &str) {
        let d = root.join("constant").join("polyMesh");
        fs::create_dir_all(&d).unwrap();
        fs::write(d.join("points"), POINTS).unwrap();
        fs::write(d.join("faces"), FACES).unwrap();
        fs::write(d.join("owner"), OWNER).unwrap();
        fs::write(d.join("neighbour"), NEIGHBOUR).unwrap();
        fs::write(d.join("boundary"), boundary).unwrap();
    }

    fn expected_faces() -> Vec<Vec<Label>> {
        vec![
            vec![1, 4, 10, 7],
            vec![0, 6, 9, 3],
            vec![2, 5, 11, 8],
            vec![0, 1, 7, 6],
            vec![1, 2, 8, 7],
            vec![3, 9, 10, 4],
            vec![4, 10, 11, 5],
            vec![0, 3, 4, 1],
            vec![1, 4, 5, 2],
            vec![6, 7, 10, 9],
            vec![7, 8, 11, 10],
        ]
    }

    #[test]
    fn reads_both_face_forms_and_resolves_the_cyclic_pair() -> Result<()> {
        let root = scratch("read");
        write_case(&root, BOUNDARY);

        let raw = read_poly_mesh(&root)?;

        assert_eq!(raw.points.len(), 12);
        assert_eq!(raw.points[11], Vec3::new(2.0, 1.0, 1.0));
        // The compact and the long face form must yield the same lists.
        assert_eq!(raw.faces, expected_faces());
        assert_eq!(raw.owner, vec![0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]);
        assert_eq!(raw.neighbour, vec![1]);

        assert_eq!(raw.patches.len(), 6);
        let starts: Vec<usize> = raw.patches.iter().map(|p| p.start).collect();
        let sizes: Vec<usize> = raw.patches.iter().map(|p| p.size).collect();
        // startFace - nInternalFaces, not the global face index.
        assert_eq!(starts, vec![0, 1, 2, 4, 6, 10]);
        assert_eq!(sizes, vec![1, 1, 2, 2, 4, 0]);

        assert_eq!(raw.patches[2].nbr_patch, Some(3));
        assert_eq!(raw.patches[3].nbr_patch, Some(2));
        assert_eq!(raw.patches[0].nbr_patch, None);
        assert_eq!(raw.patches[4].kind, PatchKind::Empty);
        assert_eq!(raw.patches[4].type_name, "empty");
        assert_eq!(raw.patches[5].size, 0);

        let _ = fs::remove_dir_all(&root);
        Ok(())
    }

    #[test]
    fn probes_case_root_constant_and_polymesh() -> Result<()> {
        let root = scratch("probe");
        write_case(&root, BOUNDARY);

        for d in [
            root.clone(),
            root.join("constant"),
            root.join("constant").join("polyMesh"),
        ] {
            let raw = read_poly_mesh(&d)?;
            assert_eq!(raw.faces.len(), 11, "probing failed from {}", d.display());
            assert_eq!(raw.patches.len(), 6);
        }

        let _ = fs::remove_dir_all(&root);
        Ok(())
    }

    /// A gap in the boundary shifts every later `start` by the size of the gap
    /// and silently mis-addresses every boundary face, so it has to be fatal.
    #[test]
    fn rejects_non_contiguous_patches() {
        let root = scratch("gap");
        let bad = BOUNDARY.replace("startFace       2;", "startFace       3;");
        write_case(&root, &bad);

        assert!(
            read_poly_mesh(&root).is_err(),
            "a boundary with a gap must not be accepted"
        );

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn rejects_a_cyclic_naming_a_patch_that_does_not_exist() {
        let root = scratch("nbr");
        let bad = BOUNDARY.replace("neighbourPatch  cyc_hi;", "neighbourPatch  nope;");
        write_case(&root, &bad);

        assert!(read_poly_mesh(&root).is_err());

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn rejects_a_binary_file_before_tokenising_it() {
        let root = scratch("binary");
        write_case(&root, BOUNDARY);
        let pts = root.join("constant").join("polyMesh").join("points");
        fs::write(&pts, POINTS.replace("format      ascii;", "format      binary;")).unwrap();

        let msg = match read_poly_mesh(&root) {
            Err(e) => e.to_string(),
            Ok(_) => panic!("a binary file must be refused"),
        };
        assert!(msg.contains("foamFormatConvert"), "unhelpful message: {msg}");

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn builds_the_flattened_boundary_and_the_cyclic_addressing() -> Result<()> {
        let root = scratch("build");
        write_case(&root, BOUNDARY);

        let m = build_host_mesh(&read_poly_mesh(&root)?)?;

        assert_eq!(m.n_cells, 2);
        assert_eq!(m.n_internal_faces, 1);
        assert_eq!(m.n_boundary_faces, 10);
        assert_eq!(m.n_points, 12);

        assert_eq!(m.owner, vec![0]);
        assert_eq!(m.neighbour, vec![1]);
        assert_eq!(m.b_face_cells, vec![0, 1, 0, 1, 0, 1, 0, 1, 0, 1]);
        assert_eq!(m.b_patch, vec![0, 1, 2, 2, 3, 3, 4, 4, 4, 4]);

        let empty = PatchKind::Empty as Label;
        assert_eq!(&m.b_kind[6..10], &[empty, empty, empty, empty]);

        // Face k of cyc_lo pairs with face k of cyc_hi, so the cell across is
        // a direct index. Non-cyclic faces stay at -1.
        assert_eq!(m.b_nbr_cell, vec![-1, -1, 0, 1, 0, 1, -1, -1, -1, -1]);

        // Two unit cells; the geometry has to agree or something upstream of
        // the solver is already wrong.
        assert!((m.v.iter().sum::<Scalar>() - 2.0).abs() < 1e-12);

        let _ = fs::remove_dir_all(&root);
        Ok(())
    }

    /// Every gather kernel indexes on `owner[f] < neighbour[f]`; a mesh in the
    /// wrong order would produce a wrong matrix, not a crash.
    #[test]
    fn rejects_a_mesh_that_is_not_upper_triangular() {
        let raw = PolyMeshRaw {
            points: vec![Vec3::ZERO; 4],
            faces: vec![vec![0, 1, 2, 3]],
            owner: vec![1],
            neighbour: vec![0],
            patches: Vec::new(),
        };
        assert!(build_host_mesh(&raw).is_err());
    }
}
