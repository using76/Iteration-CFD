// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! Decomposed I/O (SPEC-LIT 139.7, PAR-15): one rank's fields under
//! `processorN/`, the reconstruct that gathers a set back to the whole mesh,
//! and the restart that reads a set back into a run.
//!
//! # What is written where
//!
//! Rank `N` of a run on `P` ranks writes, under the case directory:
//!
//! ```text
//! processorN/decomposition                        the set's shape
//! processorN/constant/polyMesh/cellProcAddressing the rank's cells, whole-mesh order
//! processorN/<time>/U    volVectorField           owned cells; every patch's boundary faces
//! processorN/<time>/p    volScalarField           the same
//! processorN/<time>/phi  surfaceScalarField       the rank's internal faces; every patch
//! ```
//!
//! Every value is an OWNER's: the halo is refilled, never written. Every
//! patch of the part - processor and empty included - carries its slice of
//! the boundary-face array, so every boundary value comes back bit for bit.
//! [`reconstruct`] gathers a set to the whole mesh and writes the case
//! fields `<time>/U` and `<time>/p` (post-processing only) and one
//! `VTK/reconstructed_<time>.vtu`. `phi` is written per rank and restored
//! on restart; it is NOT reconstructed, because a reconstructed case is for
//! post-processing while a restart reads the processor set, where the flux
//! still lives.
//!
//! # Why bitwise
//!
//! Every number is written at [`ROUND_TRIP_DIGITS`] with no uniform
//! collapsing - the precision at which a written value reads back BIT FOR
//! BIT (SPEC-LIT 118.6) - so a restart's fields are the writer's fields,
//! not their six-digit rounding. A rank's own file types every patch
//! `calculated`, the type the case writer gives a `value` on every patch,
//! so the processor and empty patches' values are written too; the part's
//! patch kinds come from the decomposition, never from the file. A restart
//! writes only OWNED values with
//! the halo left zero and lets `Simple::initialise`'s exchange refill the
//! halo from the owners (SPEC-LIT 139.4).
//!
//! # The refusal table
//!
//! [`check_processor_set`] runs FIRST, host only and nothing collective -
//! every rank reads the same disk, so every rank refuses identically:
//!
//! | what is on disk | refusal |
//! |---|---|
//! | no `processor0` | "holds no processor set (no processor0)" |
//! | a gap after a consecutive run `processor0..processor{k-1}` | "processor{k} is missing" |
//! | `k != n_parts` | "a processorN set from a different P is refused" |
//! | `decomposition` disagrees with its directory | names the file and both numbers |
//! | `nGlobalCells` differs | names both numbers |
//! | `cellProcAddressing` cut by another partitioner | names the first differing cell |
//! | no `processorN/<time>` | "does not exist" |
//!
//! Provenance: ORIGINAL - the processorN layout is the case format as it
//! appears in data files, not any program's source. No external source. No
//! GPL-licensed source was consulted.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use crate::decompose::Decomposition;
use crate::error::{Error, IoContext, Result};
use crate::io::dict::FoamDict;
use crate::io::fields::{
    read_scalar_field, read_vector_field, write_scalar_field_prec,
    write_surface_scalar_field_prec, write_vector_field_prec, PatchFieldSpec, RawScalarField,
    RawVectorField, ROUND_TRIP_DIGITS,
};
use crate::io::output_types::OutputField;
use crate::io::tokenizer::{check_ascii_format, slurp, Tok, Tokenizer};
use crate::io::vtu::write_vtu;
use crate::mesh::{HostMesh, PatchInfo, PatchKind};
use crate::{Label, Scalar, Vec3};

/// The directory name a rank's fields sit under: `processor<N>`.
pub const PROCESSOR_PREFIX: &str = "processor";

/// The `type_name` `Decomposition` gives a part's `procBoundary{p}to{q}`
/// patches - the faces a cut made, which the whole mesh holds as internal
/// faces.
const PROCESSOR_PATCH_TYPE: &str = "processor";

/// `<case>/processor<part>`.
pub fn processor_dir(case: &Path, part: usize) -> PathBuf {
    case.join(format!("{PROCESSOR_PREFIX}{part}"))
}

/// The dimensions of the three fields this module writes, in the order U, p,
/// phi - the cavity case's own sets.
const U_DIMS: &str = "[0 1 -1 0 0 0 0]";
const P_DIMS: &str = "[0 2 -2 0 0 0 0]";
const PHI_DIMS: &str = "[0 3 -1 0 0 0 0]";

/// The incompressible SIMPLE restart set of one rank, in the PART's local
/// order: owned cells first, then the part's boundary faces and internal
/// faces. Exactly what [`write_rank`] writes and [`read_rank`] reads back.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RankFields {
    /// `[part.mesh.n_cells]` the owned cells' velocity.
    pub u: Vec<Vec3>,
    /// `[part.mesh.n_boundary_faces]` every boundary face of the part,
    /// processor patches included.
    pub u_bf: Vec<Vec3>,
    /// `[part.mesh.n_cells]` the owned cells' pressure.
    pub p: Vec<Scalar>,
    /// `[part.mesh.n_boundary_faces]`.
    pub p_bf: Vec<Scalar>,
    /// `[part.mesh.n_internal_faces]` the flux through the part's internal
    /// faces.
    pub phi_f: Vec<Scalar>,
    /// `[part.mesh.n_boundary_faces]` the flux through every boundary face.
    pub phi_bf: Vec<Scalar>,
}

// ==========================================================================
//  Writing a rank
// ==========================================================================

/// Write rank `part`'s restart set: `decomposition`, `cellProcAddressing`,
/// and the three fields of `time`. The directories are created as needed;
/// an existing set of the same shape is overwritten.
///
/// Every number goes out at [`ROUND_TRIP_DIGITS`] (SPEC-LIT 118.6), so what
/// [`read_rank`] reads back is what came in, bit for bit.
pub fn write_rank(
    case: &Path,
    dec: &Decomposition,
    part: usize,
    time: &str,
    f: &RankFields,
) -> Result<()> {
    let p = dec.parts.get(part).ok_or_else(|| {
        Error::Config(format!(
            "decomposed: this decomposition has {} part(s), part {part} was \
             asked to write",
            dec.n_parts
        ))
    })?;
    let m = &p.mesh;
    check_len(&format!("part {part}'s U"), m.n_cells, f.u.len())?;
    check_len(&format!("part {part}'s U boundary"), m.n_boundary_faces, f.u_bf.len())?;
    check_len(&format!("part {part}'s p"), m.n_cells, f.p.len())?;
    check_len(&format!("part {part}'s p boundary"), m.n_boundary_faces, f.p_bf.len())?;
    check_len(&format!("part {part}'s phi"), m.n_internal_faces, f.phi_f.len())?;
    check_len(&format!("part {part}'s phi boundary"), m.n_boundary_faces, f.phi_bf.len())?;

    let dir = processor_dir(case, part);
    write_decomposition_file(&dir.join("decomposition"), dec, part)?;
    write_cell_proc_addressing(
        &dir.join("constant/polyMesh/cellProcAddressing"),
        &p.global_cell[..m.n_cells],
    )?;

    let tdir = dir.join(time);
    let uf = case_vector_field("U", U_DIMS, &f.u, m, &f.u_bf, true);
    write_vector_field_prec(&tdir.join("U"), &uf, time, ROUND_TRIP_DIGITS, false)?;
    let pf = case_scalar_field("p", P_DIMS, &f.p, m, &f.p_bf, true);
    write_scalar_field_prec(&tdir.join("p"), &pf, time, ROUND_TRIP_DIGITS, false)?;
    let ff = case_scalar_field("phi", PHI_DIMS, &f.phi_f, m, &f.phi_bf, true);
    write_surface_scalar_field_prec(&tdir.join("phi"), &ff, time, ROUND_TRIP_DIGITS, false)?;
    Ok(())
}

fn check_len(what: &str, want: usize, got: usize) -> Result<()> {
    if want == got {
        return Ok(());
    }
    Err(Error::Config(format!(
        "decomposed: {what} has {got} value(s), its part holds {want}"
    )))
}

/// The text of the two LABEL files - `decomposition` and
/// `cellProcAddressing` - which carry no floating point and have no
/// case-field writer to borrow, so they keep this small local one.
fn write_all(path: &Path, text: &str) -> Result<()> {
    if let Some(dir) = path.parent() {
        if !dir.as_os_str().is_empty() {
            std::fs::create_dir_all(dir).path(dir)?;
        }
    }
    // LF everywhere, including on Windows - the convention every other
    // writer of this crate keeps.
    std::fs::write(path, text.as_bytes()).path(path)
}

/// OpenFOAM pads a keyword to 16 characters - the two LABEL files' header
/// only; the case fields carry theirs through `fields.rs`'s writer.
fn kw(out: &mut String, indent: &str, name: &str) {
    out.push_str(indent);
    out.push_str(name);
    if name.len() < 16 {
        for _ in 0..(16 - name.len()) {
            out.push(' ');
        }
    } else {
        out.push(' ');
    }
}

/// The FoamFile header of the two LABEL files, which carry no floating
/// point and so no case-field writer to borrow.
fn header(out: &mut String, class: &str, location: &str, object: &str) {
    out.push_str("FoamFile\n{\n");
    kw(out, "    ", "format");
    out.push_str("ascii;\n");
    kw(out, "    ", "class");
    out.push_str(class);
    out.push_str(";\n");
    kw(out, "    ", "location");
    out.push('"');
    out.push_str(location);
    out.push_str("\";\n");
    kw(out, "    ", "object");
    out.push_str(object);
    out.push_str(";\n");
    out.push_str("}\n");
}

/// `processorN/decomposition`: the set's shape, read back by
/// [`check_processor_set`] with `FoamDict::label`.
fn write_decomposition_file(path: &Path, dec: &Decomposition, part: usize) -> Result<()> {
    let mut out = String::new();
    header(&mut out, "dictionary", "decomposition", "decomposition");
    kw(&mut out, "", "nParts");
    out.push_str(&dec.n_parts.to_string());
    out.push_str(";\n");
    kw(&mut out, "", "part");
    out.push_str(&part.to_string());
    out.push_str(";\n");
    kw(&mut out, "", "nGlobalCells");
    out.push_str(&dec.n_global_cells.to_string());
    out.push_str(";\n");
    write_all(path, &out)
}

/// `processorN/constant/polyMesh/cellProcAddressing`: an OpenFOAM
/// `labelList`, the whole-mesh id of every owned cell.
fn write_cell_proc_addressing(path: &Path, cells: &[Label]) -> Result<()> {
    let mut out = String::new();
    header(&mut out, "labelList", "constant/polyMesh", "cellProcAddressing");
    out.push_str(&cells.len().to_string());
    out.push_str("\n(\n");
    for c in cells {
        out.push_str(&c.to_string());
        out.push('\n');
    }
    out.push_str(")\n");
    write_all(path, &out)
}

fn read_cell_proc_addressing(path: &Path) -> Result<Vec<Label>> {
    let text = slurp(path)?;
    let shown = path.display().to_string();
    check_ascii_format(&text, &shown)?;
    let mut tz = Tokenizer::new(&text, &shown);
    tz.skip_header()?;
    // An OpenFOAM labelList is `N ( ... )`; a leading count directly before
    // the '(' is the list's size, not an entry, so it is dropped.
    let counted = matches!(tz.peek()?, Some(Tok::Num(_)))
        && matches!(tz.peek_at(1), Some(Tok::Punct('(')));
    if counted {
        tz.next()?;
    }
    let mut cells = Vec::new();
    while !tz.done() {
        if tz.peek()?.map(|t| t.is_punct_any()).unwrap_or(false) {
            tz.next()?;
            continue;
        }
        cells.push(tz.expect_label()?);
    }
    tz.check_scan_error()?;
    Ok(cells)
}

// ==========================================================================
//  Reading a rank back
// ==========================================================================

/// Read rank `part`'s restart set. [`check_processor_set`] runs FIRST (host
/// only, nothing collective), then the rank's three files are read - `U` and
/// `p` with `read_vector_field`/`read_scalar_field` at the owned cell count,
/// `phi` at the internal face count - and every patch's `value` is checked
/// against the patch's face count, a length-1 list expanding to the patch.
/// A size-0 patch may carry no value at all: the writer drops an empty
/// list, and an absent value there reads as the empty slice it is.
pub fn read_rank(
    case: &Path,
    dec: &Decomposition,
    part: usize,
    time: &str,
) -> Result<RankFields> {
    check_processor_set(case, dec, time)?;
    let p = dec.parts.get(part).ok_or_else(|| {
        Error::Config(format!(
            "decomposed: this decomposition has {} part(s), part {part} was \
             asked to read",
            dec.n_parts
        ))
    })?;
    let m = &p.mesh;
    let tdir = processor_dir(case, part).join(time);

    let uf = read_vector_field(&tdir.join("U"), m.n_cells)?;
    let u_bf = boundary_vector(&uf.boundary, "U", m)?;
    let pf = read_scalar_field(&tdir.join("p"), m.n_cells)?;
    let p_bf = boundary_scalar(&pf.boundary, "p", m)?;
    let ff = read_scalar_field(&tdir.join("phi"), m.n_internal_faces)?;
    let phi_bf = boundary_scalar(&ff.boundary, "phi", m)?;

    Ok(RankFields {
        u: uf.internal,
        u_bf,
        p: pf.internal,
        p_bf,
        phi_f: ff.internal,
        phi_bf,
    })
}

/// The part's boundary-face array out of a field's `boundaryField`: one
/// entry per patch of the part, every one of them present, every `value`
/// the patch's face count (a length-1 list expands to the patch).
fn boundary_scalar(
    b: &BTreeMap<String, PatchFieldSpec>,
    field: &str,
    m: &HostMesh,
) -> Result<Vec<Scalar>> {
    let mut out = vec![0.0 as Scalar; m.n_boundary_faces];
    for pi in &m.patches {
        let s = patch_entry(b, field, &pi.name, pi.size)?;
        let slice = &mut out[pi.start..pi.start + pi.size];
        if s.value.len() == slice.len() {
            slice.copy_from_slice(&s.value);
        } else {
            slice.fill(s.value[0]);
        }
    }
    Ok(out)
}

fn boundary_vector(
    b: &BTreeMap<String, PatchFieldSpec>,
    field: &str,
    m: &HostMesh,
) -> Result<Vec<Vec3>> {
    let mut out = vec![Vec3::ZERO; m.n_boundary_faces];
    for pi in &m.patches {
        let s = patch_entry(b, field, &pi.name, pi.size)?;
        let slice = &mut out[pi.start..pi.start + pi.size];
        if s.value_v.len() == slice.len() {
            slice.copy_from_slice(&s.value_v);
        } else {
            slice.fill(s.value_v[0]);
        }
    }
    Ok(out)
}

/// One patch's entry, refused by name when it is absent or the wrong length.
fn patch_entry<'a>(
    b: &'a BTreeMap<String, PatchFieldSpec>,
    field: &str,
    patch: &str,
    size: usize,
) -> Result<&'a PatchFieldSpec> {
    let Some(s) = b.get(patch) else {
        return Err(Error::Config(format!(
            "decomposed: {field}'s boundaryField has no entry for patch \
             '{patch}' (the part has {size} face(s) there)"
        )));
    };
    let n = s.value.len().max(s.value_v.len());
    if n == size || n == 1 {
        return Ok(s);
    }
    Err(Error::Config(format!(
        "decomposed: {field}'s boundaryField entry for patch '{patch}' has \
         {n} value(s), the patch has {size} face(s)"
    )))
}

// ==========================================================================
//  The refusal table
// ==========================================================================

/// Is the processor set on disk THIS decomposition's? Every message is
/// `Error::Config` starting `decomposed: `. The checks are local reads -
/// nothing collective, so every rank of a run over the same disk refuses
/// identically, before anything else happens.
pub fn check_processor_set(case: &Path, dec: &Decomposition, time: &str) -> Result<()> {
    if !processor_dir(case, 0).is_dir() {
        return Err(Error::Config(format!(
            "decomposed: {} holds no processor set (no processor0)",
            case.display()
        )));
    }

    // k: the consecutive run processor0..processor{k-1}. A processor{M}
    // directory with M >= k is a gap: rank k's fields are gone.
    let mut k = 0usize;
    while processor_dir(case, k).is_dir() {
        k += 1;
    }
    for n in processor_dir_numbers(case)? {
        if n >= k {
            return Err(Error::Config(format!(
                "decomposed: processor{k} is missing from the processor set \
                 in {}",
                case.display()
            )));
        }
    }

    if k != dec.n_parts {
        return Err(Error::Config(format!(
            "decomposed: {} holds a processor set of {k} parts \
             (processor0..processor{}) and this run has {} ranks: a \
             processorN set from a different P is refused - reconstruct it, \
             or restart on {k} ranks",
            case.display(),
            k - 1,
            dec.n_parts
        )));
    }

    for n in 0..dec.n_parts {
        let dir = processor_dir(case, n);
        let dpath = dir.join("decomposition");
        let d = FoamDict::read(&dpath)?;
        let n_parts_disk = d.label("nParts", 0);
        let part_disk = d.label("part", -1);
        if n_parts_disk != k as Label || part_disk != n as Label {
            return Err(Error::Config(format!(
                "decomposed: {} says nParts {n_parts_disk}, part {part_disk}; \
                 the directory sits in a set of {k} parts as part {n}",
                dpath.display()
            )));
        }
        let n_global_disk = d.label("nGlobalCells", -1);
        if n_global_disk != dec.n_global_cells as Label {
            return Err(Error::Config(format!(
                "decomposed: {} says nGlobalCells {n_global_disk}, this run's \
                 decomposition has {} cells",
                dpath.display(),
                dec.n_global_cells
            )));
        }

        let cells =
            read_cell_proc_addressing(&dir.join("constant/polyMesh/cellProcAddressing"))?;
        let here = &dec.parts[n].global_cell[..dec.parts[n].mesh.n_cells];
        if let Some((i, a, b)) = first_differing(&cells, here) {
            return Err(Error::Config(format!(
                "decomposed: processor{n}'s cellProcAddressing is not this \
                 run's decomposition (local cell {i}: {a} on disk, {b} here) \
                 - the set was cut by another partitioner"
            )));
        }

        if !dir.join(time).is_dir() {
            return Err(Error::Config(format!(
                "decomposed: processor{n}/{time} does not exist"
            )));
        }
    }
    Ok(())
}

/// The numbers of every `processor<N>` directory of `case`.
fn processor_dir_numbers(case: &Path) -> Result<Vec<usize>> {
    let mut out = Vec::new();
    for e in std::fs::read_dir(case).path(case)? {
        let e = e.path(case)?;
        let name = e.file_name().to_string_lossy().into_owned();
        if let Some(n) = name
            .strip_prefix(PROCESSOR_PREFIX)
            .and_then(|s| s.parse::<usize>().ok())
        {
            out.push(n);
        }
    }
    Ok(out)
}

/// The first entry the two maps disagree on; `None` only when they are the
/// same length and equal throughout. A length mismatch reports the first
/// slot one list lacks, as `-1`.
fn first_differing(disk: &[Label], here: &[Label]) -> Option<(usize, Label, Label)> {
    for (i, (&a, &b)) in disk.iter().zip(here.iter()).enumerate() {
        if a != b {
            return Some((i, a, b));
        }
    }
    if disk.len() != here.len() {
        let i = disk.len().min(here.len());
        let a = disk.get(i).copied().unwrap_or(-1);
        let b = here.get(i).copied().unwrap_or(-1);
        return Some((i, a, b));
    }
    None
}

// ==========================================================================
//  The field builders - the case writer's own types, filled per patch
// ==========================================================================

/// A case field as the case writer's own type: `internal` inside, every
/// patch of `m` in the `boundaryField`. On a rank's OWN file (`own_file`)
/// every patch carries its slice of the boundary-face array
/// ([`rank_patch_vector`]); in a RECONSTRUCTED case field an Empty patch
/// carries an EMPTY list ([`whole_patch_vector`]) - there is nothing to
/// reconstruct behind a 2-D patch.
fn case_vector_field(
    name: &str,
    dims: &str,
    internal: &[Vec3],
    m: &HostMesh,
    bf: &[Vec3],
    own_file: bool,
) -> RawVectorField {
    let mut boundary = BTreeMap::new();
    for pi in &m.patches {
        let s = if own_file {
            rank_patch_vector(&bf[pi.start..pi.start + pi.size])
        } else {
            whole_patch_vector(pi, &bf[pi.start..pi.start + pi.size])
        };
        boundary.insert(pi.name.clone(), s);
    }
    RawVectorField {
        name: name.to_string(),
        dimensions: dims.to_string(),
        internal: internal.to_vec(),
        boundary,
        boundary_patterns: Vec::new(),
    }
}

/// [`case_vector_field`]'s scalar twin.
fn case_scalar_field(
    name: &str,
    dims: &str,
    internal: &[Scalar],
    m: &HostMesh,
    bf: &[Scalar],
    own_file: bool,
) -> RawScalarField {
    let mut boundary = BTreeMap::new();
    for pi in &m.patches {
        let s = if own_file {
            rank_patch_scalar(&bf[pi.start..pi.start + pi.size])
        } else {
            whole_patch_scalar(pi, &bf[pi.start..pi.start + pi.size])
        };
        boundary.insert(pi.name.clone(), s);
    }
    RawScalarField {
        name: name.to_string(),
        dimensions: dims.to_string(),
        internal: internal.to_vec(),
        boundary,
        boundary_patterns: Vec::new(),
    }
}

/// One patch of a rank's own file: `type calculated` and its slice of the
/// boundary-face array. EVERY patch carries the slice, processor and empty
/// included: it is the field's state there and [`read_rank`] brings it back
/// bit for bit. The case writer gives no `value` to a constraint type
/// (`empty`, `processor`, ...), where OpenFOAM writes none, so a rank's own
/// file types every patch `calculated`: the part's patch kinds come from the
/// decomposition, never from this file, and `calculated` is the type the
/// writer gives a value on every patch.
fn rank_patch_scalar(v: &[Scalar]) -> PatchFieldSpec {
    PatchFieldSpec {
        type_name: "calculated".to_string(),
        value: v.to_vec(),
        ..PatchFieldSpec::default()
    }
}

/// [`rank_patch_scalar`]'s vector twin.
fn rank_patch_vector(v: &[Vec3]) -> PatchFieldSpec {
    PatchFieldSpec {
        type_name: "calculated".to_string(),
        value_v: v.to_vec(),
        ..PatchFieldSpec::default()
    }
}

/// The boundaryField `type` of a patch of a RECONSTRUCTED whole-mesh case
/// field: `empty` for a 2-D front/back patch, `calculated` otherwise (the
/// whole mesh has no processor patch).
fn patch_type(pi: &PatchInfo) -> &'static str {
    if pi.kind == PatchKind::Empty {
        "empty"
    } else {
        "calculated"
    }
}

/// One patch of a RECONSTRUCTED case field: `type empty;` with an EMPTY
/// value list on an Empty patch - there is nothing to reconstruct behind a
/// 2-D patch, and the writer then writes no value at all - and
/// `calculated` with the patch's values on every other.
fn whole_patch_scalar(pi: &PatchInfo, bf: &[Scalar]) -> PatchFieldSpec {
    let mut s = PatchFieldSpec {
        type_name: patch_type(pi).to_string(),
        ..PatchFieldSpec::default()
    };
    if pi.kind != PatchKind::Empty {
        s.value = bf.to_vec();
    }
    s
}

/// [`whole_patch_scalar`]'s vector twin.
fn whole_patch_vector(pi: &PatchInfo, bf: &[Vec3]) -> PatchFieldSpec {
    let mut s = PatchFieldSpec {
        type_name: patch_type(pi).to_string(),
        ..PatchFieldSpec::default()
    };
    if pi.kind != PatchKind::Empty {
        s.value_v = bf.to_vec();
    }
    s
}

// ==========================================================================
//  The reconstruct
// ==========================================================================

/// The whole mesh's fields, gathered out of a processor set: cells in
/// whole-mesh order, boundary faces in the whole mesh's boundary order.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Reconstructed {
    /// `[whole.n_cells]`.
    pub u: Vec<Vec3>,
    /// `[whole.n_cells]`.
    pub p: Vec<Scalar>,
    /// `[whole.n_boundary_faces]` the boundary faces of the NON-processor
    /// patches - a cut face is an internal face of the whole mesh, not a
    /// boundary one, and no rank owns it as a boundary face.
    pub u_bf: Vec<Vec3>,
    /// `[whole.n_boundary_faces]`.
    pub p_bf: Vec<Scalar>,
}

/// The gather rule of [`reconstruct`], on sets already in hand: U and p go
/// to whole-mesh order by `global_cell`, the boundary faces of the
/// NON-processor patches by `global_face` (the whole boundary face is the
/// global polyMesh face id minus `dec.n_global_internal_faces`; a part whose
/// `global_face` is empty - the whole mesh, whose face ids ARE the global
/// ones - is the identity, the slot's own `n_internal_faces + bf`). Every
/// whole cell and every whole boundary face must be set EXACTLY once, or
/// the first gap is named and nothing is returned.
pub fn gather_written(
    whole: &HostMesh,
    dec: &Decomposition,
    written: &[RankFields],
) -> Result<Reconstructed> {
    if written.len() != dec.n_parts {
        return Err(Error::Config(format!(
            "decomposed: {} rank field set(s) given, the decomposition has {} \
             part(s)",
            written.len(),
            dec.n_parts
        )));
    }
    let mut r = Reconstructed {
        u: vec![Vec3::ZERO; whole.n_cells],
        p: vec![0.0 as Scalar; whole.n_cells],
        u_bf: vec![Vec3::ZERO; whole.n_boundary_faces],
        p_bf: vec![0.0 as Scalar; whole.n_boundary_faces],
    };
    let mut cell_seen = vec![0u32; whole.n_cells];
    let mut bf_seen = vec![0u32; whole.n_boundary_faces];

    for (p, w) in written.iter().enumerate() {
        let part = &dec.parts[p];
        let m = &part.mesh;
        check_len(&format!("part {p}'s U"), m.n_cells, w.u.len())?;
        check_len(&format!("part {p}'s U boundary"), m.n_boundary_faces, w.u_bf.len())?;
        check_len(&format!("part {p}'s p"), m.n_cells, w.p.len())?;
        check_len(&format!("part {p}'s p boundary"), m.n_boundary_faces, w.p_bf.len())?;
        for (i, &g) in part.global_cell.iter().enumerate().take(m.n_cells) {
            let g = g as usize;
            r.u[g] = w.u[i];
            r.p[g] = w.p[i];
            cell_seen[g] += 1;
        }
        for pi in &m.patches {
            if pi.type_name == PROCESSOR_PATCH_TYPE {
                continue;
            }
            for k in 0..pi.size {
                let bf = pi.start + k;
                // Empty means the identity: the whole mesh's face ids ARE
                // the global ones, slot for slot.
                let id = if m.global_face.is_empty() {
                    m.n_internal_faces + bf
                } else {
                    m.global_face[m.n_internal_faces + bf] as usize
                };
                let g = id - dec.n_global_internal_faces;
                r.u_bf[g] = w.u_bf[bf];
                r.p_bf[g] = w.p_bf[bf];
                bf_seen[g] += 1;
            }
        }
    }

    if let Some(g) = cell_seen.iter().position(|&c| c != 1) {
        return Err(cover_refusal("cell", g, cell_seen[g]));
    }
    if let Some(g) = bf_seen.iter().position(|&c| c != 1) {
        return Err(cover_refusal("boundary face", g, bf_seen[g]));
    }
    Ok(r)
}

/// The exactly-once refusal of [`gather_written`]: the first whole-mesh
/// index no rank wrote, or more than one did.
fn cover_refusal(what: &str, g: usize, seen: u32) -> Error {
    if seen == 0 {
        Error::Config(format!(
            "decomposed: the processor set does not cover the mesh: whole \
             {what} {g} was written by no rank"
        ))
    } else {
        Error::Config(format!(
            "decomposed: whole {what} {g} was written {seen} times by the \
             processor set"
        ))
    }
}

/// Gather a processor set back to the whole mesh and write it: the case
/// fields `<case>/<time>/U` and `<case>/<time>/p` - `empty` patches typed
/// `empty` with no value, every other patch `calculated` with its values -
/// and `<case>/VTK/reconstructed_<time>.vtu` with cell data U and p.
///
/// `phi` is deliberately absent: a reconstructed case is for
/// post-processing, and the flux a restart needs is read from the processor
/// set by [`read_rank`], where it lives per rank.
pub fn reconstruct(
    case: &Path,
    whole: &HostMesh,
    dec: &Decomposition,
    time: &str,
) -> Result<Reconstructed> {
    let mut written = Vec::with_capacity(dec.n_parts);
    for p in 0..dec.n_parts {
        written.push(read_rank(case, dec, p, time)?);
    }
    let r = gather_written(whole, dec, &written)?;

    let uf = case_vector_field("U", U_DIMS, &r.u, whole, &r.u_bf, false);
    write_vector_field_prec(&case.join(time).join("U"), &uf, time, ROUND_TRIP_DIGITS, false)?;
    let pf = case_scalar_field("p", P_DIMS, &r.p, whole, &r.p_bf, false);
    write_scalar_field_prec(&case.join(time).join("p"), &pf, time, ROUND_TRIP_DIGITS, false)?;

    let t = time.parse::<Scalar>().map_err(|_| {
        Error::Config(format!(
            "decomposed: the time name '{time}' is not a number; the \
             reconstructed VTU needs a numeric time"
        ))
    })?;
    let vdir = case.join("VTK");
    std::fs::create_dir_all(&vdir).path(&vdir)?;
    write_vtu(
        &case.join("VTK").join(format!("reconstructed_{time}.vtu")),
        whole,
        &[
            OutputField::vector("U", &r.u),
            OutputField::scalar("p", &r.p),
        ],
        Some(t),
    )?;
    Ok(r)
}
