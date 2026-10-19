// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! The one-domain vent (docs/17 DST-09, SPEC-LIT §158.1). The membrane M is a
//! set of INTERNAL faces of the case's own mesh (the "merged" mesh), selected
//! by a plane disc or by face id. While closed the run solves on a SPLIT mesh:
//! the same points and cells, with M's faces lifted out of the internal list
//! and appended as two wall patches, `ventIn` (each face as seen from its
//! in-side cell) and `ventOut` (as seen from its out-side cell). The split
//! mesh's boundary faces are the merged ones in the same order FOLLOWED by the
//! two vent patches, so every merged boundary array is a PREFIX of the split
//! one and every cell index is the same in both meshes. The trigger (158.1):
//! the membrane opens at the first step whose area-weighted face-neighbour
//! pressure difference reaches `pStat`, once and irreversibly. On opening the
//! state is carried onto a solver on the merged mesh, cells copied bit for
//! bit, M's faces starting at mdot = 0. No kernel is touched: the gather is
//! `presGather`, already in `cuda/pressure.cu`. All geometry here is host f64
//! from [`PolyMeshRaw::points`], in BOTH builds.
//!
//! Written from:
//!   ofgpu SPEC-LIT.md §158.1 (158.1) - the membrane, the trigger, the
//!     split mesh and the hand-over
//!   ofgpu SPEC-LIT.md §158.5 - what DST-09 must make true
//!   ofgpu SPEC-LIT.md §158.8 - the refusals and the step count a
//!     handed-over run continues
//!
//! ORIGINAL
//!
//! The specification was informed by a review of FSCDustFoam (Huang &
//! Lipatnikov, GPL) and no code was copied.
//! No GPL-licensed source was consulted.

use crate::compressible::bc::Lowered;
use crate::compressible::{Compressible, CompressibleControls, CompressibleState};
use crate::device::{cfg_for, DevBuf, Gpu, KernelSet};
use crate::error::{Error, Result};
use crate::field::{GpuScalarField, GpuVectorField};
use crate::field_setup::{setup_scalar_field, setup_vector_field};
use crate::io::fields::PatchFieldSpec;
use crate::io::polymesh::PolyMeshRaw;
use crate::mesh::{GpuMesh, HostMesh, PatchInfo, PatchKind};
use crate::{DVec3, Label, Scalar};
use cudarc::driver::{CudaFunction, PushKernelArg};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

// ==========================================================================
//  The case's `vent` block and its lowering (host, no mesh)
// ==========================================================================

/// The membrane's wall patch as seen from the IN-side cell (SPEC-LIT §158.1).
pub const VENT_IN: &str = "ventIn";
/// The membrane's wall patch as seen from the OUT-side cell (SPEC-LIT §158.1).
pub const VENT_OUT: &str = "ventOut";

/// The case's `vent` block, `JsonCase::vent` (SPEC-LIT §158.8): a normal and
/// `pStat`, then either the disc (`centre` + `radius`) or the face set (`faces`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct JsonVent {
    /// Points from the inside (the vessel) to the outside; any length > 0.
    pub normal: [f64; 3],
    /// Pa, the static activation pressure: opens when p_in - p_out >= pStat.
    #[serde(rename = "pStat")]
    pub p_stat: f64,
    /// The disc route: a point on the membrane plane, and the disc radius [m].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub centre: Option<[f64; 3]>,
    /// The disc route: the radius [m] about `centre`, in the plane.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub radius: Option<f64>,
    /// The face-set route: internal face ids of the case's mesh.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub faces: Option<Vec<usize>>,
    /// `noSlip` (default) | `slip`: U on the membrane while closed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub wall: Option<String>,
    /// SPEC-LIT §158.10: named pressure probes. Absent is every case this
    /// reader has built; present, each probe is the cell nearest its point
    /// ([`probe_cells`]) and the driver records that cell's gauge pressure.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub probes: Option<Vec<JsonProbe>>,
}

/// One `vent.probes[]` entry (SPEC-LIT §158.10): a name made of
/// `[A-Za-z0-9_]` and a world point in m.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct JsonProbe {
    pub name: String,
    pub point: [f64; 3],
}

/// U on the membrane while it is closed (SPEC-LIT §158.1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MembraneWall {
    NoSlip,
    Slip,
}

/// How the case picks M out of the merged mesh's internal faces (§158.1).
#[derive(Debug, Clone, PartialEq)]
pub enum VentSelect {
    Disc { centre: DVec3, radius: f64 },
    Faces(Vec<usize>),
}

/// A lowered `vent` block: the unit normal, the selection, the static
/// activation pressure and the closed-stage wall treatment.
#[derive(Debug, Clone, PartialEq)]
pub struct VentSpec {
    /// Unit, in f64.
    pub normal: DVec3,
    pub select: VentSelect,
    pub p_stat: f64,
    pub wall: MembraneWall,
    /// SPEC-LIT §158.10: the lowered `vent.probes`, in the case's order;
    /// empty when the block states none.
    pub probes: Vec<VentProbe>,
}

/// A lowered `vent.probes[]` entry (SPEC-LIT §158.10).
#[derive(Debug, Clone, PartialEq)]
pub struct VentProbe {
    pub name: String,
    pub point: DVec3,
}

/// A `vent.<key>` refusal naming the key and the value it saw (§158.8).
fn vent_refused(key: &str, value: impl std::fmt::Display, why: &str) -> Error {
    Error::Refused(format!("vent.{key}: {value} - {why} (SPEC-LIT §158.8)"))
}

/// A `vent.probes` refusal naming the key, the probe (its name or index) and
/// the value it saw (SPEC-LIT §158.10). [`vent_refused`] hard-codes §158.8.
fn probe_refused(which: impl std::fmt::Display, value: impl std::fmt::Display, why: &str) -> Error {
    Error::Refused(format!("vent.probes: {which}: {value} - {why} (SPEC-LIT §158.10)"))
}

/// Lower `vent.probes` (SPEC-LIT §158.10): `Ok(empty)` when the block states
/// none; refused, naming the key and the probe, for an empty list, a name
/// that is empty, repeated or not made only of `[A-Za-z0-9_]`, and a point
/// with a non-finite coordinate.
fn lower_probes(probes: Option<&Vec<JsonProbe>>) -> Result<Vec<VentProbe>> {
    let Some(list) = probes else { return Ok(Vec::new()) };
    if list.is_empty() {
        return Err(Error::Refused(
            "vent.probes: [] - is an empty probe list: state a probe or drop the key \
             (SPEC-LIT §158.10)"
                .into(),
        ));
    }
    let mut seen = std::collections::HashSet::new();
    let mut out = Vec::with_capacity(list.len());
    for (i, p) in list.iter().enumerate() {
        if p.name.is_empty() {
            return Err(probe_refused(format!("probe {i}"), "\"\"", "has an empty name"));
        }
        if !p.name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
            return Err(probe_refused(
                format!("probe \"{}\" (index {i})", p.name),
                format!("\"{}\"", p.name),
                "a name is made only of [A-Za-z0-9_]",
            ));
        }
        if !seen.insert(p.name.as_str()) {
            return Err(probe_refused(
                format!("probe \"{}\" (index {i})", p.name),
                format!("\"{}\"", p.name),
                "the name is repeated - each probe has its own",
            ));
        }
        if !p.point.iter().all(|v| v.is_finite()) {
            return Err(probe_refused(
                format!("probe \"{}\" (index {i})", p.name),
                format!("{:?}", p.point),
                "is not a finite point",
            ));
        }
        out.push(VentProbe { name: p.name.clone(), point: DVec3::new(p.point[0], p.point[1], p.point[2]) });
    }
    Ok(out)
}

impl JsonVent {
    /// Lower the block. Every refusal names its key and the value it saw
    /// (SPEC-LIT §158.8).
    pub fn lower(&self) -> Result<VentSpec> {
        let n = self.normal;
        let len = DVec3::new(n[0], n[1], n[2]).mag();
        if !n.iter().all(|v| v.is_finite()) || len == 0.0 {
            return Err(vent_refused("normal", format!("{n:?}"), "is not a finite, non-zero vector"));
        }
        let normal = DVec3::new(n[0] / len, n[1] / len, n[2] / len);
        if !self.p_stat.is_finite() || self.p_stat < 0.0 {
            return Err(vent_refused("pStat", self.p_stat, "must be a finite pressure >= 0 Pa"));
        }
        if self.faces.as_ref().is_some_and(|f| f.is_empty()) {
            return Err(vent_refused("faces", "[]", "is an empty face set"));
        }
        let select = match (self.centre, self.radius, &self.faces) {
            (Some(c), Some(r), None) => Self::lower_disc(&c, r)?,
            // The face-set route: the ids are checked against the mesh by
            // `select_faces` (range, repeats, edge-on faces).
            (None, None, Some(ids)) => VentSelect::Faces(ids.clone()),
            (None, None, None) => {
                return Err(Error::Refused(
                    "vent: states no faces: give centre + radius, or faces (SPEC-LIT §158.8)".into(),
                ));
            }
            _ => return Err(self.refuse_route()),
        };
        let wall = match self.wall.as_deref() {
            None => MembraneWall::NoSlip,
            Some("noSlip") => MembraneWall::NoSlip,
            Some("slip") => MembraneWall::Slip,
            Some(w) => {
                return Err(vent_refused("wall", format!("\"{w}\""), "is neither `noSlip` nor `slip`"));
            }
        };
        let probes = lower_probes(self.probes.as_ref())?;
        Ok(VentSpec { normal, select, p_stat: self.p_stat, wall, probes })
    }

    /// The disc route, once both keys are present.
    fn lower_disc(centre: &[f64; 3], radius: f64) -> Result<VentSelect> {
        if !radius.is_finite() || radius <= 0.0 {
            return Err(vent_refused("radius", radius, "must be a finite radius > 0 m"));
        }
        if !centre.iter().all(|v| v.is_finite()) {
            return Err(vent_refused("centre", format!("{centre:?}"), "is not a finite point"));
        }
        Ok(VentSelect::Disc { centre: DVec3::new(centre[0], centre[1], centre[2]), radius })
    }

    /// Every way a case can state both routes, half of one, or one beside the
    /// other: refused by name (SPEC-LIT §158.8).
    fn refuse_route(&self) -> Error {
        let why = "a case states one route: the disc or the face set (SPEC-LIT §158.8)";
        if self.faces.is_some() && (self.centre.is_some() || self.radius.is_some()) {
            return Error::Refused(format!(
                "vent.faces: {:?} beside vent.radius/vent.centre - {why}",
                self.faces
            ));
        }
        if let Some(r) = self.radius {
            return Error::Refused(format!("vent.radius: {r} without vent.centre - {why}"));
        }
        Error::Refused(format!(
            "vent.centre: {:?} without vent.radius - {why}",
            self.centre
        ))
    }
}

// ==========================================================================
//  Selecting M on the merged raw mesh
// ==========================================================================

/// One membrane face of the merged mesh (SPEC-LIT §158.1).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VentFace {
    /// The merged internal face id.
    pub face: usize,
    /// True when the face's OWNER is on the inside: `S . n > 0`, `S` pointing
    /// owner -> neighbour, so the owner sits on the side `n` points away from.
    pub owner_is_in: bool,
    /// The face area `|S|`, in f64 in BOTH builds.
    pub area: f64,
}

/// The area vector `S` of face `f`, from `PolyMeshRaw::points` in f64 in BOTH
/// builds: `S = 1/2 sum_{i=1}^{k-2} (p_i - p_0) x (p_{i+1} - p_0)`.
pub fn face_area_vector(raw: &PolyMeshRaw, f: usize) -> DVec3 {
    let v = &raw.faces[f];
    let p0 = raw.points[v[0] as usize];
    let mut s = DVec3::ZERO;
    for i in 1..v.len() - 1 {
        let a = raw.points[v[i] as usize] - p0;
        let b = raw.points[v[i + 1] as usize] - p0;
        s += a.cross(b);
    }
    s * 0.5
}

/// The face centre as the vertex average `(1/k) sum p_i`.
pub fn face_centre(raw: &PolyMeshRaw, f: usize) -> DVec3 {
    let v = &raw.faces[f];
    let mut c = DVec3::ZERO;
    for p in v {
        c += raw.points[*p as usize];
    }
    c / v.len() as f64
}

/// The membrane M: the merged mesh's internal faces the spec selects,
/// ascending by face id. Refused by name: a face id past the internal range,
/// a duplicate, a face edge-on to `vent.normal`, and an empty M (SPEC-LIT
/// §158.8).
pub fn select_faces(raw: &PolyMeshRaw, spec: &VentSpec) -> Result<Vec<VentFace>> {
    let n_internal = raw.neighbour.len();
    let mut out: Vec<VentFace> = Vec::new();
    match &spec.select {
        VentSelect::Disc { centre, radius } => {
            // tol = 1e-6 L, L the diagonal of the points' bounding box.
            let (mut lo, mut hi) = (raw.points[0], raw.points[0]);
            for p in &raw.points {
                lo = lo.cmpt_min(*p);
                hi = hi.cmpt_max(*p);
            }
            let tol = 1.0e-6 * (hi - lo).mag();
            for f in 0..n_internal {
                let s = face_area_vector(raw, f);
                let area = s.mag();
                if area == 0.0 {
                    continue;
                }
                if spec.normal.dot(s).abs() / area < 1.0 - 1.0e-6 {
                    continue; // not in the membrane plane
                }
                let x = face_centre(raw, f);
                let d = spec.normal.dot(x - *centre);
                if d.abs() > tol {
                    continue; // off the plane through `centre`
                }
                let radial = ((x - *centre) - spec.normal * d).mag();
                if radial <= *radius {
                    out.push(VentFace {
                        face: f,
                        owner_is_in: spec.normal.dot(s) > 0.0,
                        area,
                    });
                }
            }
        }
        VentSelect::Faces(ids) => {
            let mut seen = std::collections::HashSet::new();
            for &id in ids {
                if id >= n_internal {
                    return Err(Error::Refused(format!(
                        "vent.faces: {id} is not an internal face of the mesh, \
                         which has {n_internal} (SPEC-LIT §158.8)"
                    )));
                }
                if !seen.insert(id) {
                    return Err(Error::Refused(format!(
                        "vent.faces: {id} appears twice (SPEC-LIT §158.8)"
                    )));
                }
                let s = face_area_vector(raw, id);
                let area = s.mag();
                if area == 0.0 || spec.normal.dot(s).abs() / area < 1.0e-6 {
                    return Err(Error::Refused(format!(
                        "vent.faces: face {id} is edge-on to vent.normal, \
                         so it has no inside (SPEC-LIT §158.8)"
                    )));
                }
                out.push(VentFace {
                    face: id,
                    owner_is_in: spec.normal.dot(s) > 0.0,
                    area,
                });
            }
            out.sort_by_key(|f| f.face);
        }
    }
    if out.is_empty() {
        return Err(Error::Refused(
            "vent: selects no internal face (SPEC-LIT §158.8)".into(),
        ));
    }
    Ok(out)
}

// ==========================================================================
//  The split mesh
// ==========================================================================

/// The case's mesh SPLIT by the membrane: the same points and cells, with M's
/// faces moved out of the internal list and appended as the two wall patches
/// `ventIn`/`ventOut` (SPEC-LIT §158.1). The merged boundary arrays are a
/// PREFIX of the split ones, so the merged patches keep their `start`s.
#[derive(Debug, Clone)]
pub struct VentSplit {
    /// The split mesh.
    pub raw: PolyMeshRaw,
    /// M, ascending by merged face id.
    pub faces: Vec<VentFace>,
    /// `kept[i]` = the merged internal face id of split internal face i.
    pub kept: Vec<usize>,
    pub n_internal_merged: usize,
    /// The shared prefix of the boundary arrays.
    pub n_boundary_merged: usize,
    /// Per M face, in M's order: the cell on the inside.
    pub in_cells: Vec<usize>,
    /// Per M face, in M's order: the cell on the outside.
    pub out_cells: Vec<usize>,
    /// `|S_f|` per M face.
    pub areas: Vec<f64>,
    /// The sum, in M's order.
    pub area: f64,
}

impl VentSplit {
    /// The flattened boundary-face range of `ventIn`.
    pub fn vent_in_range(&self) -> std::ops::Range<usize> {
        self.n_boundary_merged..self.n_boundary_merged + self.faces.len()
    }

    /// The flattened boundary-face range of `ventOut`.
    pub fn vent_out_range(&self) -> std::ops::Range<usize> {
        let b = self.n_boundary_merged + self.faces.len();
        b..b + self.faces.len()
    }
}

/// Split `raw` by `faces`. Refused by name: a raw that already carries a
/// `ventIn`/`ventOut` patch, and a face id that is not internal (SPEC-LIT
/// §158.8).
pub fn split_raw(raw: &PolyMeshRaw, faces: Vec<VentFace>) -> Result<VentSplit> {
    for p in &raw.patches {
        if p.name == VENT_IN || p.name == VENT_OUT {
            return Err(Error::Refused(format!(
                "vent: the mesh already carries a patch named {}, so it is a \
                 split mesh already (SPEC-LIT §158.8)",
                p.name
            )));
        }
    }
    let n_internal = raw.neighbour.len();
    if let Some(f) = faces.iter().find(|f| f.face >= n_internal) {
        return Err(Error::Refused(format!(
            "vent.faces: face {} is not an internal face of the mesh, which \
             has {n_internal} (SPEC-LIT §158.8)",
            f.face
        )));
    }
    let in_m: std::collections::HashSet<usize> = faces.iter().map(|f| f.face).collect();

    // The kept internal faces, in their order; then every original boundary
    // face, in its order, so the merged patches keep their starts.
    let mut kept: Vec<usize> = Vec::with_capacity(n_internal.saturating_sub(in_m.len()));
    let mut new_faces: Vec<Vec<Label>> = Vec::new();
    let mut new_owner: Vec<Label> = Vec::new();
    for i in 0..n_internal {
        if !in_m.contains(&i) {
            kept.push(i);
            new_faces.push(raw.faces[i].clone());
            new_owner.push(raw.owner[i]);
        }
    }
    for i in n_internal..raw.faces.len() {
        new_faces.push(raw.faces[i].clone());
        new_owner.push(raw.owner[i]);
    }

    // Per M face, in M's order: the cell on each side. `owner_is_in` says the
    // merged owner is the inside; otherwise the neighbour is.
    let mut in_cells = Vec::with_capacity(faces.len());
    let mut out_cells = Vec::with_capacity(faces.len());
    for f in &faces {
        let owner = raw.owner[f.face] as usize;
        let nbr = raw.neighbour[f.face] as usize;
        let (i, o) = if f.owner_is_in { (owner, nbr) } else { (nbr, owner) };
        in_cells.push(i);
        out_cells.push(o);
    }

    // The two membrane patches. A face is written AS IS when its new owner is
    // the merged owner, and REVERSED when its new owner is the merged
    // neighbour; the reversal keeps the first vertex and reverses the rest:
    // [v0, v1, ..., vk-1] -> [v0, vk-1, ..., v1].
    for cells in [&in_cells, &out_cells] {
        for (k, &cell) in cells.iter().enumerate() {
            let merged = faces[k].face;
            let mut verts = raw.faces[merged].clone();
            if cell != raw.owner[merged] as usize {
                let rest: Vec<Label> = verts[1..].iter().rev().copied().collect();
                verts[1..].copy_from_slice(&rest);
            }
            new_faces.push(verts);
            new_owner.push(cell as Label);
        }
    }

    // The original patches are unchanged - their `start` is relative to the
    // boundary arrays, whose order did not move - then the two vent patches.
    let n_b = raw.faces.len() - n_internal;
    let mut patches = raw.patches.clone();
    patches.push(PatchInfo {
        name: VENT_IN.to_string(),
        type_name: "wall".to_string(),
        kind: PatchKind::Wall,
        start: n_b,
        size: faces.len(),
        nbr_patch: None,
    });
    patches.push(PatchInfo {
        name: VENT_OUT.to_string(),
        type_name: "wall".to_string(),
        kind: PatchKind::Wall,
        start: n_b + faces.len(),
        size: faces.len(),
        nbr_patch: None,
    });

    Ok(VentSplit {
        raw: PolyMeshRaw {
            points: raw.points.clone(),
            faces: new_faces,
            owner: new_owner,
            neighbour: kept.iter().map(|&i| raw.neighbour[i]).collect(),
            patches,
        },
        kept,
        n_internal_merged: n_internal,
        n_boundary_merged: n_b,
        in_cells,
        out_cells,
        areas: faces.iter().map(|f| f.area).collect(),
        area: faces.iter().map(|f| f.area).sum(),
        faces,
    })
}

// ==========================================================================
//  The fields and the solver, on either mesh
// ==========================================================================

/// `low` (the case's lowered fields and rules) plus the two membrane patches:
/// U `noSlip` ([`MembraneWall::NoSlip`]) or `slip`, p and T `zeroGradient` -
/// each a [`PatchFieldSpec`] inserted under [`VENT_IN`] and [`VENT_OUT`]. The
/// rules are cloned unchanged: their face ranges lie in the boundary prefix
/// the two meshes share. Refused: a key [`VENT_IN`]/[`VENT_OUT`] already in
/// any of the three maps, and a pattern key that would govern a vent patch.
pub fn closed_lowered(low: &Lowered, wall: MembraneWall) -> Result<Lowered> {
    let mut out = low.clone();
    for (field, map) in
        [("U", &out.u.boundary), ("p", &out.p.boundary), ("T", &out.t.boundary)]
    {
        for key in [VENT_IN, VENT_OUT] {
            if map.contains_key(key) {
                return Err(Error::Refused(format!(
                    "vent: the case already states a boundaryField entry for \
                     {key} on {field} (SPEC-LIT §158.8)"
                )));
            }
        }
    }
    let u_type = match wall {
        MembraneWall::NoSlip => "noSlip",
        MembraneWall::Slip => "slip",
    };
    for key in [VENT_IN, VENT_OUT] {
        out.u.boundary.insert(
            key.to_string(),
            PatchFieldSpec { type_name: u_type.to_string(), ..Default::default() },
        );
        out.p.boundary.insert(
            key.to_string(),
            PatchFieldSpec { type_name: "zeroGradient".to_string(), ..Default::default() },
        );
        out.t.boundary.insert(
            key.to_string(),
            PatchFieldSpec { type_name: "zeroGradient".to_string(), ..Default::default() },
        );
    }
    // An explicit key wins over a pattern key when a patch matches both
    // (`field_setup`'s `governing_key` tests the exact key first), and the
    // exact keys now exist - so no pattern can govern a vent patch. Refuse
    // one that would, so the precedence the run rests on is checked, not
    // assumed (SPEC-LIT §158.8).
    for (boundary, patterns) in [
        (&out.u.boundary, &out.u.boundary_patterns),
        (&out.p.boundary, &out.p.boundary_patterns),
        (&out.t.boundary, &out.t.boundary_patterns),
    ] {
        for key in [VENT_IN, VENT_OUT] {
            match crate::io::fields::governing_key(boundary, patterns, key)? {
                Some(k) if k == key => {}
                Some(k) => {
                    return Err(Error::Refused(format!(
                        "vent: the pattern {k} would govern {key}; state the \
                         patch explicitly (SPEC-LIT §158.8)"
                    )));
                }
                None => {}
            }
        }
    }
    Ok(out)
}

/// The state a step reads, on either mesh: the lowered U/p/T built as
/// `src/bin/compressible.rs` builds the case's own state, and the derived
/// fields allocated as zeros for [`Compressible::initialise`] to fill.
pub fn setup_state(
    gpu: &Gpu,
    gm: &GpuMesh,
    hm: &HostMesh,
    low: &Lowered,
) -> Result<CompressibleState> {
    let mut u = GpuVectorField::zeros(gpu, gm, "U")?;
    let mut p = GpuScalarField::zeros(gpu, gm, "p")?;
    let mut t = GpuScalarField::zeros(gpu, gm, "T")?;
    setup_vector_field(gpu, &mut u, &low.u, hm)?;
    setup_scalar_field(gpu, &mut p, &low.p, hm)?;
    setup_scalar_field(gpu, &mut t, &low.t, hm)?;
    CompressibleState::new(gpu, gm, u, p, t)
}

/// A [`Compressible`] on `gm` with `low`'s four rule vectors, as the driver
/// sets them, in its order: boundary, outlets, far-fields, free streams.
pub fn build_solver<'m>(
    gpu: &Gpu,
    gm: &'m GpuMesh,
    ctrl: CompressibleControls,
    low: &Lowered,
) -> Result<Compressible<'m>> {
    let mut solver = Compressible::new(gpu, gm, ctrl)?;
    solver.set_boundary_rules(gpu, low.rules.clone())?;
    solver.set_outlet_rules(gpu, low.outlets.clone())?;
    solver.set_farfield_rules(gpu, low.farfields.clone())?;
    solver.set_freestream_rules(gpu, low.freestreams.clone())?;
    Ok(solver)
}

// ==========================================================================
//  The trigger (158.1) and the membrane state
// ==========================================================================

/// One reading of the membrane: the two area-weighted face-neighbour
/// pressures and their difference (158.1).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VentReading {
    pub p_in: f64,
    pub p_out: f64,
    pub dp: f64,
}

/// The area-weighted mean `sum_f a_f v_f / sum_f a_f`, both sums left to
/// right in the slice order, in f64 (158.1's `p‾`).
pub fn area_weighted(values: &[f64], areas: &[f64]) -> f64 {
    let mut num = 0.0;
    let mut den = 0.0;
    for (v, a) in values.iter().zip(areas) {
        num += a * v;
        den += a;
    }
    num / den
}

/// The membrane's pressure gather: `presGather` collects the gauge pressure
/// at the in-side and out-side cells of M onto the device, and `read` forms
/// the two area-weighted means and their difference (SPEC-LIT §158.1).
pub struct VentTrigger {
    gather: CudaFunction,
    in_cells: DevBuf<Label>,
    out_cells: DevBuf<Label>,
    in_p: DevBuf<Scalar>,
    out_p: DevBuf<Scalar>,
    areas: Vec<f64>,
}

impl VentTrigger {
    pub fn new(gpu: &Gpu, split: &VentSplit) -> Result<Self> {
        let gather = KernelSet::new(gpu, crate::kernels::PRESSURE)?.func("presGather")?;
        Ok(Self {
            gather,
            in_cells: gpu.upload(&(split.in_cells.iter().map(|&c| c as Label).collect::<Vec<_>>()))?,
            out_cells: gpu
                .upload(&(split.out_cells.iter().map(|&c| c as Label).collect::<Vec<_>>()))?,
            in_p: gpu.zeros(split.faces.len())?,
            out_p: gpu.zeros(split.faces.len())?,
            areas: split.areas.clone(),
        })
    }

    /// Gather p at the in and out cells on the device, download both, widen
    /// with `f64::from` and form the two area-weighted means. `p` is the
    /// gauge p's cell array (`state.p.f`); the datum cancels in `dp`. A
    /// non-finite `dp` is refused (SPEC-LIT §158.8).
    pub fn read(&mut self, gpu: &Gpu, p: &DevBuf<Scalar>) -> Result<VentReading> {
        let n = self.areas.len();
        let m = n as Label;
        let f = self.gather.clone();
        unsafe {
            gpu.stream()
                .launch_builder(&f)
                .arg(&mut self.in_p)
                .arg(p)
                .arg(&self.in_cells)
                .arg(&m)
                .launch(cfg_for(n))?;
            gpu.stream()
                .launch_builder(&f)
                .arg(&mut self.out_p)
                .arg(p)
                .arg(&self.out_cells)
                .arg(&m)
                .launch(cfg_for(n))?;
        }
        let widen = |v: Vec<Scalar>| v.into_iter().map(f64::from).collect::<Vec<f64>>();
        let p_in = area_weighted(&widen(gpu.download(&self.in_p)?), &self.areas);
        let p_out = area_weighted(&widen(gpu.download(&self.out_p)?), &self.areas);
        let dp = p_in - p_out;
        if !dp.is_finite() {
            return Err(Error::Refused(format!(
                "vent: the membrane pressure difference is not finite: dp = {dp} \
                 (SPEC-LIT §158.8)"
            )));
        }
        Ok(VentReading { p_in, p_out, dp })
    }
}

// ==========================================================================
//  The probes (SPEC-LIT §158.10)
// ==========================================================================

/// The cell each probe reads (SPEC-LIT §158.10): the cell whose WORLD centre,
/// `hm.origin + hm.geom64().c[i]` in f64, is nearest the probe's point; a tie
/// goes to the lowest id. A point outside the axis-aligned box of the world
/// boundary-face centres (`hm.origin + hm.geom64().b_cf[j]`) is refused,
/// naming `vent.probes` and the probe - [`HostMesh`] keeps no points, and on a
/// box mesh this box is the mesh's own.
pub fn probe_cells(hm: &HostMesh, probes: &[VentProbe]) -> Result<Vec<usize>> {
    let g = hm.geom64();
    let Some(&first) = g.b_cf.first() else {
        return Err(Error::Refused(
            "vent.probes: the mesh has no boundary face, so it has no box to place a probe in \
             (SPEC-LIT §158.10)"
                .into(),
        ));
    };
    let (mut lo, mut hi) = (hm.origin + first, hm.origin + first);
    for &b in g.b_cf.iter() {
        let w = hm.origin + b;
        lo = lo.cmpt_min(w);
        hi = hi.cmpt_max(w);
    }
    let mut out = Vec::with_capacity(probes.len());
    for p in probes {
        let q = p.point;
        let inside = q.x >= lo.x
            && q.x <= hi.x
            && q.y >= lo.y
            && q.y <= hi.y
            && q.z >= lo.z
            && q.z <= hi.z;
        if !inside {
            return Err(Error::Refused(format!(
                "vent.probes: probe \"{}\" at ({} {} {}) - lies outside the box of the mesh's \
                 boundary-face centres, ({} {} {}) to ({} {} {}) (SPEC-LIT §158.10)",
                p.name, q.x, q.y, q.z, lo.x, lo.y, lo.z, hi.x, hi.y, hi.z
            )));
        }
        let mut best = 0usize;
        let mut best_d = f64::INFINITY;
        for i in 0..hm.n_cells {
            let d = ((hm.origin + g.c[i]) - q).mag_sqr();
            // Strictly less: a tie keeps the lowest id.
            if d < best_d {
                best_d = d;
                best = i;
            }
        }
        out.push(best);
    }
    Ok(out)
}

/// The probes' pressure gather (SPEC-LIT §158.10): `presGather`, exactly as
/// [`VentTrigger`] launches it, collects the gauge pressure at the probes'
/// cells and `read` returns them widened to f64, in probe order.
pub struct VentProbes {
    gather: CudaFunction,
    cells: DevBuf<Label>,
    out: DevBuf<Scalar>,
    n: usize,
}

impl VentProbes {
    pub fn new(gpu: &Gpu, cells: &[usize]) -> Result<Self> {
        let gather = KernelSet::new(gpu, crate::kernels::PRESSURE)?.func("presGather")?;
        Ok(Self {
            gather,
            cells: gpu.upload(&(cells.iter().map(|&c| c as Label).collect::<Vec<_>>()))?,
            out: gpu.zeros(cells.len())?,
            n: cells.len(),
        })
    }

    /// Gather `p` (the gauge p's cell array) at the probes' cells on the
    /// device, download and widen with `f64::from`, in probe order.
    pub fn read(&mut self, gpu: &Gpu, p: &DevBuf<Scalar>) -> Result<Vec<f64>> {
        let m = self.n as Label;
        let f = self.gather.clone();
        unsafe {
            gpu.stream()
                .launch_builder(&f)
                .arg(&mut self.out)
                .arg(p)
                .arg(&self.cells)
                .arg(&m)
                .launch(cfg_for(self.n))?;
        }
        Ok(gpu.download(&self.out)?.into_iter().map(f64::from).collect())
    }
}

/// The membrane's state: closed, or open from the step, time and Δp it
/// opened at (158.1, irreversible).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum VentState {
    Closed,
    Open { step: u64, time: f64, dp: f64 },
}

/// The membrane and its static activation pressure (SPEC-LIT §158.1).
#[derive(Debug, Clone, PartialEq)]
pub struct Membrane {
    pub p_stat: f64,
    pub state: VentState,
}

impl Membrane {
    pub fn new(p_stat: f64) -> Self {
        Self { p_stat, state: VentState::Closed }
    }

    /// (158.1): Closed and `r.dp >= p_stat` opens the membrane at this step,
    /// returning true. Open stays Open - the rupture is irreversible - and
    /// returns false; Closed with `r.dp < p_stat` returns false.
    pub fn check(&mut self, step: u64, time: f64, r: &VentReading) -> bool {
        match self.state {
            VentState::Open { .. } => false,
            VentState::Closed if r.dp >= self.p_stat => {
                self.state = VentState::Open { step, time, dp: r.dp };
                true
            }
            VentState::Closed => false,
        }
    }

    pub fn is_open(&self) -> bool {
        matches!(self.state, VentState::Open { .. })
    }
}

// ==========================================================================
//  The hand-over (the opening)
// ==========================================================================

/// Copy one scalar field by host round trip: `f`, `f0` and `f00` whole, and
/// the first `n_bf` entries of every `[n_bf]` value buffer the struct has.
fn hand_over_scalar(
    gpu: &Gpu,
    from: &GpuScalarField,
    to: &mut GpuScalarField,
    n_bf: usize,
) -> Result<()> {
    for (f, t) in [(&from.f, &mut to.f), (&from.f0, &mut to.f0), (&from.f00, &mut to.f00)] {
        gpu.write(t, &gpu.download(f)?)?;
    }
    for (f, t) in [
        (&from.bf, &mut to.bf),
        (&from.fr, &mut to.fr),
        (&from.ref_value, &mut to.ref_value),
        (&from.ref_grad, &mut to.ref_grad),
    ] {
        let v = gpu.download(f)?;
        gpu.write(t, &v[..n_bf])?;
    }
    Ok(())
}

/// As [`hand_over_scalar`], on a vector field.
fn hand_over_vector(
    gpu: &Gpu,
    from: &GpuVectorField,
    to: &mut GpuVectorField,
    n_bf: usize,
) -> Result<()> {
    for (f, t) in [(&from.f, &mut to.f), (&from.f0, &mut to.f0), (&from.f00, &mut to.f00)] {
        gpu.write(t, &gpu.download(f)?)?;
    }
    for (f, t) in [
        (&from.bf, &mut to.bf),
        (&from.ref_value, &mut to.ref_value),
        (&from.ref_grad, &mut to.ref_grad),
    ] {
        let v = gpu.download(f)?;
        gpu.write(t, &v[..n_bf])?;
    }
    // valueFraction is SCALAR on a vector field too (SPEC-LIT 120.3).
    {
        let v = gpu.download(&from.fr)?;
        gpu.write(&mut to.fr, &v[..n_bf])?;
    }
    Ok(())
}

/// Copy a closed run's state onto the merged mesh's state (SPEC-LIT §158.1).
/// Refused by name on a mismatch: the cell counts, the two mdot lengths, and
/// the two boundary lengths against the shared prefix and `prefix + 2m`.
pub fn hand_over_state(
    gpu: &Gpu,
    split: &VentSplit,
    from: &CompressibleState,
    to: &mut CompressibleState,
) -> Result<()> {
    let n_bf = split.n_boundary_merged;
    let m = split.faces.len();
    let refuse = |what: String| {
        Error::Refused(format!("vent: the hand-over refused: {what} (SPEC-LIT §158.8)"))
    };
    if from.u.n_cells != to.u.n_cells
        || from.p.n_cells != to.p.n_cells
        || from.t.n_cells != to.t.n_cells
    {
        return Err(refuse("the two meshes hold different cell counts".into()));
    }
    if from.mdot.f.len() != split.kept.len() {
        return Err(refuse(format!(
            "the carried mdot holds {} faces, the split mesh kept {}",
            from.mdot.f.len(),
            split.kept.len()
        )));
    }
    if to.mdot.f.len() != split.n_internal_merged {
        return Err(refuse(format!(
            "the merged mdot holds {} faces, the merged mesh has {}",
            to.mdot.f.len(),
            split.n_internal_merged
        )));
    }
    let to_scalars = [&to.p, &to.t, &to.rho, &to.psi, &to.mu, &to.kappa];
    if to.u.n_boundary_faces != n_bf || to_scalars.iter().any(|f| f.n_boundary_faces != n_bf) {
        return Err(refuse(format!(
            "the merged state's boundary does not hold the shared prefix of {n_bf} faces"
        )));
    }
    let from_scalars = [&from.p, &from.t, &from.rho, &from.psi, &from.mu, &from.kappa];
    let want = n_bf + 2 * m;
    if from.u.n_boundary_faces != want
        || from_scalars.iter().any(|f| f.n_boundary_faces != want)
    {
        return Err(refuse(format!(
            "the carried state's boundary does not hold {n_bf} + 2 x {m} faces"
        )));
    }
    if to.mdot.bf.len() != n_bf || from.mdot.bf.len() != n_bf + 2 * m {
        return Err(refuse(format!(
            "the mdot boundary lengths are {} and {}, the split mesh has \
             {n_bf} + 2 x {m}",
            to.mdot.bf.len(),
            from.mdot.bf.len()
        )));
    }
    hand_over_vector(gpu, &from.u, &mut to.u, n_bf)?;
    hand_over_scalar(gpu, &from.p, &mut to.p, n_bf)?;
    hand_over_scalar(gpu, &from.t, &mut to.t, n_bf)?;
    hand_over_scalar(gpu, &from.rho, &mut to.rho, n_bf)?;
    hand_over_scalar(gpu, &from.psi, &mut to.psi, n_bf)?;
    hand_over_scalar(gpu, &from.mu, &mut to.mu, n_bf)?;
    hand_over_scalar(gpu, &from.kappa, &mut to.kappa, n_bf)?;
    // The face mass flux: kept faces copy to their merged ids, M's faces stay
    // exactly 0.0 (SPEC-LIT §158.1).
    let mdot_f = gpu.download(&from.mdot.f)?;
    let mut out = vec![Scalar::default(); to.mdot.f.len()];
    for (i, &k) in split.kept.iter().enumerate() {
        out[k] = mdot_f[i];
    }
    gpu.write(&mut to.mdot.f, &out)?;
    let mdot_bf = gpu.download(&from.mdot.bf)?;
    gpu.write(&mut to.mdot.bf, &mdot_bf[..n_bf])?;
    Ok(())
}

/// A handed-over run continues its step count and both ddt clocks (SPEC-LIT
/// §158.8).
pub fn hand_over_solver(from: &Compressible<'_>, to: &mut Compressible<'_>) {
    to.set_steps_taken(from.steps_taken());
    to.mom.ddt.state = from.mom.ddt.state;
    to.energy.ddt.state = from.energy.ddt.state;
}

/// The opening: build the merged-mesh state and solver from `low`, carry the
/// closed run's state across, and hand the solver over. Opened on the initial
/// state (`steps_taken() == 0`) the membrane never existed, so the run is bit
/// for bit the run without one: `initialise` only. Otherwise the state is
/// carried, `initialise` derives the gas state and the boundary values from
/// it, the state is carried AGAIN - restoring the mdot and the old levels
/// `initialise` re-seeds - and the solver handed over (SPEC-LIT §158.1).
pub fn open_onto<'m>(
    gpu: &Gpu,
    split: &VentSplit,
    from: &Compressible<'_>,
    from_state: &CompressibleState,
    merged: &'m GpuMesh,
    merged_hm: &HostMesh,
    ctrl: CompressibleControls,
    low: &Lowered,
) -> Result<(Compressible<'m>, CompressibleState)> {
    let mut state = setup_state(gpu, merged, merged_hm, low)?;
    let mut solver = build_solver(gpu, merged, ctrl, low)?;
    if from.steps_taken() == 0 {
        solver.initialise(gpu, &mut state)?;
        return Ok((solver, state));
    }
    hand_over_state(gpu, split, from_state, &mut state)?;
    solver.initialise(gpu, &mut state)?;
    hand_over_state(gpu, split, from_state, &mut state)?;
    hand_over_solver(from, &mut solver);
    Ok((solver, state))
}

#[cfg(test)]
mod tests {
    //! DST-09's tests, written once in `tests/shared/dust_vent_cases.rs` and included here and by
    //! `tests/dust_vent.rs`, which runs the same tests in both builds (SPEC-LIT §158.5, §158.8).

    use crate as ofgpu;

    include!("../../tests/shared/dust_vent_cases.rs");
}
