// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! The mesh quality gate of SPEC-LIT §92.3 - the seven checks a mesh must
//! hold before this crate will solve on it.
//!
//! Written from: ofgpu `SPEC-LIT.md` §92.3 (the gate, equations
//! (92.11)-(92.15), the repair sequence, and the refusal message format
//! fixed there so tests can assert it).
//!
//! ORIGINAL - the seven checks are this project's own, stated in §92.3: G1
//! positive volume, G2 closure, G3 one cell region, G4 non-orthogonality, G5
//! thickness, G6 gradient conditioning, G7 addressing. Nothing here is new
//! physics; what is new is that the answers are a precondition on leaving
//! the mesher, not a report the user is invited to read.
//!
//! # The shape of it
//!
//! [`measure`] reports and never refuses - a bad mesh is what it measured.
//! [`check`] is `measure` plus the refusal §92.3's repair sequence ends in:
//! the gate that failed, the cell ids, their centroids to six figures, the
//! measured values and the thresholds.
//!
//! No GPL-licensed source was consulted.

use std::collections::BTreeMap;

use crate::error::{Error, Result};
use crate::io::polymesh::PolyMeshRaw;
use crate::mesh::HostMesh;
use crate::{Label, Scalar, Vec3};

/// At most this many failing cells are recorded per gate. A refusal names
/// cells; it does not enumerate a mesh.
pub const GATE_CELL_CAP: usize = 200;

/// G5 groups a cell's faces into planar face groups by outward unit normal:
/// a face joins the first group whose representative normal agrees with it to
/// within this angle (§92.3 (92.14)). The gate compares against this angle's
/// cosine; exactly coplanar split faces need only exact agreement, and the
/// smallest separation between two genuinely distinct faces of a hex or a cut
/// cell is tens of degrees.
pub const PLANAR_GROUP_DEG: Scalar = 5.0;

// ==========================================================================
//  The gates
// ==========================================================================

/// The seven checks of §92.3, in gate order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Gate {
    /// (92.11) `V_c > 0` for every cell.
    Volume,
    /// (92.12) `|sum_f s_cf Sf| / V_c^(2/3) < 1e-10`.
    Closure,
    /// `mesh::geometry::cell_regions` leaves the mesh in one piece.
    Regions,
    /// (92.13) face normal against the owner-neighbour line.
    NonOrth,
    /// (92.14) the dimensionless thickness ratio.
    Thickness,
    /// (92.15) the Green-Gauss area tensor's condition number.
    Conditioning,
    /// §1's addressing contract - the lower/diagonal/upper storage and the
    /// upper-triangular order §74.2 emits against - stated as a check.
    Addressing,
}

impl Gate {
    /// The name a refusal prints: `G1 (positive volume)` and so on.
    pub fn name(self) -> &'static str {
        match self {
            Gate::Volume => "G1 (positive volume)",
            Gate::Closure => "G2 (closure)",
            Gate::Regions => "G3 (one cell region)",
            Gate::NonOrth => "G4 (non-orthogonality)",
            Gate::Thickness => "G5 (thickness)",
            Gate::Conditioning => "G6 (gradient conditioning)",
            Gate::Addressing => "G7 (addressing)",
        }
    }

    /// What the gate names in its refusal block: a cell for G1, G2, G5 and
    /// G6; a face for G4 and G7. G3 names the mesh and never lists any.
    pub fn subject(self) -> Subject {
        match self {
            Gate::Volume | Gate::Closure | Gate::Thickness | Gate::Conditioning => {
                Subject::Cell
            }
            Gate::NonOrth | Gate::Addressing => Subject::Face,
            Gate::Regions => Subject::Cell,
        }
    }
}

/// What a gate names: a cell for G1, G2, G5 and G6; a face for G4 and G7.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Subject {
    Cell,
    Face,
}

/// The noun a refusal's header counts: `cell` or `face`.
fn noun(s: Subject) -> &'static str {
    match s {
        Subject::Cell => "cell",
        Subject::Face => "face",
    }
}

/// The gate's numbers. Each is quoted from §92.3 rather than re-chosen there,
/// so this type only carries them.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QualityThresholds {
    /// G2: a closure error above this means a mis-wound face.
    /// `mesh::geometry::CLOSURE_LIMIT`, the crate's own, quoted; §10 states
    /// no number.
    pub max_closure: Scalar,
    /// G4: where §2.4's explicit non-orthogonal correction stops being a
    /// correction and starts being the whole term (`1/cos(70 deg)` is 2.9).
    pub max_non_orth_deg: Scalar,
    /// G4: one face past this is a curiosity, four thousand is a mesh.
    pub report_non_orth_deg: Scalar,
    /// G5: (92.14)'s ratio - a cube is 3, the ammonia mesh's 5 cm cell over
    /// 13 m was 0.0115.
    pub min_thickness_ratio: Scalar,
    /// G6: four decades of the six an f64 residual can afford to lose.
    pub max_cond: Scalar,
}

impl Default for QualityThresholds {
    fn default() -> Self {
        Self {
            max_closure: 1e-10,
            max_non_orth_deg: 70.0,
            report_non_orth_deg: 60.0,
            min_thickness_ratio: 0.05,
            max_cond: 1e4,
        }
    }
}

/// One subject a gate failed on.
#[derive(Debug, Clone)]
pub struct BadSubject {
    /// Cell id, or face id for G4 and G7.
    pub id: usize,
    /// The cell centre, or the face centre for G4. `None` for G7, which runs
    /// before any geometry is built and has no centroid to print.
    pub centre: Option<Vec3>,
    /// The measured value; 0.0 for G7, whose failure is categorical.
    pub value: Scalar,
    /// G4's `between cell 0 and cell 1`, G7's `repeats the point set of face
    /// 7`. Empty where the line is value-and-threshold alone.
    pub why: String,
}

/// One gate's failure: the gate, the threshold it was measured against, the
/// subjects that failed it - empty when the failure is mesh-wide, as G3's is
/// - and a note carrying what the subjects cannot.
#[derive(Debug, Clone)]
pub struct GateFailure {
    pub gate: Gate,
    pub threshold: Scalar,
    /// How many subjects failed this gate IN TOTAL. `subjects` holds at most
    /// [`GATE_CELL_CAP`] of them, so this is the number a refusal prints -
    /// "failed on 200 cell(s)" on a mesh with five thousand bad cells would
    /// be a false report, not a short one.
    pub n_failed: usize,
    /// The recorded prefix of the failing subjects, capped at
    /// [`GATE_CELL_CAP`]. A refusal names subjects; it does not enumerate a
    /// mesh.
    pub subjects: Vec<BadSubject>,
    pub note: String,
}

/// What [`measure`] found. It never refuses: a bad mesh is what it reports,
/// and [`check`] is the one that turns `failures` into an error.
#[derive(Debug, Clone)]
pub struct QualityReport {
    pub n_cells: usize,
    pub n_internal_faces: usize,
    pub n_boundary_faces: usize,
    pub n_points: usize,

    pub min_volume: Scalar,
    pub min_volume_cell: usize,

    pub max_closure: Scalar,
    pub max_closure_cell: usize,

    pub n_regions: usize,
    /// Cells per region, largest first - `MeshReport`'s own.
    pub region_sizes: Vec<usize>,

    pub max_non_orth_deg: Scalar,
    pub mean_non_orth_deg: Scalar,
    /// Faces past `QualityThresholds::report_non_orth_deg`, which §92.3 has
    /// reported rather than refused: one face at 68 degrees is a curiosity,
    /// four thousand is a mesh.
    pub n_non_orth_over_report: usize,

    pub min_thickness_ratio: Scalar,
    pub min_thickness_cell: usize,

    pub max_cond: Scalar,
    pub max_cond_cell: usize,

    pub n_duplicate_faces: usize,
    pub ldu_ordered: bool,

    pub failures: Vec<GateFailure>,
}

impl QualityReport {
    /// `true` when every gate passed and the mesher may emit the mesh.
    pub fn passed(&self) -> bool {
        self.failures.is_empty()
    }

    /// A few lines of the measured numbers, for a run log.
    pub fn summary(&self) -> String {
        let mut s = String::new();
        s.push_str(&format!(
            "automesher: {} cells, {} internal faces, {} boundary faces, \
             {} points\n",
            self.n_cells,
            self.n_internal_faces,
            self.n_boundary_faces,
            self.n_points
        ));
        s.push_str(&format!(
            "  volume: min {:.6} (cell {}); regions: {} {}\n",
            self.min_volume,
            self.min_volume_cell,
            self.n_regions,
            crate::mesh::geometry::region_sizes_text(&self.region_sizes)
        ));
        s.push_str(&format!(
            "  closure: max {:.3e} (cell {})\n",
            self.max_closure, self.max_closure_cell
        ));
        s.push_str(&format!(
            "  non-orthogonality: max {:.3} deg, mean {:.3} deg, {} face(s) \
             past the report mark\n",
            self.max_non_orth_deg,
            self.mean_non_orth_deg,
            self.n_non_orth_over_report
        ));
        s.push_str(&format!(
            "  thickness: min tau {:.6} (cell {}); conditioning: max cond \
             {:.3} (cell {})\n",
            self.min_thickness_ratio,
            self.min_thickness_cell,
            self.max_cond,
            self.max_cond_cell
        ));
        s.push_str(&format!(
            "  duplicate faces: {}; ldu ordered: {}; gate: {}",
            self.n_duplicate_faces,
            if self.ldu_ordered { "yes" } else { "no" },
            if self.passed() { "passed" } else { "FAILED" }
        ));
        s
    }
}

impl QualityReport {
    /// The refusal §92.3's repair sequence ends in: one block per failed
    /// gate, in gate order, naming the gate, the subject ids, their centres
    /// to six figures where geometry exists, the measured values and the
    /// thresholds, each gate in its own words. At most ten subjects are
    /// listed per block; the rest are counted.
    pub fn refusal_text(&self) -> String {
        if self.failures.is_empty() {
            return String::from("automesher: quality gate passed");
        }
        let mut blocks: Vec<String> = Vec::with_capacity(self.failures.len());
        for f in &self.failures {
            let mut b = String::new();
            if f.subjects.is_empty() {
                b.push_str(&format!(
                    "automesher: quality gate {} failed: {}",
                    f.gate.name(),
                    f.note
                ));
                blocks.push(b);
                continue;
            }
            let total = f.n_failed.max(f.subjects.len());
            b.push_str(&format!(
                "automesher: quality gate {} failed on {} {}(s)",
                f.gate.name(),
                total,
                noun(f.gate.subject())
            ));
            let listed = f.subjects.len().min(10);
            for s in &f.subjects[..listed] {
                let c = s.centre.unwrap_or(Vec3::ZERO);
                match f.gate {
                    Gate::Volume => b.push_str(&format!(
                        "\n  cell {} at ({:.6}, {:.6}, {:.6}): \
                         V = {:.6e}, need > 0",
                        s.id, c.x, c.y, c.z, s.value
                    )),
                    Gate::Closure => b.push_str(&format!(
                        "\n  cell {} at ({:.6}, {:.6}, {:.6}): \
                         E = {:.3e}, need < {:e}",
                        s.id, c.x, c.y, c.z, s.value, f.threshold
                    )),
                    Gate::NonOrth => b.push_str(&format!(
                        "\n  face {} at ({:.6}, {:.6}, {:.6}) {}: \
                         theta = {:.3}, need < {}",
                        s.id, c.x, c.y, c.z, s.why, s.value, f.threshold
                    )),
                    Gate::Thickness => b.push_str(&format!(
                        "\n  cell {} at ({:.6}, {:.6}, {:.6}): \
                         tau = {:.6}, need >= {}",
                        s.id, c.x, c.y, c.z, s.value, f.threshold
                    )),
                    Gate::Conditioning => b.push_str(&format!(
                        "\n  cell {} at ({:.6}, {:.6}, {:.6}): \
                         cond = {:.3e}, need < {:e}",
                        s.id, c.x, c.y, c.z, s.value, f.threshold
                    )),
                    Gate::Addressing => b.push_str(&format!(
                        "\n  face {}: {}",
                        s.id, s.why
                    )),
                    // G3 carries no subjects; the mesh-wide branch above ran.
                    Gate::Regions => {}
                }
            }
            if total > listed {
                b.push_str(&format!("\n  ... and {} more", total - listed));
            }
            if !f.note.is_empty() {
                b.push_str(&format!("\n  {}", f.note));
            }
            blocks.push(b);
        }
        blocks.join("\n")
    }
}

/// `1, 2, 7` - at most ten ids, then `...`.
fn list_faces(ids: &[usize]) -> String {
    let mut parts: Vec<String> =
        ids.iter().take(10).map(|f| f.to_string()).collect();
    if ids.len() > 10 {
        parts.push(String::from("..."));
    }
    parts.join(", ")
}

/// §92.3 (92.14)'s `A_max(c)`: the summed `|Sf|` of the largest PLANAR FACE
/// GROUP, not the largest single face. `faces` carries `(n_c(f), |Sf|)` per
/// face of the cell, `n_c(f)` its OUTWARD unit normal at this cell; a face
/// joins the first group whose representative normal `g` satisfies
/// `g . n_c(f) >= cos [`PLANAR_GROUP_DEG`]`, else starts its own group.
/// Grouping by outward normal is what keeps a slab's top and bottom two
/// groups instead of one of twice the area, and four coplanar quarter-faces
/// of a 2:1 interface the one face they are.
fn a_max_planar(faces: &[(Vec3, Scalar)]) -> Scalar {
    let cos_tol = PLANAR_GROUP_DEG.to_radians().cos();
    let mut groups: Vec<(Vec3, Scalar)> = Vec::new();
    for (n, area) in faces {
        match groups
            .iter_mut()
            .find(|(g, _): &&mut (Vec3, Scalar)| g.dot(*n) >= cos_tol)
        {
            Some((_, a)) => *a += *area,
            None => groups.push((*n, *area)),
        }
    }
    groups
        .iter()
        .map(|(_, a)| *a)
        .fold(0.0, Scalar::max)
}

/// Measure everything §92.3 measures, on the mesh as it stands. Never fails
/// on a bad mesh - a bad mesh is what it reports - so the only `Err` is
/// `build_host_mesh`'s, for a mesh too broken to load at all.
pub fn measure(raw: &PolyMeshRaw, t: &QualityThresholds) -> Result<QualityReport> {
    measure_capped(raw, t, GATE_CELL_CAP)
}

/// [`measure`] with the subject cap as a parameter. The cap is a REPORTING
/// cap - a refusal names subjects, it does not enumerate a mesh - and every
/// count in the report is exact whatever it is set to. §92.11's snapping
/// guard (92.31) needs the whole failing set rather than a prefix of it and
/// passes `usize::MAX`; everything a human reads goes through [`measure`].
///
/// Written by the supervising session, not by the coding agent: the body is
/// [`measure`]'s, unchanged except that the six `GATE_CELL_CAP` comparisons
/// now read `cap`.
pub fn measure_capped(
    raw: &PolyMeshRaw,
    t: &QualityThresholds,
    cap: usize,
) -> Result<QualityReport> {
    let n_faces = raw.faces.len();
    let n_if = raw.neighbour.len().min(n_faces);
    let n_bf = n_faces - n_if;

    // ---- G7 first, on the raw mesh ---------------------------------------
    // `build_host_mesh` refuses a mesh that is not in upper-triangular order,
    // and a refusal this gate can name must not surface as an opaque load
    // error. So the addressing sub-checks run on the raw arrays, and only a
    // mesh that passes all four reaches `build_host_mesh`.
    // Pass 1: the FIRST face carrying each sorted point set, and the FIRST
    // internal face carrying each (owner, neighbour) pair.
    let mut point_sets: BTreeMap<Vec<Label>, usize> = BTreeMap::new();
    for f in 0..n_faces {
        let mut key = raw.faces[f].clone();
        key.sort_unstable();
        point_sets.entry(key).or_insert(f);
    }
    let mut pairs: BTreeMap<(Label, Label), usize> = BTreeMap::new();
    for f in 0..n_if {
        pairs.entry((raw.owner[f], raw.neighbour[f])).or_insert(f);
    }

    // Pass 2: every face against both maps, all four conditions at once. A
    // face that fires several sub-checks is ONE failing face carrying all
    // its reasons; the count is exact and only the recorded subjects are
    // capped.
    let (mut n_not_lt, mut n_unsorted, mut n_dup_pairs) =
        (0usize, 0usize, 0usize);
    let mut n_duplicate_faces = 0usize;
    let mut g7_n = 0usize;
    let mut g7_subjects: Vec<BadSubject> = Vec::new();
    for f in 0..n_faces {
        let mut why: Vec<String> = Vec::new();
        if f < n_if {
            let (o, n) = (raw.owner[f], raw.neighbour[f]);
            if o >= n {
                n_not_lt += 1;
                why.push(format!("owner {} >= neighbour {}", o, n));
            }
            if f > 0 && (o, n) <= (raw.owner[f - 1], raw.neighbour[f - 1]) {
                n_unsorted += 1;
                why.push(format!(
                    "out of ascending (owner, neighbour) order after face {}",
                    f - 1
                ));
            }
            if let Some(&first) = pairs.get(&(o, n)) {
                if first != f {
                    n_dup_pairs += 1;
                    why.push(format!(
                        "repeats the (owner, neighbour) pair of face {}",
                        first
                    ));
                }
            }
        }
        let mut key = raw.faces[f].clone();
        key.sort_unstable();
        if let Some(&first) = point_sets.get(&key) {
            if first != f {
                n_duplicate_faces += 1;
                why.push(format!("repeats the point set of face {}", first));
            }
        }
        if !why.is_empty() {
            g7_n += 1;
            if g7_subjects.len() < cap {
                g7_subjects.push(BadSubject {
                    id: f,
                    centre: None,
                    value: 0.0,
                    why: why.join("; "),
                });
            }
        }
    }
    let ldu_ordered = n_not_lt == 0 && n_unsorted == 0;

    if g7_n > 0 {
        // No geometry was built, so nothing past G7 is measured. Zeroes, and
        // the note says so - a NaN would print as a number that was read.
        let note = format!(
            "face(s) with owner >= neighbour: {}; internal face(s) out of \
             ascending (owner, neighbour) order: {}; face(s) repeating an \
             earlier face's point set: {}; internal face(s) repeating an \
             earlier (owner, neighbour) pair: {}; no geometry was measured, \
             so every measured number in this report is 0.0",
            n_not_lt, n_unsorted, n_duplicate_faces, n_dup_pairs
        );
        let n_cells = raw
            .owner
            .iter()
            .chain(raw.neighbour.iter())
            .fold(0i64, |acc, &l| acc.max(i64::from(l) + 1)) as usize;
        return Ok(QualityReport {
            n_cells,
            n_internal_faces: n_if,
            n_boundary_faces: n_bf,
            n_points: raw.points.len(),
            min_volume: 0.0,
            min_volume_cell: 0,
            max_closure: 0.0,
            max_closure_cell: 0,
            n_regions: 0,
            region_sizes: Vec::new(),
            max_non_orth_deg: 0.0,
            mean_non_orth_deg: 0.0,
            n_non_orth_over_report: 0,
            min_thickness_ratio: 0.0,
            min_thickness_cell: 0,
            max_cond: 0.0,
            max_cond_cell: 0,
            n_duplicate_faces,
            ldu_ordered,
            failures: vec![GateFailure {
                gate: Gate::Addressing,
                threshold: 0.0,
                n_failed: g7_n,
                subjects: g7_subjects,
                note,
            }],
        });
    }

    let m: HostMesh = crate::io::polymesh::build_host_mesh(raw)?;
    let r = m.check();
    let n_cells = m.n_cells;
    let mut failures: Vec<GateFailure> = Vec::new();

    // ---- G1 positive volume (92.11) ---------------------------------------
    // A negative volume is an inverted cell, and every flux through it has
    // the wrong sign.
    let mut min_volume = Scalar::INFINITY;
    let mut min_volume_cell = 0usize;
    let mut g1_cells: Vec<BadSubject> = Vec::new();
    let mut g1_n = 0usize;
    for c in 0..n_cells {
        if m.v[c] < min_volume {
            min_volume = m.v[c];
            min_volume_cell = c;
        }
        if m.v[c] <= 0.0 {
            g1_n += 1;
            if g1_cells.len() < cap {
                g1_cells.push(BadSubject {
                    id: c,
                    centre: Some(m.c[c]),
                    value: m.v[c],
                    why: String::new(),
                });
            }
        }
    }
    if g1_n > 0 {
        failures.push(GateFailure {
            gate: Gate::Volume,
            threshold: 0.0,
            n_failed: g1_n,
            subjects: g1_cells,
            note: String::new(),
        });
    }

    // ---- G2 closure (92.12) ------------------------------------------------
    // The report carries `MeshReport`'s worst closure; the failure names
    // every cell past the tolerance, which the report does not.
    let mut max_closure_e = 0.0;
    let mut max_closure_cell = 0usize;
    let mut g2_cells: Vec<BadSubject> = Vec::new();
    let mut g2_n = 0usize;
    for c in 0..n_cells {
        let v = m.v[c];
        if v <= 0.0 {
            continue; // G1 already has these
        }
        let mut s = Vec3::ZERO;
        for k in m.cf_offset[c] as usize..m.cf_offset[c + 1] as usize {
            let f = m.cf_face[k] as usize;
            let sign = if m.cf_own[k] != 0 { 1.0 } else { -1.0 };
            s += m.sf[f] * sign;
        }
        for k in m.bcf_offset[c] as usize..m.bcf_offset[c + 1] as usize {
            s += m.b_sf[m.bcf_face[k] as usize]; // boundary faces are +1
        }
        let e = s.mag() / v.powf(2.0 / 3.0);
        if e > max_closure_e {
            max_closure_e = e;
            max_closure_cell = c;
        }
        if e >= t.max_closure {
            g2_n += 1;
            if g2_cells.len() < cap {
                g2_cells.push(BadSubject {
                    id: c,
                    centre: Some(m.c[c]),
                    value: e,
                    why: String::new(),
                });
            }
        }
    }
    if g2_n > 0 {
        failures.push(GateFailure {
            gate: Gate::Closure,
            threshold: t.max_closure,
            n_failed: g2_n,
            subjects: g2_cells,
            note: String::new(),
        });
    }

    // ---- G3 one region ------------------------------------------------------
    // A sealed pocket's pressure equation is singular up to a constant; the
    // solve stalls or wanders. The gate's job is to make the mesher unable
    // to emit one, so the failure is mesh-wide and carries no cells.
    // `MeshReport` already walked the regions; taking `r.n_regions` keeps
    // the mesh from being swept a second time for a number in hand.
    let n_regions = r.n_regions;
    let region_sizes = r.region_sizes.clone();
    if n_regions != 1 {
        failures.push(GateFailure {
            gate: Gate::Regions,
            threshold: 1.0,
            n_failed: 0,
            subjects: Vec::new(),
            note: format!(
                "the mesh is {} region(s): {}",
                n_regions,
                crate::mesh::geometry::region_sizes_text(&region_sizes)
            ),
        });
    }

    // ---- G4 non-orthogonality (92.13) ---------------------------------------
    // The report's non-orth numbers come from THIS loop, so the summary and
    // the refusal can never disagree: max and mean are over the internal
    // faces the loop measured, and the mean is 0.0 when it measured none.
    let mut max_non_orth = 0.0;
    let mut theta_sum = 0.0;
    let mut n_theta = 0usize;
    let mut n_over_report = 0usize;
    let mut g4_n = 0usize;
    let mut g4_subjects: Vec<BadSubject> = Vec::new();
    for f in 0..m.n_internal_faces {
        let (p, nb) = (m.owner[f] as usize, m.neighbour[f] as usize);
        let (Some(&cp), Some(&cn)) = (m.c.get(p), m.c.get(nb)) else {
            continue;
        };
        let mag_s = m.mag_sf[f];
        let d = cn - cp;
        let mag_d = d.mag();
        if mag_s <= 0.0 || mag_d <= 0.0 {
            continue;
        }
        let theta = (m.sf[f].dot(d) / (mag_s * mag_d))
            .clamp(-1.0, 1.0)
            .acos()
            .to_degrees();
        theta_sum += theta;
        n_theta += 1;
        if theta > max_non_orth {
            max_non_orth = theta;
        }
        if theta > t.report_non_orth_deg {
            n_over_report += 1;
        }
        if theta >= t.max_non_orth_deg {
            g4_n += 1;
            if g4_subjects.len() < cap {
                g4_subjects.push(BadSubject {
                    id: f, // the subject is the FACE, not its owner
                    centre: Some(m.cf[f]),
                    value: theta,
                    why: format!("between cell {} and cell {}", p, nb),
                });
            }
        }
    }
    if g4_n > 0 {
        let ids: Vec<usize> = g4_subjects.iter().map(|s| s.id).collect();
        failures.push(GateFailure {
            gate: Gate::NonOrth,
            threshold: t.max_non_orth_deg,
            n_failed: g4_n,
            subjects: g4_subjects,
            note: format!("face(s) past the limit: {}", list_faces(&ids)),
        });
    }

    // ---- G5 thickness (92.14) -----------------------------------------------
    // 3 V_c / A_max^(3/2) is the thickness of the cell in the direction that
    // matters, made dimensionless by the side of the square with its largest
    // planar face group's area. A cube is 3; a plate of thickness t spanning
    // L is 3t/L.
    let mut min_tau = Scalar::INFINITY;
    let mut min_tau_cell = 0usize;
    let mut g5_cells: Vec<BadSubject> = Vec::new();
    let mut g5_n = 0usize;
    for c in 0..n_cells {
        let v = m.v[c];
        if v <= 0.0 {
            continue; // G1 already has these
        }
        // The cell's faces as (outward unit normal, |Sf|): +Sf where the cell
        // owns the face, -Sf where it neighbours it; boundary faces are
        // always outward. Faces with |Sf| <= 0 are skipped.
        let mut c_faces: Vec<(Vec3, Scalar)> = Vec::new();
        for k in m.cf_offset[c] as usize..m.cf_offset[c + 1] as usize {
            let f = m.cf_face[k] as usize;
            let mag = m.mag_sf[f];
            if mag <= 0.0 {
                continue;
            }
            let sign = if m.cf_own[k] != 0 { 1.0 } else { -1.0 };
            c_faces.push((m.sf[f] * (sign / mag), mag));
        }
        for k in m.bcf_offset[c] as usize..m.bcf_offset[c + 1] as usize {
            let bf = m.bcf_face[k] as usize;
            let mag = m.b_mag_sf[bf];
            if mag <= 0.0 {
                continue;
            }
            c_faces.push((m.b_sf[bf] * (1.0 / mag), mag));
        }
        let a_max = a_max_planar(&c_faces);
        if a_max <= 0.0 {
            continue;
        }
        let tau = 3.0 * v / a_max.powf(1.5);
        if tau < min_tau {
            min_tau = tau;
            min_tau_cell = c;
        }
        if tau < t.min_thickness_ratio {
            g5_n += 1;
            if g5_cells.len() < cap {
                g5_cells.push(BadSubject {
                    id: c,
                    centre: Some(m.c[c]),
                    value: tau,
                    why: String::new(),
                });
            }
        }
    }
    if g5_n > 0 {
        failures.push(GateFailure {
            gate: Gate::Thickness,
            threshold: t.min_thickness_ratio,
            n_failed: g5_n,
            subjects: g5_cells,
            note: String::new(),
        });
    }

    // ---- G6 gradient conditioning (92.15) ------------------------------------
    // T_c is the Green-Gauss gradient's own operator. A cell whose faces are
    // nearly coplanar makes it nearly singular, and the reconstructed
    // gradient in the collapsed direction is noise divided by nothing. A
    // cube gives T_c = 2 h^2 I, cond = 1 exactly (§92.7).
    let mut max_cond = 0.0;
    let mut max_cond_cell = 0usize;
    let mut g6_cells: Vec<BadSubject> = Vec::new();
    let mut g6_n = 0usize;
    for c in 0..n_cells {
        if m.v[c] <= 0.0 {
            continue; // G1 already has these
        }
        let (mut t00, mut t01, mut t02) = (0.0, 0.0, 0.0);
        let (mut t11, mut t12, mut t22) = (0.0, 0.0, 0.0);
        for k in m.cf_offset[c] as usize..m.cf_offset[c + 1] as usize {
            let f = m.cf_face[k] as usize;
            let s = m.sf[f];
            let mag = m.mag_sf[f];
            if mag <= 0.0 {
                continue;
            }
            t00 += s.x * s.x / mag;
            t01 += s.x * s.y / mag;
            t02 += s.x * s.z / mag;
            t11 += s.y * s.y / mag;
            t12 += s.y * s.z / mag;
            t22 += s.z * s.z / mag;
        }
        for k in m.bcf_offset[c] as usize..m.bcf_offset[c + 1] as usize {
            let bf = m.bcf_face[k] as usize;
            let s = m.b_sf[bf];
            let mag = m.b_mag_sf[bf];
            if mag <= 0.0 {
                continue;
            }
            t00 += s.x * s.x / mag;
            t01 += s.x * s.y / mag;
            t02 += s.x * s.z / mag;
            t11 += s.y * s.y / mag;
            t12 += s.y * s.z / mag;
            t22 += s.z * s.z / mag;
        }
        let a = [[t00, t01, t02], [t01, t11, t12], [t02, t12, t22]];
        let [e1, e2, e3] = sym3_eigenvalues(&a);
        let lam_min = e1.min(e2).min(e3);
        let lam_max = e1.max(e2).max(e3);
        let cond = if lam_min <= 0.0 {
            Scalar::INFINITY
        } else {
            lam_max / lam_min
        };
        if cond > max_cond {
            max_cond = cond;
            max_cond_cell = c;
        }
        if cond >= t.max_cond {
            g6_n += 1;
            if g6_cells.len() < cap {
                g6_cells.push(BadSubject {
                    id: c,
                    centre: Some(m.c[c]),
                    value: cond,
                    why: String::new(),
                });
            }
        }
    }
    if g6_n > 0 {
        failures.push(GateFailure {
            gate: Gate::Conditioning,
            threshold: t.max_cond,
            n_failed: g6_n,
            subjects: g6_cells,
            note: String::new(),
        });
    }

    // G7 passed on the raw arrays, or the early return above ran; what is
    // left records that fact alongside the rest of the measurements.
    Ok(QualityReport {
        n_cells,
        n_internal_faces: m.n_internal_faces,
        n_boundary_faces: m.n_boundary_faces,
        n_points: m.n_points,
        min_volume: if min_volume.is_finite() {
            min_volume
        } else {
            0.0
        },
        min_volume_cell,
        max_closure: max_closure_e,
        max_closure_cell,
        n_regions,
        region_sizes,
        max_non_orth_deg: max_non_orth,
        mean_non_orth_deg: if n_theta > 0 {
            theta_sum / n_theta as Scalar
        } else {
            0.0
        },
        n_non_orth_over_report: n_over_report,
        min_thickness_ratio: if min_tau.is_finite() {
            min_tau
        } else {
            0.0
        },
        min_thickness_cell: min_tau_cell,
        max_cond,
        max_cond_cell,
        n_duplicate_faces,
        ldu_ordered,
        failures,
    })
}

/// `measure`, then the refusal §92.3's repair sequence ends in when any gate
/// failed: `Error::Mesh(refusal_text())`.
pub fn check(raw: &PolyMeshRaw, t: &QualityThresholds) -> Result<QualityReport> {
    let rep = measure(raw, t)?;
    if !rep.passed() {
        return Err(Error::Mesh(rep.refusal_text()));
    }
    Ok(rep)
}

/// The three eigenvalues of a real symmetric 3x3, by the
/// characteristic-polynomial closed form - a trigonometric solution, no
/// iteration, exact enough for a condition number and ORIGINAL (it is the
/// standard solution of the cubic, not anyone's code).
fn sym3_eigenvalues(a: &[[Scalar; 3]; 3]) -> [Scalar; 3] {
    let p1 = a[0][1] * a[0][1] + a[0][2] * a[0][2] + a[1][2] * a[1][2];
    // Already diagonal.
    if p1 <= 0.0 {
        return [a[0][0], a[1][1], a[2][2]];
    }
    let q = (a[0][0] + a[1][1] + a[2][2]) / 3.0;
    let p2 = (a[0][0] - q) * (a[0][0] - q)
        + (a[1][1] - q) * (a[1][1] - q)
        + (a[2][2] - q) * (a[2][2] - q)
        + 2.0 * p1;
    let p = (p2 / 6.0).sqrt();
    let b = [
        [(a[0][0] - q) / p, a[0][1] / p, a[0][2] / p],
        [a[1][0] / p, (a[1][1] - q) / p, a[1][2] / p],
        [a[2][0] / p, a[2][1] / p, (a[2][2] - q) / p],
    ];
    let det_b = b[0][0] * (b[1][1] * b[2][2] - b[1][2] * b[2][1])
        - b[0][1] * (b[1][0] * b[2][2] - b[1][2] * b[2][0])
        + b[0][2] * (b[1][0] * b[2][1] - b[1][1] * b[2][0]);
    let r = (det_b / 2.0).clamp(-1.0, 1.0);
    let phi = r.acos() / 3.0;
    let e1 = q + 2.0 * p * phi.cos();
    let two_pi_3 = 2.0 * (std::f64::consts::PI as Scalar) / 3.0;
    let e3 = q + 2.0 * p * (phi + two_pi_3).cos();
    let e2 = 3.0 * q - e1 - e3;
    [e1, e2, e3]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 4x3x2 block mesh whose cells are 0.25-side cubes: x over [0, 1],
    /// y over [0, 0.75], z over [0, 0.5]. §92.7 states `tau_c` = 3 and
    /// `cond(T_c)` = 1 for a CUBE; a 4x3x2 block over the unit cube would
    /// have 0.25 x 1/3 x 0.5 cells, whose cond is exactly 2, so the domain
    /// is sized to the cell counts and every cell is a cube.
    fn cube_block_mesh() -> PolyMeshRaw {
        use crate::blockgen::{BlockSpec, GradedAxis};
        let names = ["xmin", "xmax", "ymin", "ymax", "zmin", "zmax"]
            .map(String::from);
        let types = ["wall", "wall", "wall", "wall", "wall", "wall"]
            .map(String::from);
        let b = BlockSpec {
            x: GradedAxis {
                lo: 0.0,
                hi: 1.0,
                n: 4,
                expansion: 1.0,
                two_sided: false,
            },
            y: GradedAxis {
                lo: 0.0,
                hi: 0.75,
                n: 3,
                expansion: 1.0,
                two_sided: false,
            },
            z: GradedAxis {
                lo: 0.0,
                hi: 0.5,
                n: 2,
                expansion: 1.0,
                two_sided: false,
            },
            patch_name: names,
            patch_type: types,
            windows: vec![],
            cyclic: vec![],
        };
        crate::blockgen::raw_mesh(&b).unwrap()
    }

    #[test]
    fn block_mesh_passes_every_gate() {
        let raw = cube_block_mesh();
        let rep = check(&raw, &QualityThresholds::default())
            .expect("block mesh must pass");
        assert!(rep.passed(), "{}", rep.refusal_text());
        assert_eq!(rep.n_regions, 1);
        assert!(
            rep.max_non_orth_deg < 1e-9,
            "max non-orth {} deg",
            rep.max_non_orth_deg
        );
        // §92.7: a cube's tau_c is exactly 3. The loose `> 0.5` this
        // replaced is why a broken G5 reference length survived a
        // mutation test.
        assert!(
            (rep.min_thickness_ratio - 3.0).abs() < 1e-12,
            "min tau {}",
            rep.min_thickness_ratio
        );
        assert!(rep.max_cond < 1.0 + 1e-9, "max cond {}", rep.max_cond);
        assert!(rep.ldu_ordered);
        assert_eq!(rep.n_duplicate_faces, 0);
    }

    /// The gate must also pass a block whose cells are NOT cubes, which is
    /// every real mesh. 4x3x2 over the unit cube gives 0.25 x 1/3 x 0.5
    /// cells, whose area tensor is `diag(2 dy dz, 2 dx dz, 2 dx dy)` and
    /// whose condition number is therefore exactly `(1/3)/(1/6) = 2` - well
    /// inside G6's `1e4`, which is the point: G6 is a collapsed-cell
    /// detector, not an aspect-ratio limit.
    ///
    /// Written by the supervising session, not by the coding agent.
    #[test]
    fn a_non_cubic_block_passes_and_its_cond_is_exactly_two() {
        use crate::blockgen::{BlockSpec, GradedAxis};
        let ax = |hi: Scalar, n: usize| GradedAxis {
            lo: 0.0,
            hi,
            n,
            expansion: 1.0,
            two_sided: false,
        };
        let b = BlockSpec {
            x: ax(1.0, 4),
            y: ax(1.0, 3),
            z: ax(1.0, 2),
            patch_name: ["xmin", "xmax", "ymin", "ymax", "zmin", "zmax"]
                .map(String::from),
            patch_type: ["wall"; 6].map(String::from),
            windows: vec![],
            cyclic: vec![],
        };
        let raw = crate::blockgen::raw_mesh(&b).unwrap();
        let rep = check(&raw, &QualityThresholds::default())
            .expect("a non-cubic block must still pass the gate");
        assert!((rep.max_cond - 2.0).abs() < 1e-9, "cond {}", rep.max_cond);
        assert!(rep.max_non_orth_deg < 1e-9, "{}", rep.max_non_orth_deg);
        assert_eq!(rep.n_regions, 1);
        // tau = 3 V / A_max^(3/2), V = 1/24, A_max = 1/6.
        let tau = 3.0 * (1.0 / 24.0) / (1.0 as Scalar / 6.0).powf(1.5);
        assert!(
            (rep.min_thickness_ratio - tau).abs() < 1e-9,
            "tau {} vs {}",
            rep.min_thickness_ratio,
            tau
        );
    }

    #[test]
    fn a_point_pushed_through_its_opposite_face_fails_g1_naming_the_cell() {
        let mut raw = cube_block_mesh();
        let i = raw
            .points
            .iter()
            .position(|p| p.x == 0.0 && p.y == 0.0 && p.z == 0.0)
            .expect("the block mesh has a point at the origin");
        raw.points[i] = Vec3::new(10.0, 0.0, 0.0);

        let err = check(&raw, &QualityThresholds::default()).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("G1 (positive volume)"), "{msg}");
        assert!(msg.contains("cell "), "{msg}");
        // (92.11) is a strict inequality; the refusal must not say
        // `need >= 0` beside a cell it just refused.
        assert!(msg.contains("V = "), "{msg}");
        assert!(msg.contains("need > 0"), "{msg}");

        let rep = measure(&raw, &QualityThresholds::default()).unwrap();
        assert!(
            rep.failures.iter().any(|f| {
                f.gate == Gate::Volume
                    && f.subjects.iter().any(|c| c.id == 0)
            }),
            "G1 did not name cell 0: {}",
            rep.summary()
        );
    }

    /// A refusal must print how many cells actually failed, not how many
    /// it kept: `cells` is capped at [`GATE_CELL_CAP`], and a mesh with five
    /// thousand inverted cells that reported "failed on 200 cell(s)" would be
    /// a false report rather than a short one.
    ///
    /// Written by the supervising session, not by the coding agent.
    #[test]
    fn a_refusal_prints_the_true_failure_count_not_the_capped_one() {
        let rep = QualityReport {
            n_cells: 5000,
            n_internal_faces: 0,
            n_boundary_faces: 0,
            n_points: 0,
            min_volume: -1.0,
            min_volume_cell: 7,
            max_closure: 0.0,
            max_closure_cell: 0,
            n_regions: 1,
            region_sizes: vec![5000],
            max_non_orth_deg: 0.0,
            mean_non_orth_deg: 0.0,
            n_non_orth_over_report: 0,
            min_thickness_ratio: 3.0,
            min_thickness_cell: 0,
            max_cond: 1.0,
            max_cond_cell: 0,
            n_duplicate_faces: 0,
            ldu_ordered: true,
            failures: vec![GateFailure {
                gate: Gate::Volume,
                threshold: 0.0,
                n_failed: 5000,
                subjects: (0..GATE_CELL_CAP)
                    .map(|c| BadSubject {
                        id: c,
                        centre: Some(Vec3::ZERO),
                        value: -1.0,
                        why: String::new(),
                    })
                    .collect(),
                note: String::new(),
            }],
        };
        let text = rep.refusal_text();
        assert!(text.contains("failed on 5000 cell(s)"), "{text}");
        assert!(text.contains("... and 4990 more"), "{text}");
        assert!(!rep.passed());
    }

    #[test]
    fn two_disjoint_cubes_fail_g3() {
        let mut points: Vec<Vec3> = Vec::new();
        let mut faces: Vec<Vec<Label>> = Vec::new();
        let mut owner: Vec<Label> = Vec::new();
        // Two unit cubes sharing nothing: cube B is A translated by
        // (10, 0, 0), its point labels offset by 8. The winding of the six
        // faces is OUTWARD, as §92.3's hand-built pair is written.
        for (ox, cell) in [(0.0, 0 as Label), (10.0, 1)] {
            let base = points.len() as Label;
            let o = Vec3::new(ox, 0.0, 0.0);
            points.extend_from_slice(&[
                o,
                o + Vec3::new(1.0, 0.0, 0.0),
                o + Vec3::new(1.0, 1.0, 0.0),
                o + Vec3::new(0.0, 1.0, 0.0),
                o + Vec3::new(0.0, 0.0, 1.0),
                o + Vec3::new(1.0, 0.0, 1.0),
                o + Vec3::new(1.0, 1.0, 1.0),
                o + Vec3::new(0.0, 1.0, 1.0),
            ]);
            for ks in [
                [0usize, 3, 2, 1], // z-min
                [4, 5, 6, 7],      // z-max
                [0, 1, 5, 4],      // y-min
                [2, 3, 7, 6],      // y-max
                [0, 4, 7, 3],      // x-min
                [1, 2, 6, 5],      // x-max
            ] {
                faces.push(ks.map(|k| base + k as Label).to_vec());
                owner.push(cell);
            }
        }
        let raw = PolyMeshRaw {
            points,
            faces,
            owner,
            neighbour: Vec::new(),
            patches: vec![crate::mesh::PatchInfo {
                name: String::from("walls"),
                type_name: String::from("wall"),
                kind: crate::mesh::PatchKind::Wall,
                start: 0,
                size: 12,
                nbr_patch: None,
            }],
        };

        let err = check(&raw, &QualityThresholds::default()).unwrap_err();
        assert!(
            err.to_string().contains("G3 (one cell region)"),
            "{}",
            err
        );

        let rep = measure(&raw, &QualityThresholds::default()).unwrap();
        assert_eq!(rep.n_regions, 2);
        // A hand-built pair of cubes is otherwise perfect: if anything but
        // G3 fired, the winding or a formula is wrong.
        assert_eq!(rep.failures.len(), 1, "{}", rep.summary());
        assert_eq!(rep.failures[0].gate, Gate::Regions);
    }

    // ------------------------------------------------------------------------
    // AM-1e: a refusal test per gate. The standard every test here meets: if
    // that gate's threshold were changed so the gate never refuses, the test
    // fails - each one asserts the gate FIRED, not that a number came back.
    // ------------------------------------------------------------------------

    /// A uniform block over [0, xh] x [0, yh] x [0, zh] with xn x yn x zn
    /// cells, all six patches `wall`, no windows, no cyclics - the shape
    /// [`cube_block_mesh`] builds, parameterised.
    fn block(
        xh: Scalar,
        xn: usize,
        yh: Scalar,
        yn: usize,
        zh: Scalar,
        zn: usize,
    ) -> PolyMeshRaw {
        use crate::blockgen::{BlockSpec, GradedAxis};
        let ax = |hi: Scalar, n: usize| GradedAxis {
            lo: 0.0,
            hi,
            n,
            expansion: 1.0,
            two_sided: false,
        };
        let b = BlockSpec {
            x: ax(xh, xn),
            y: ax(yh, yn),
            z: ax(zh, zn),
            patch_name: ["xmin", "xmax", "ymin", "ymax", "zmin", "zmax"]
                .map(String::from),
            patch_type: ["wall"; 6].map(String::from),
            windows: vec![],
            cyclic: vec![],
        };
        crate::blockgen::raw_mesh(&b).unwrap()
    }

    /// The ammonia site's own defect: 5 cm spanning 13 m. §92.3 prints
    /// tau = 0.0115 for it; the gate must refuse it and nothing else about
    /// the block.
    #[test]
    fn the_ammonia_sliver_fails_g5_and_nothing_else() {
        let raw = block(13.0, 1, 13.0, 1, 0.05, 1);
        let rep = measure(&raw, &QualityThresholds::default()).unwrap();
        assert!(
            (rep.min_thickness_ratio - 0.011_538_461_538).abs() < 1e-9,
            "tau {}",
            rep.min_thickness_ratio
        );
        assert_eq!(rep.failures.len(), 1, "{}", rep.summary());
        assert_eq!(rep.failures[0].gate, Gate::Thickness);
        assert_eq!(rep.failures[0].n_failed, 1);
    }

    /// §92.3's planar grouping, and the reason it exists: the SAME 13 x 13 x
    /// 0.05 slab built as ONE cell whose two large faces are four coplanar
    /// quads each must measure the same tau as the whole-face slab of
    /// [`the_ammonia_sliver_fails_g5_and_nothing_else`]. Before the grouping
    /// this measured 0.092308 and passed - a factor of 4^(3/2) = 8 - which
    /// is how the one defect the gate exists to refuse walked through it at
    /// a 2:1 interface.
    #[test]
    fn the_same_sliver_measures_the_same_tau_whole_or_split() {
        let (l, t) = (13.0, 0.05);
        // A 3x3 grid of points on z = 0 (ids 0..9) and its copy at z = t
        // (ids 9..18); point (i, j) is id i + 3 j, at (i l/2, j l/2).
        let mut points: Vec<Vec3> = Vec::new();
        for z in [0.0, t] {
            for j in 0..3 {
                for i in 0..3 {
                    points.push(Vec3::new(
                        i as Scalar * l / 2.0,
                        j as Scalar * l / 2.0,
                        z,
                    ));
                }
            }
        }
        let at = |i: usize, j: usize, top: bool| {
            (i + 3 * j) as Label + if top { 9 } else { 0 }
        };
        let mut q: Vec<[Label; 4]> = Vec::new();
        for j in 0..2 {
            for i in 0..2 {
                // z-min, outward -z; z-max, outward +z.
                q.push([
                    at(i, j, false),
                    at(i, j + 1, false),
                    at(i + 1, j + 1, false),
                    at(i + 1, j, false),
                ]);
                q.push([
                    at(i, j, true),
                    at(i + 1, j, true),
                    at(i + 1, j + 1, true),
                    at(i, j + 1, true),
                ]);
            }
        }
        // The four sides, corner to corner, outward-wound.
        q.push([at(0, 0, false), at(2, 0, false), at(2, 0, true), at(0, 0, true)]);
        q.push([at(0, 2, false), at(0, 2, true), at(2, 2, true), at(2, 2, false)]);
        q.push([at(0, 0, false), at(0, 0, true), at(0, 2, true), at(0, 2, false)]);
        q.push([at(2, 0, false), at(2, 2, false), at(2, 2, true), at(2, 0, true)]);
        let raw = PolyMeshRaw {
            points,
            faces: q.iter().map(|k| k.to_vec()).collect(),
            owner: vec![0; q.len()],
            neighbour: Vec::new(),
            patches: vec![crate::mesh::PatchInfo {
                name: String::from("walls"),
                type_name: String::from("wall"),
                kind: crate::mesh::PatchKind::Wall,
                start: 0,
                size: q.len(),
                nbr_patch: None,
            }],
        };
        let whole = block(13.0, 1, 13.0, 1, 0.05, 1);
        let split = measure(&raw, &QualityThresholds::default()).unwrap();
        let whole = measure(&whole, &QualityThresholds::default()).unwrap();
        assert!(
            (split.min_thickness_ratio - whole.min_thickness_ratio).abs() < 1e-12,
            "split {} vs whole {}",
            split.min_thickness_ratio,
            whole.min_thickness_ratio
        );
        assert!(
            split.failures.iter().any(|f| f.gate == Gate::Thickness),
            "G5 did not fire on the split slab: {}",
            split.summary()
        );
    }

    /// Every internal face of a block sheared by `p.x += 3 p.z` sits at
    /// atan(3) = 71.565 deg, past G4's 70. G4's subject is the FACE, and the
    /// block's two internal faces both fail.
    #[test]
    fn a_sheared_block_fails_g4_naming_the_face() {
        let mut raw = block(1.0, 1, 1.0, 1, 1.0, 3);
        for p in &mut raw.points {
            p.x += 3.0 * p.z;
        }
        let rep = measure(&raw, &QualityThresholds::default()).unwrap();
        assert!(
            (rep.max_non_orth_deg - 71.565).abs() < 1e-2,
            "max non-orth {} deg",
            rep.max_non_orth_deg
        );
        assert_eq!(rep.failures.len(), 1, "{}", rep.summary());
        assert_eq!(rep.failures[0].gate, Gate::NonOrth);
        assert_eq!(rep.failures[0].n_failed, 2);
        assert_eq!(rep.n_non_orth_over_report, 2);
        let text = rep.refusal_text();
        assert!(text.contains("failed on 2 face(s)"), "{text}");
        assert!(text.contains("between cell"), "{text}");
        assert!(text.contains("theta = 71.565, need < 70"), "{text}");
    }

    /// The control that keeps the test above honest: the same block sheared
    /// by 2.5 sits at atan(2.5) = 68.199 deg - past the 60 deg report mark,
    /// inside the 70 deg refusal - so `check` SUCCEEDS while the report
    /// still counts both faces. Without it the test above could pass because
    /// the shear broke something else.
    #[test]
    fn a_shear_below_seventy_is_reported_not_refused() {
        let mut raw = block(1.0, 1, 1.0, 1, 1.0, 3);
        for p in &mut raw.points {
            p.x += 2.5 * p.z;
        }
        let rep = check(&raw, &QualityThresholds::default())
            .expect("68.199 deg is inside the 70 deg refusal");
        assert_eq!(rep.n_non_orth_over_report, 2);
        assert!(
            (rep.max_non_orth_deg - 68.199).abs() < 1e-2,
            "max non-orth {} deg",
            rep.max_non_orth_deg
        );
    }

    /// G2 (closure) is the gate nothing exercised at all before this test.
    /// Reversing internal face 0 un-winds it: the two cells that share it no
    /// longer close, and both are named.
    #[test]
    fn a_reversed_face_winding_fails_g2() {
        let mut raw = cube_block_mesh();
        raw.faces[0].reverse();
        let rep = measure(&raw, &QualityThresholds::default()).unwrap();
        let g2 = rep
            .failures
            .iter()
            .find(|f| f.gate == Gate::Closure)
            .expect("G2 must fire on a reversed face");
        assert_eq!(g2.n_failed, 2, "{}", rep.summary());
        let text = rep.refusal_text();
        assert!(text.contains("E = "), "{text}");
        assert!(text.contains("need < 1e-10"), "{text}");
    }

    /// A 1 x 1 x 1e-6 slab: cond(T_c) = 1e6, a hundred times past G6's 1e4.
    /// G5 fires here too - on a slab that is expected - so G6 is asserted by
    /// name.
    #[test]
    fn a_collapsed_cell_fails_g6() {
        let raw = block(1.0, 1, 1.0, 1, 1.0e-6, 1);
        let rep = measure(&raw, &QualityThresholds::default()).unwrap();
        assert!(
            (rep.max_cond - 1.0e6).abs() < 1.0e-3 * 1.0e6,
            "cond {}",
            rep.max_cond
        );
        assert!(
            rep.failures.iter().any(|f| f.gate == Gate::Conditioning),
            "G6 must fire on the collapsed cell: {}",
            rep.summary()
        );
        assert!(
            rep.refusal_text()
                .contains("cond = 1.000e6, need < 1e4"),
            "{}",
            rep.refusal_text()
        );
    }

    /// SPEC-LIT (92.16): on a box `tau_c * cond(T_c) = 3 sqrt(a/b)`, which
    /// on a slab L x L x t is exactly 3. G5's 0.05 is there cond = 60, so
    /// G6's 1e4 is 167 times looser and can never fire first on a box. This
    /// records what G6 is actually for: the shapes G5 cannot see.
    #[test]
    fn on_a_slab_tau_times_cond_is_exactly_three() {
        for t in [1.0e-1, 1.0e-2, 1.0e-3, 1.0e-4] {
            let raw = block(1.0, 1, 1.0, 1, t, 1);
            let rep = measure(&raw, &QualityThresholds::default()).unwrap();
            let prod = rep.min_thickness_ratio * rep.max_cond;
            assert!(
                (prod - 3.0).abs() < 1e-9,
                "t = {t}: tau {} * cond {} = {}",
                rep.min_thickness_ratio,
                rep.max_cond,
                prod
            );
        }
    }

    /// A duplicated face is two matrix entries for one flux. G7 runs on the
    /// raw arrays, before `build_host_mesh`, and names the face that repeats
    /// an earlier face's point set.
    #[test]
    fn a_duplicated_face_fails_g7_naming_the_face() {
        let mut raw = block(1.0, 2, 1.0, 1, 1.0, 1);
        let n = raw.faces.len();
        raw.faces[n - 1] = raw.faces[n - 2].clone();
        let msg = check(&raw, &QualityThresholds::default())
            .unwrap_err()
            .to_string();
        assert!(msg.contains("G7 (addressing)"), "{msg}");
        assert!(msg.contains("repeats the point set of face"), "{msg}");
    }

    /// A mesh out of upper-triangular order must surface as the G7 refusal
    /// that names the face, not as `build_host_mesh`'s opaque load error.
    #[test]
    fn broken_ldu_order_fails_g7_by_name() {
        let mut raw = block(1.0, 3, 1.0, 1, 1.0, 1);
        raw.owner.swap(0, 1);
        raw.neighbour.swap(0, 1);
        let msg = check(&raw, &QualityThresholds::default())
            .unwrap_err()
            .to_string();
        assert!(msg.contains("G7 (addressing)"), "{msg}");
        assert!(msg.contains("face "), "{msg}");
    }

    /// `n_failed` counts EVERY broken face; `subjects` holds at most
    /// [`GATE_CELL_CAP`] of them. Until today G7 reported the capped 200 as
    /// if it were the total.
    #[test]
    fn g7_counts_every_broken_face_not_only_the_ones_it_lists() {
        let mut raw = block(1.0, 300, 1.0, 1, 1.0, 1);
        let n_if = raw.neighbour.len();
        assert_eq!(n_if, 299);
        for f in 0..n_if {
            let o = raw.owner[f];
            raw.owner[f] = raw.neighbour[f];
            raw.neighbour[f] = o;
        }
        let rep = measure(&raw, &QualityThresholds::default()).unwrap();
        let g7 = rep
            .failures
            .iter()
            .find(|f| f.gate == Gate::Addressing)
            .expect("G7 must fire on every face with owner > neighbour");
        assert_eq!(g7.n_failed, 299);
        assert_eq!(g7.subjects.len(), GATE_CELL_CAP);
        let text = rep.refusal_text();
        assert!(text.contains("failed on 299 face(s)"), "{text}");
    }

    /// §92.3 fixed the refusal line's exact form so that tests could assert
    /// it. This is that line, character for character, on the ammonia
    /// sliver of [`the_ammonia_sliver_fails_g5_and_nothing_else`].
    #[test]
    fn a_refusal_line_is_exactly_what_section_92_3_fixed() {
        let raw = block(13.0, 1, 13.0, 1, 0.05, 1);
        let rep = measure(&raw, &QualityThresholds::default()).unwrap();
        assert_eq!(
            rep.refusal_text(),
            "automesher: quality gate G5 (thickness) failed on 1 cell(s)\n  \
             cell 0 at (6.500000, 6.500000, 0.025000): tau = 0.011538, \
             need >= 0.05"
        );
    }
}
