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

// ==========================================================================
//  The gates
// ==========================================================================

/// The seven checks of §92.3, in gate order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Gate {
    /// (92.11) `V_c > 0` for every cell.
    Volume,
    /// (92.12) `|sum_f s_cf Sf| / V_c^(2/3) < 1e-9`.
    Closure,
    /// `mesh::geometry::cell_regions` leaves the mesh in one piece.
    Regions,
    /// (92.13) face normal against the owner-neighbour line.
    NonOrth,
    /// (92.14) the dimensionless thickness ratio.
    Thickness,
    /// (92.15) the Green-Gauss area tensor's condition number.
    Conditioning,
    /// §2's addressing contract, stated as a check.
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
}

/// The gate's numbers. Each is quoted from §92.3 rather than re-chosen there,
/// so this type only carries them.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QualityThresholds {
    /// G2: a closure error above this means a mis-wound face. §10's own
    /// tolerance, quoted.
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
            max_closure: 1e-9,
            max_non_orth_deg: 70.0,
            report_non_orth_deg: 60.0,
            min_thickness_ratio: 0.05,
            max_cond: 1e4,
        }
    }
}

/// One cell a gate failed on: its id, its centroid, the measured value.
#[derive(Debug, Clone, Copy)]
pub struct BadCell {
    pub cell: usize,
    pub centre: Vec3,
    pub value: Scalar,
}

/// One gate's failure: the gate, the threshold it was measured against, the
/// cells that failed it - empty when the failure is mesh-wide, as G3's is -
/// and a note carrying what the cells cannot.
#[derive(Debug, Clone)]
pub struct GateFailure {
    pub gate: Gate,
    pub threshold: Scalar,
    /// How many cells failed this gate IN TOTAL. `cells` holds at most
    /// [`GATE_CELL_CAP`] of them, so this is the number a refusal prints -
    /// "failed on 200 cell(s)" on a mesh with five thousand bad cells would
    /// be a false report, not a short one.
    pub n_failed: usize,
    /// The recorded prefix of the failing cells, capped at
    /// [`GATE_CELL_CAP`]. A refusal names cells; it does not enumerate a mesh.
    pub cells: Vec<BadCell>,
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

/// The comparison a gate's cell line states. `None` where no per-line
/// threshold makes sense: G3 carries no cells, and G7's cells name faces.
fn need(gate: Gate) -> Option<&'static str> {
    match gate {
        Gate::Volume | Gate::Thickness => Some(">="),
        Gate::Closure | Gate::NonOrth | Gate::Conditioning => Some("<"),
        Gate::Regions | Gate::Addressing => None,
    }
}

impl QualityReport {
    /// The refusal §92.3's repair sequence ends in: one block per failed
    /// gate, in gate order, naming the gate, the cell ids, their centroids
    /// to six figures, the measured values and the thresholds. At most ten
    /// cells are listed per block; the rest are counted.
    pub fn refusal_text(&self) -> String {
        if self.failures.is_empty() {
            return String::from("automesher: quality gate passed");
        }
        let mut blocks: Vec<String> = Vec::with_capacity(self.failures.len());
        for f in &self.failures {
            let mut b = String::new();
            if f.cells.is_empty() {
                b.push_str(&format!(
                    "automesher: quality gate {} failed: {}",
                    f.gate.name(),
                    f.note
                ));
                blocks.push(b);
                continue;
            }
            b.push_str(&format!(
                "automesher: quality gate {} failed on {} cell(s)",
                f.gate.name(),
                f.n_failed.max(f.cells.len())
            ));
            let listed = f.cells.len().min(10);
            for c in &f.cells[..listed] {
                b.push_str(&format!(
                    "\n  cell {} at ({:.6}, {:.6}, {:.6})",
                    c.cell, c.centre.x, c.centre.y, c.centre.z
                ));
                match need(f.gate) {
                    Some(op) => b.push_str(&format!(
                        ": value = {:.6}, need {} {}",
                        c.value, op, f.threshold
                    )),
                    None => b.push_str(": face addressing"),
                }
            }
            let total = f.n_failed.max(f.cells.len());
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

/// Measure everything §92.3 measures, on the mesh as it stands. Never fails
/// on a bad mesh - a bad mesh is what it reports - so the only `Err` is
/// `build_host_mesh`'s, for a mesh too broken to load at all.
pub fn measure(raw: &PolyMeshRaw, t: &QualityThresholds) -> Result<QualityReport> {
    let n_faces = raw.faces.len();
    let n_if = raw.neighbour.len().min(n_faces);
    let n_bf = n_faces - n_if;

    // ---- G7 first, on the raw mesh ---------------------------------------
    // `build_host_mesh` refuses a mesh that is not in upper-triangular order,
    // and a refusal this gate can name must not surface as an opaque load
    // error. So the addressing sub-checks run on the raw arrays, and only a
    // mesh that passes all four reaches `build_host_mesh`.
    let mut not_lt: Vec<usize> = Vec::new();
    let mut unsorted: Vec<usize> = Vec::new();
    let mut dup_pairs: Vec<(usize, usize)> = Vec::new();
    let mut pairs: BTreeMap<(Label, Label), usize> = BTreeMap::new();
    for f in 0..n_if {
        let (o, n) = (raw.owner[f], raw.neighbour[f]);
        if o >= n {
            not_lt.push(f);
        }
        if f > 0 && (o, n) <= (raw.owner[f - 1], raw.neighbour[f - 1]) {
            unsorted.push(f);
        }
        match pairs.get(&(o, n)) {
            Some(&first) => {
                if dup_pairs.len() < GATE_CELL_CAP {
                    dup_pairs.push((f, first));
                }
            }
            None => {
                pairs.insert((o, n), f);
            }
        }
    }

    let mut point_sets: BTreeMap<Vec<Label>, usize> = BTreeMap::new();
    let mut n_duplicate_faces = 0usize;
    let mut dup_faces: Vec<(usize, usize)> = Vec::new();
    for f in 0..n_faces {
        let mut key = raw.faces[f].clone();
        key.sort_unstable();
        match point_sets.get(&key) {
            Some(&first) => {
                n_duplicate_faces += 1;
                if dup_faces.len() < GATE_CELL_CAP {
                    dup_faces.push((f, first));
                }
            }
            None => {
                point_sets.insert(key, f);
            }
        }
    }

    let ldu_ordered = not_lt.is_empty() && unsorted.is_empty();
    let mut g7_notes: Vec<String> = Vec::new();
    let mut g7_cells: Vec<BadCell> = Vec::new();
    if !not_lt.is_empty() {
        g7_notes.push(format!(
            "face(s) with owner >= neighbour: {}",
            list_faces(&not_lt)
        ));
    }
    if !unsorted.is_empty() {
        g7_notes.push(format!(
            "internal face(s) out of ascending (owner, neighbour) order: {}",
            list_faces(&unsorted)
        ));
    }
    let dup_face_ids: Vec<usize> = dup_faces.iter().map(|&(f, _)| f).collect();
    if !dup_face_ids.is_empty() {
        g7_notes.push(format!(
            "face(s) repeating an earlier face's point set: {}",
            list_faces(&dup_face_ids)
        ));
    }
    let dup_pair_ids: Vec<usize> = dup_pairs.iter().map(|&(f, _)| f).collect();
    if !dup_pair_ids.is_empty() {
        g7_notes.push(format!(
            "internal face(s) repeating an earlier (owner, neighbour) pair: {}",
            list_faces(&dup_pair_ids)
        ));
    }
    if !g7_notes.is_empty() {
        for ids in [&not_lt, &unsorted, &dup_face_ids, &dup_pair_ids] {
            for &f in ids.iter().take(GATE_CELL_CAP) {
                g7_cells.push(BadCell {
                    cell: f,
                    centre: Vec3::ZERO,
                    value: 0.0,
                });
            }
        }
    }

    if !g7_notes.is_empty() {
        // No geometry was built, so nothing past G7 is measured. Zeroes, and
        // the note says so - a NaN would print as a number that was read.
        let mut note = g7_notes.join("; ");
        note.push_str("; no geometry was measured, so every measured number \
                       in this report is 0.0");
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
                n_failed: g7_cells.len(),
                cells: g7_cells,
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
    let mut g1_cells: Vec<BadCell> = Vec::new();
    let mut g1_n = 0usize;
    for c in 0..n_cells {
        if m.v[c] <= 0.0 {
            g1_n += 1;
            if g1_cells.len() < GATE_CELL_CAP {
                g1_cells.push(BadCell {
                    cell: c,
                    centre: m.c[c],
                    value: m.v[c],
                });
            }
        }
    }
    if g1_n > 0 {
        failures.push(GateFailure {
            gate: Gate::Volume,
            threshold: 0.0,
            n_failed: g1_n,
            cells: g1_cells,
            note: String::new(),
        });
    }

    // ---- G2 closure (92.12) ------------------------------------------------
    // The report carries `MeshReport`'s worst closure; the failure names
    // every cell past the tolerance, which the report does not.
    let mut g2_cells: Vec<BadCell> = Vec::new();
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
        if e >= t.max_closure {
            g2_n += 1;
            if g2_cells.len() < GATE_CELL_CAP {
                g2_cells.push(BadCell {
                    cell: c,
                    centre: m.c[c],
                    value: e,
                });
            }
        }
    }
    if g2_n > 0 {
        failures.push(GateFailure {
            gate: Gate::Closure,
            threshold: t.max_closure,
            n_failed: g2_n,
            cells: g2_cells,
            note: String::new(),
        });
    }

    // ---- G3 one region ------------------------------------------------------
    // A sealed pocket's pressure equation is singular up to a constant; the
    // solve stalls or wanders. The gate's job is to make the mesher unable
    // to emit one, so the failure is mesh-wide and carries no cells.
    let (n_regions, _) = crate::mesh::geometry::cell_regions(&m);
    let region_sizes = r.region_sizes.clone();
    if n_regions != 1 {
        failures.push(GateFailure {
            gate: Gate::Regions,
            threshold: 1.0,
            n_failed: 0,
            cells: Vec::new(),
            note: format!(
                "the mesh is {} region(s): {}",
                n_regions,
                crate::mesh::geometry::region_sizes_text(&region_sizes)
            ),
        });
    }

    // ---- G4 non-orthogonality (92.13) ---------------------------------------
    let mut g4_cells: Vec<BadCell> = Vec::new();
    let mut g4_faces: Vec<usize> = Vec::new();
    let mut g4_n = 0usize;
    let mut n_over_report = 0usize;
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
        if theta > t.report_non_orth_deg {
            n_over_report += 1;
        }
        if theta >= t.max_non_orth_deg {
            g4_n += 1;
            if g4_cells.len() < GATE_CELL_CAP {
                g4_cells.push(BadCell {
                    cell: p, // the owner, since d starts there
                    centre: cp,
                    value: theta,
                });
                g4_faces.push(f);
            }
        }
    }
    if g4_n > 0 {
        failures.push(GateFailure {
            gate: Gate::NonOrth,
            threshold: t.max_non_orth_deg,
            n_failed: g4_n,
            cells: g4_cells,
            note: format!("face(s) past the limit: {}", list_faces(&g4_faces)),
        });
    }

    // ---- G5 thickness (92.14) -----------------------------------------------
    // 3 V_c / A_max^(3/2) is the thickness of the cell in the direction that
    // matters, made dimensionless by the side of the square with its largest
    // face's area. A cube is 3; a plate of thickness t spanning L is 3t/L.
    let mut min_tau = Scalar::INFINITY;
    let mut min_tau_cell = 0usize;
    let mut g5_cells: Vec<BadCell> = Vec::new();
    let mut g5_n = 0usize;
    for c in 0..n_cells {
        let v = m.v[c];
        if v <= 0.0 {
            continue; // G1 already has these
        }
        let mut a_max: Scalar = 0.0;
        for k in m.cf_offset[c] as usize..m.cf_offset[c + 1] as usize {
            a_max = a_max.max(m.mag_sf[m.cf_face[k] as usize]);
        }
        for k in m.bcf_offset[c] as usize..m.bcf_offset[c + 1] as usize {
            a_max = a_max.max(m.b_mag_sf[m.bcf_face[k] as usize]);
        }
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
            if g5_cells.len() < GATE_CELL_CAP {
                g5_cells.push(BadCell {
                    cell: c,
                    centre: m.c[c],
                    value: tau,
                });
            }
        }
    }
    if g5_n > 0 {
        failures.push(GateFailure {
            gate: Gate::Thickness,
            threshold: t.min_thickness_ratio,
            n_failed: g5_n,
            cells: g5_cells,
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
    let mut g6_cells: Vec<BadCell> = Vec::new();
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
            if g6_cells.len() < GATE_CELL_CAP {
                g6_cells.push(BadCell {
                    cell: c,
                    centre: m.c[c],
                    value: cond,
                });
            }
        }
    }
    if g6_n > 0 {
        failures.push(GateFailure {
            gate: Gate::Conditioning,
            threshold: t.max_cond,
            n_failed: g6_n,
            cells: g6_cells,
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
        min_volume: r.min_volume,
        min_volume_cell: r.min_volume_cell,
        max_closure: r.max_closure_error,
        max_closure_cell: r.max_closure_cell,
        n_regions,
        region_sizes,
        max_non_orth_deg: r.max_non_orth_deg,
        mean_non_orth_deg: r.mean_non_orth_deg,
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
        assert!(
            rep.min_thickness_ratio > 0.5,
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

        let rep = measure(&raw, &QualityThresholds::default()).unwrap();
        assert!(
            rep.failures.iter().any(|f| {
                f.gate == Gate::Volume
                    && f.cells.iter().any(|c| c.cell == 0)
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
                cells: (0..GATE_CELL_CAP)
                    .map(|c| BadCell {
                        cell: c,
                        centre: Vec3::ZERO,
                        value: -1.0,
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
}
