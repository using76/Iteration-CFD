// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! The visualisation lattice of SPEC-LIT section 117: every lattice voxel
//! gets the cell that contains its centre, or [`NONE`].
//!
//! [`VoxelMap::build`] runs once, at setup, before the time loop. It lays a
//! cubic lattice over the mesh ((117.1), (117.2)), coarsens it to the voxel
//! budget ((117.3)-(117.4)), rasterises the cells in ascending order with
//! the first claim winning (pass 1, 117.4), then fills the gaps a warped or
//! non-convex cell left with the nearest-centred cell whose box still holds
//! the voxel and whose boundary faces do not exclude it (pass 2, 117.4).
//! The result is (117.6):
//!
//! ```text
//! M[v] = min { c : inside_c( x_v ) }     (NONE if the set is empty)
//! ```
//!
//! with `inside_c` the half-space test of (117.5) - `parcels::locate_cell`'s
//! test, evaluated in f64 through `HostMesh::geom64()` (117.2), so the map
//! is bit-identical in the f64 and the `single` builds. The refusals are
//! 117.12's first four rows, by name; [`LatticeReport`] is 117.7, key by
//! key. The writers gather through this map (VIO-02 samples, VIO-03
//! writes); neither is implemented here.
//!
//! Written from SPEC-LIT section 117 (117.2-117.4, 117.7).
//! Provenance: ORIGINAL.
//! No GPL-licensed source was consulted.

use std::time::Instant;

use crate::error::{Error, Result};
use crate::io::polymesh::PolyMeshRaw;
use crate::mesh::{Geom64, HostMesh};
use crate::types::DVec3;

/// "No cell owns this voxel."
pub const NONE: u32 = u32::MAX;

/// The default voxel budget (117.3): `isaac_export.py`'s `MAX_VOXELS`.
pub const DEFAULT_MAX_VOXELS: usize = 8_000_000;

/// The largest coarsening factor the budget may push an EXPLICIT `voxel` to
/// (117.3): a lattice 16 times coarser than the one asked for is no longer
/// the one asked for. The default voxel carries no such limit. `s == 16.0`
/// exactly is allowed.
pub const MAX_EXPLICIT_COARSEN: f64 = 16.0;

// ==========================================================================
//  The spec (117.3)
// ==========================================================================

/// Which sampler VIO-02 will apply at each owned voxel (117.5). Carried on
/// the spec; `build` reads it nowhere.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LatticeSample {
    /// `s_v = phi_c`, the solver's own cell value.
    #[default]
    Cell,
    /// Cell value plus the least-squares gradient, clipped (117.7)-(117.8).
    Linear,
}

/// Which voxels VIO-03 will write (117.6). Carried on the spec; `build`
/// reads it nowhere.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LatticeActivation {
    /// Every voxel of the lattice is active; `NONE` voxels hold `background`.
    #[default]
    Dense,
    /// A voxel is active if and only if `M[v] != NONE`.
    Fluid,
}

/// `output.visualisation.lattice` (117.3), as [`VoxelMap::build`] takes it.
#[derive(Debug, Clone, PartialEq)]
pub struct LatticeSpec {
    /// The voxel edge `h`, in metres; `None` = `(min_c V_c)^(1/3)`, the cube
    /// root of the smallest cell volume.
    pub voxel: Option<f64>,
    /// `[xlo, ylo, zlo, xhi, yhi, zhi]`; `None` = the bounding box of
    /// `raw.points` (117.3).
    pub bbox: Option<[f64; 6]>,
    /// The voxel budget; (117.3)-(117.4) coarsen to hold it.
    pub max_voxels: usize,
    /// The sampler VIO-02 will use (117.5).
    pub sample: LatticeSample,
    /// The activation VIO-03 will use (117.6).
    pub activation: LatticeActivation,
    /// The value of a voxel that no cell claims (117.5, 117.6).
    pub background: f64,
}

impl Default for LatticeSpec {
    fn default() -> Self {
        Self {
            voxel: None,
            bbox: None,
            max_voxels: DEFAULT_MAX_VOXELS,
            sample: LatticeSample::Cell,
            activation: LatticeActivation::Dense,
            background: 0.0,
        }
    }
}

// ==========================================================================
//  The report (117.7)
// ==========================================================================

/// What one `build` did, key by key as 117.7's table. Built once, printed
/// before the time loop (by the drivers, VIO-05a - `build` prints nothing),
/// and copied into the run summary under `lattice`.
#[derive(Debug, Clone, PartialEq)]
pub struct LatticeReport {
    /// `[n_x, n_y, n_z]` of (117.1).
    pub n: [usize; 3],
    /// `h'`, the voxel edge actually laid.
    pub voxel: f64,
    /// `h`, the voxel edge asked for.
    pub voxel_requested: f64,
    /// `s = h' / h` of (117.4); exactly `1.0` when the budget held.
    pub coarsen: f64,
    /// The lattice size, `n_x n_y n_z`.
    pub n_voxels: usize,
    /// The voxels with an owner, `M[v] != NONE`.
    pub n_fluid: usize,
    /// Of those, the ones given out by pass 2 (117.4). A large count on a
    /// real mesh is a signal, not an error.
    pub n_filled: usize,
    /// `n_fluid / n_voxels`.
    pub fluid_fraction: f64,
    /// `(n_fluid h'^3 - sum_c V_c) / sum_c V_c`, 117.7's coverage error.
    /// `Some` only when the box is the default (`spec.bbox == None`), since
    /// only then does the lattice volume bound the mesh volume.
    pub volume_error: Option<f64>,
    /// (117.7)'s fallback count, `linear` sampling only; `0` here. VIO-02
    /// fills it.
    pub n_linear_fallback: usize,
    /// Wall time of `VoxelMap::build`.
    pub build_s: f64,
    /// The coarsening note, naming `h`, `h'` and `s` (117.3); `None` when
    /// `coarsen == 1.0`.
    pub note: Option<String>,
}

// ==========================================================================
//  The map (117.4)
// ==========================================================================

/// The setup-time map the `.vdb`/`.nvdb`/`.usda` writers gather through
/// (117.4): one `u32` per voxel - a cell index or [`NONE`] - so a write is a
/// gather with no geometry. At 8 M voxels the array is 32 MB.
#[derive(Debug, Clone)]
pub struct VoxelMap {
    /// `[n_x, n_y, n_z]` of (117.1).
    pub n: [usize; 3],
    /// The low corner of the lattice box, world coordinates (117.2).
    pub lo: DVec3,
    /// `h'`, the voxel edge actually laid (117.3).
    pub voxel: f64,
    /// `[n_voxels]`, flat index `i + n_x (j + n_y k)`: a cell index or
    /// [`NONE`] (117.6).
    pub cell: Vec<u32>,
    /// The spec the map was built from.
    pub spec: LatticeSpec,
    /// 117.7's report.
    pub report: LatticeReport,
}

impl VoxelMap {
    /// The lattice size, `n_x n_y n_z`.
    pub fn n_voxels(&self) -> usize {
        self.cell.len()
    }

    /// The flat index `i + n_x (j + n_y k)` of (117.3).
    pub fn index(&self, i: usize, j: usize, k: usize) -> usize {
        i + self.n[0] * (j + self.n[1] * k)
    }

    /// The voxel centre (117.2): per axis `lo_a + h (i + 1/2)` - the one
    /// fixed expression, written the same way inside `build`.
    pub fn centre(&self, i: usize, j: usize, k: usize) -> DVec3 {
        DVec3::new(
            self.lo.x + self.voxel * (i as f64 + 0.5),
            self.lo.y + self.voxel * (j as f64 + 0.5),
            self.lo.z + self.voxel * (k as f64 + 0.5),
        )
    }

    /// The centre of voxel `(0, 0, 0)` - the file transform's `origin`
    /// (117.3).
    pub fn origin(&self) -> DVec3 {
        self.centre(0, 0, 0)
    }

    /// The cell that owns voxel `(i, j, k)`, or `None` (117.6).
    pub fn owner(&self, i: usize, j: usize, k: usize) -> Option<usize> {
        match self.cell[self.index(i, j, k)] {
            NONE => None,
            c => Some(c as usize),
        }
    }
}

// ==========================================================================
//  The lattice dimensions and the budget (117.1, 117.3)
// ==========================================================================

/// (117.1): `n_a = max(1, ceil(L_a / h - 1e-9))`. The epsilon absorbs the
/// rounding of `L_a / h`, so a length the exact rational `L_a / h` divides
/// evenly never gains a voxel from a 1-ulp excess.
pub fn lattice_dims(l: [f64; 3], h: f64) -> [usize; 3] {
    let dim = |len: f64| ((len / h - 1e-9).ceil().max(1.0)) as usize;
    [dim(l[0]), dim(l[1]), dim(l[2])]
}

/// `N(h) = n_x n_y n_z` (117.3), or `None` when the count leaves `usize` -
/// which counts as over budget, never as a wrap.
fn lattice_count(l: [f64; 3], h: f64) -> Option<u128> {
    let mut acc = 1u128;
    for &v in lattice_dims(l, h).iter() {
        acc = acc.checked_mul(v as u128)?;
    }
    Some(acc)
}

/// `N(h) <= max_voxels`.
fn fits(l: [f64; 3], h: f64, max_voxels: usize) -> bool {
    match lattice_count(l, h) {
        Some(n) => n <= max_voxels as u128,
        None => false,
    }
}

/// Axis `a`'s candidate `L_a / m`, as a fitting (117.3) candidate: at or
/// above `h`, with its lattice within `max_voxels`.
fn candidate_fits(l: [f64; 3], a: usize, m: usize, h: f64, max_voxels: usize) -> bool {
    let cand = l[a] / m as f64;
    cand >= h && fits(l, cand, max_voxels)
}

/// The coarsened voxel `(h', s)` of (117.3)-(117.4). When the budget holds
/// at `h` the answer is `(h, 1.0)` exactly. Otherwise `N(h)` is a
/// non-increasing step function of `h` with steps at `h = L_a / m`, and `h'`
/// is the smallest such step at or above `h` that fits `max_voxels` - the
/// minimum of (117.3). There are at most `n_x + n_y + n_z` candidates; on
/// axis `a` the fitting `m` form a prefix (larger `m`, smaller step, bigger
/// lattice), so the axis's smallest fitting candidate is `L_a / m*` at the
/// LARGEST fitting `m*`, found by bisection. Bisection rather than a linear
/// walk, so an absurdly small `h` cannot spin a shared machine.
pub fn coarsen_to_budget(l: [f64; 3], h: f64, max_voxels: usize) -> (f64, f64) {
    if fits(l, h, max_voxels) {
        return (h, 1.0);
    }
    let mut best: Option<f64> = None;
    for a in 0..3 {
        if !candidate_fits(l, a, 1, h, max_voxels) {
            continue;
        }
        // Invariant: m = lo_m fits, m = hi_m does not. The largest fitting
        // m gives this axis's smallest fitting candidate, L_a / lo_m.
        let mut lo_m = 1usize;
        let mut hi_m = lattice_dims(l, h)[a].saturating_add(1);
        while hi_m - lo_m > 1 {
            let mid = lo_m + (hi_m - lo_m) / 2;
            if candidate_fits(l, a, mid, h, max_voxels) {
                lo_m = mid;
            } else {
                hi_m = mid;
            }
        }
        let cand = l[a] / lo_m as f64;
        best = Some(match best {
            Some(b) => b.min(cand),
            None => cand,
        });
    }
    match best {
        // The m = 1 candidate of the longest axis always fits (its lattice is
        // (1, 1, 1) whenever some axis needs 2 or more voxels at h), so the
        // `None` arm is unreachable for a positive finite h; kept total so
        // no caller can panic.
        Some(h_prime) => (h_prime, h_prime / h),
        None => (h, 1.0),
    }
}

// ==========================================================================
//  Refusals - 117.12's first four rows, by name, before any work
// ==========================================================================

fn check_spec(spec: &LatticeSpec) -> Result<()> {
    if let Some(b) = spec.bbox {
        for (a, &e) in b.iter().enumerate() {
            if !e.is_finite() {
                return Err(Error::Refused(format!(
                    "output.visualisation.lattice.box: entry {a} is {e}, not finite - there is no lattice without a finite box"
                )));
            }
        }
        for a in 0..3 {
            if b[a] >= b[a + 3] {
                return Err(Error::Refused(format!(
                    "output.visualisation.lattice.box: lo[{a}] = {} >= hi[{a}] = {} - the lattice would be empty",
                    b[a], b[a + 3]
                )));
            }
        }
    }
    if let Some(v) = spec.voxel {
        if v <= 0.0 || !v.is_finite() {
            return Err(Error::Refused(format!(
                "output.visualisation.lattice.voxel: {v} - a non-positive or non-finite voxel edge means there is no lattice"
            )));
        }
    }
    if !spec.background.is_finite() {
        return Err(Error::Refused(format!(
            "output.visualisation.lattice.background: {} is not finite - there is no lattice to carry it",
            spec.background
        )));
    }
    if spec.max_voxels == 0 {
        return Err(Error::Refused(
            "output.visualisation.lattice.maxVoxels: 0 - there is no lattice under a zero-voxel budget".to_string(),
        ));
    }
    Ok(())
}

fn check_mesh(raw: &PolyMeshRaw, hm: &HostMesh) -> Result<()> {
    if raw.faces.len() != hm.n_internal_faces + hm.n_boundary_faces {
        return Err(Error::Mesh(format!(
            "output.visualisation.lattice: the raw mesh has {} faces but the HostMesh counts {} internal + {} boundary",
            raw.faces.len(),
            hm.n_internal_faces,
            hm.n_boundary_faces
        )));
    }
    if hm.cf_offset.len() != hm.n_cells + 1 {
        return Err(Error::Mesh(format!(
            "output.visualisation.lattice: the cell-face CSR is not built (cf_offset has {} entries, want {})",
            hm.cf_offset.len(),
            hm.n_cells + 1
        )));
    }
    if hm.bcf_offset.len() != hm.n_cells + 1 {
        return Err(Error::Mesh(format!(
            "output.visualisation.lattice: the cell-boundary-face CSR is not built (bcf_offset has {} entries, want {})",
            hm.bcf_offset.len(),
            hm.n_cells + 1
        )));
    }
    if hm.n_cells == 0 {
        return Err(Error::Mesh(
            "output.visualisation.lattice: the mesh has no cells - there is nothing to rasterise".to_string(),
        ));
    }
    if hm.n_cells >= u32::MAX as usize {
        return Err(Error::Mesh(format!(
            "output.visualisation.lattice: {} cells do not fit in the map's u32 cell ids",
            hm.n_cells
        )));
    }
    Ok(())
}

// ==========================================================================
//  Geometry helpers
// ==========================================================================

/// Widen a closed box by one point.
fn widen(p: DVec3, lo: &mut DVec3, hi: &mut DVec3) {
    lo.x = lo.x.min(p.x);
    hi.x = hi.x.max(p.x);
    lo.y = lo.y.min(p.y);
    hi.y = hi.y.max(p.y);
    lo.z = lo.z.min(p.z);
    hi.z = hi.z.max(p.z);
}

/// The closed bounding box of the points of cell `c`'s faces, internal and
/// boundary (117.4 pass 1 step 1), straight from `raw.faces` / `raw.points`.
fn cell_box(raw: &PolyMeshRaw, hm: &HostMesh, c: usize) -> (DVec3, DVec3) {
    let mut lo = DVec3::new(f64::INFINITY, f64::INFINITY, f64::INFINITY);
    let mut hi = DVec3::new(f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY);
    for k in hm.cf_offset[c] as usize..hm.cf_offset[c + 1] as usize {
        for &p in &raw.faces[hm.cf_face[k] as usize] {
            widen(raw.points[p as usize], &mut lo, &mut hi);
        }
    }
    for k in hm.bcf_offset[c] as usize..hm.bcf_offset[c + 1] as usize {
        for &p in &raw.faces[hm.n_internal_faces + hm.bcf_face[k] as usize] {
            widen(raw.points[p as usize], &mut lo, &mut hi);
        }
    }
    (lo, hi)
}

/// The voxel index range whose centres can lie in `[blo, bhi]` on one axis
/// (117.4 pass 1 step 2), widened by one voxel on each side: the inside test
/// decides, never the range.
fn axis_range(blo: f64, bhi: f64, lo: f64, h: f64, n: usize) -> (usize, usize) {
    let last = n as isize - 1;
    let a = (((blo - lo) / h - 0.5).ceil() as isize).clamp(0, last) - 1;
    let b = (((bhi - lo) / h - 0.5).floor() as isize).clamp(0, last) + 1;
    (a.max(0) as usize, b.min(last) as usize)
}

/// (117.5) in f64, exactly `parcels::locate_cell`'s test through `geom64()`
/// (117.2), in this operand order: for an internal face slot `k` of `c`
/// outside when `(x - cf[f]).dot(sf[f] * sgn) > 0` with `sgn = +1` on the
/// owner and `-1` on the neighbour; for a boundary face `bf`, outside when
/// `(x - b_cf[bf]).dot(b_sf[bf]) > 0`. A tie on a face therefore resolves
/// the same way as in `locate_cell`.
fn inside(x: DVec3, hm: &HostMesh, g: &Geom64<'_>, c: usize) -> bool {
    for k in hm.cf_offset[c] as usize..hm.cf_offset[c + 1] as usize {
        let f = hm.cf_face[k] as usize;
        let sgn: f64 = if hm.cf_own[k] != 0 { 1.0 } else { -1.0 };
        if (x - g.cf[f]).dot(g.sf[f] * sgn) > 0.0 {
            return false;
        }
    }
    boundary_faces_hold(x, hm, g, c)
}

/// (117.5) over `c`'s BOUNDARY faces only - pass 2's extra condition (117.4):
/// a voxel inside a solid body, which also lies in the boxes of the wall
/// cells around it, stays solid.
fn boundary_faces_hold(x: DVec3, hm: &HostMesh, g: &Geom64<'_>, c: usize) -> bool {
    for k in hm.bcf_offset[c] as usize..hm.bcf_offset[c + 1] as usize {
        let bf = hm.bcf_face[k] as usize;
        if (x - g.b_cf[bf]).dot(g.b_sf[bf]) > 0.0 {
            return false;
        }
    }
    true
}

// ==========================================================================
//  build
// ==========================================================================

impl VoxelMap {
    /// Build the map: the lattice of 117.3 over `raw`/`hm`, rasterised by
    /// 117.4's two passes into (117.6), reported per 117.7.
    ///
    /// Runs once, at setup, before the time loop. The map is built in f64 in
    /// BOTH builds - the boxes from the polyMesh points, which are f64 from
    /// F32-03 on, and the half-space tests through `geom64()` - so it is the
    /// same, bit for bit, in either build (117.2 item 1). The lattice box and
    /// every point are in world coordinates: this tree has no local origin
    /// yet, and when F32-10's shift arrives the map adds it back before it is
    /// built, so the written transform stays the world transform (117.2
    /// item 4). Deterministic and single-threaded; no float is hashed.
    pub fn build(raw: &PolyMeshRaw, hm: &HostMesh, spec: &LatticeSpec) -> Result<VoxelMap> {
        let t0 = Instant::now();
        check_spec(spec)?;
        check_mesh(raw, hm)?;
        let g = hm.geom64();

        // ---- the lattice box (117.3) ------------------------------------
        let (lo, hi) = match spec.bbox {
            Some(b) => (DVec3::new(b[0], b[1], b[2]), DVec3::new(b[3], b[4], b[5])),
            None => {
                let mut blo = DVec3::new(f64::INFINITY, f64::INFINITY, f64::INFINITY);
                let mut bhi = DVec3::new(f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY);
                for p in &raw.points {
                    widen(*p, &mut blo, &mut bhi);
                }
                (blo, bhi)
            }
        };
        let l = [hi.x - lo.x, hi.y - lo.y, hi.z - lo.z];

        // ---- the requested voxel (117.3) --------------------------------
        let h_req = match spec.voxel {
            Some(v) => v,
            None => {
                let v_min = g.v.iter().copied().fold(f64::INFINITY, f64::min);
                if !(v_min > 0.0 && v_min.is_finite()) {
                    return Err(Error::Mesh(format!(
                        "output.visualisation.lattice.voxel: the default voxel is (min_c V_c)^(1/3), but the smallest cell volume is {v_min}"
                    )));
                }
                v_min.cbrt()
            }
        };

        // ---- the budget (117.3, 117.4) -----------------------------------
        let (h, coarsen) = coarsen_to_budget(l, h_req, spec.max_voxels);
        if spec.voxel.is_some() && coarsen > MAX_EXPLICIT_COARSEN {
            let n_req = match lattice_count(l, h_req) {
                Some(n) => format!("{n}"),
                None => "more than usize can hold".to_string(),
            };
            return Err(Error::Refused(format!(
                "output.visualisation.lattice.voxel: the budget cannot hold the lattice asked for: voxel {h_req} m needs {n_req} voxels, over maxVoxels {}; the smallest lattice that fits has h' = {h} m, a coarsening factor s = {coarsen} over the {} limit",
                spec.max_voxels, MAX_EXPLICIT_COARSEN
            )));
        }

        // ---- the lattice (117.1) ----------------------------------------
        let n = lattice_dims(l, h);
        let n_voxels = n[0] * n[1] * n[2];
        let mut cell = vec![NONE; n_voxels];

        // The one fixed expression for a voxel centre (117.2), local to the
        // box and edge this map was built with.
        let centre_of = |i: usize, j: usize, k: usize| {
            DVec3::new(
                lo.x + h * (i as f64 + 0.5),
                lo.y + h * (j as f64 + 0.5),
                lo.z + h * (k as f64 + 0.5),
            )
        };

        // ---- pass 1: rasterisation (117.4) -------------------------------
        // Cells in ascending index; the first claim wins, so a centre on a
        // shared face, edge or corner goes to the lowest cell index that
        // contains it. An assigned voxel is skipped without a test and is
        // never reassigned.
        let boxes: Vec<(DVec3, DVec3)> = (0..hm.n_cells).map(|c| cell_box(raw, hm, c)).collect();
        for (c, &(b_lo, b_hi)) in boxes.iter().enumerate() {
            let (i0, i1) = axis_range(b_lo.x, b_hi.x, lo.x, h, n[0]);
            let (j0, j1) = axis_range(b_lo.y, b_hi.y, lo.y, h, n[1]);
            let (k0, k1) = axis_range(b_lo.z, b_hi.z, lo.z, h, n[2]);
            for k in k0..=k1 {
                for j in j0..=j1 {
                    let row = j * n[0] + k * n[0] * n[1];
                    for i in i0..=i1 {
                        let v = i + row;
                        if cell[v] != NONE {
                            continue;
                        }
                        if inside(centre_of(i, j, k), hm, &g, c) {
                            cell[v] = c as u32;
                        }
                    }
                }
            }
        }

        // ---- pass 2: gaps (117.4) ----------------------------------------
        // A warped face or a non-convex cell can leave an interior voxel no
        // cell claimed. K_v = { c : x_v in B_c }, collected cell by cell in
        // ascending order so each list is ascending, by comparing x_v with
        // the closed box directly. EVERY voxel is decided - nearest centre,
        // strict <, so ties go to the lowest index; then the boundary-face
        // condition - before ANY is assigned, so one pass-2 grant cannot
        // change another voxel's answer.
        let mut k_v: Vec<Vec<u32>> = vec![Vec::new(); n_voxels];
        for (c, &(b_lo, b_hi)) in boxes.iter().enumerate() {
            let (i0, i1) = axis_range(b_lo.x, b_hi.x, lo.x, h, n[0]);
            let (j0, j1) = axis_range(b_lo.y, b_hi.y, lo.y, h, n[1]);
            let (k0, k1) = axis_range(b_lo.z, b_hi.z, lo.z, h, n[2]);
            for k in k0..=k1 {
                for j in j0..=j1 {
                    for i in i0..=i1 {
                        let v = i + j * n[0] + k * n[0] * n[1];
                        if cell[v] != NONE {
                            continue;
                        }
                        let x = centre_of(i, j, k);
                        if b_lo.x <= x.x
                            && x.x <= b_hi.x
                            && b_lo.y <= x.y
                            && x.y <= b_hi.y
                            && b_lo.z <= x.z
                            && x.z <= b_hi.z
                        {
                            k_v[v].push(c as u32);
                        }
                    }
                }
            }
        }
        let mut fills: Vec<(usize, u32)> = Vec::new();
        for (v, ks) in k_v.iter().enumerate() {
            if ks.is_empty() {
                continue;
            }
            let x = centre_of(v % n[0], v / n[0] % n[1], v / (n[0] * n[1]));
            let mut best = ks[0] as usize;
            let d0 = g.c[best] - x;
            let mut best_d = d0.dot(d0);
            for &kc in &ks[1..] {
                let c = kc as usize;
                let d = g.c[c] - x;
                let d = d.dot(d);
                if d < best_d {
                    best_d = d;
                    best = c;
                }
            }
            if boundary_faces_hold(x, hm, &g, best) {
                fills.push((v, best as u32));
            }
        }
        for (v, c) in &fills {
            cell[*v] = *c;
        }
        let n_filled = fills.len();
        let n_fluid = cell.iter().filter(|&&c| c != NONE).count();

        // ---- the report (117.7) ------------------------------------------
        let sum_v: f64 = g.v.iter().sum();
        let volume_error = if spec.bbox.is_none() {
            Some((n_fluid as f64 * (h * h * h) - sum_v) / sum_v)
        } else {
            None
        };
        let note = if coarsen != 1.0 {
            let n_req = match lattice_count(l, h_req) {
                Some(v) => format!("{v}"),
                None => "more than usize can hold".to_string(),
            };
            Some(format!(
                "lattice: voxel {h_req} m needs {n_req} voxels, over maxVoxels {}; coarsened to {h} m ({} x {} x {}), coarsening factor {coarsen}",
                spec.max_voxels, n[0], n[1], n[2]
            ))
        } else {
            None
        };
        let report = LatticeReport {
            n,
            voxel: h,
            voxel_requested: h_req,
            coarsen,
            n_voxels,
            n_fluid,
            n_filled,
            fluid_fraction: n_fluid as f64 / n_voxels as f64,
            volume_error,
            n_linear_fallback: 0,
            build_s: t0.elapsed().as_secs_f64(),
            note,
        };
        Ok(VoxelMap {
            n,
            lo,
            voxel: h,
            cell,
            spec: spec.clone(),
            report,
        })
    }
}
