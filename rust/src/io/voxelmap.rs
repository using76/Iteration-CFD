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
//! key. Sampling (117.5) is implemented here: [`VoxelMap::sample`] writes
//! `s_v` at every voxel - the cell value, or the least-squares gradient of
//! (117.7) clipped to the point neighbours' range by (117.8), with
//! [`LinearStencil`] holding the setup-time data. VIO-03 writes.
//!
//! Written from SPEC-LIT section 117 (117.2-117.5, 117.7).
//! Provenance: ORIGINAL.
//! No GPL-licensed source was consulted.

use std::time::Instant;

use crate::error::{Error, Result};
use crate::io::polymesh::PolyMeshRaw;
use crate::mesh::{Geom64, HostMesh};
use crate::types::DVec3;
use crate::Scalar;

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

/// Which sampler [`VoxelMap::sample`] applies at each owned voxel (117.5).
/// `build` reads it to decide whether to build a [`LinearStencil`].
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
    /// (117.7)'s fallback count, `linear` sampling only - the stencil's
    /// [`LinearStencil::n_fallback`], filled by `build`; `0` for `cell`.
    pub n_linear_fallback: usize,
    /// Wall time of `VoxelMap::build`.
    pub build_s: f64,
    /// The coarsening note, naming `h`, `h'` and `s` (117.3); `None` when
    /// `coarsen == 1.0`.
    pub note: Option<String>,
}

// ==========================================================================
//  The linear stencil (117.5)
// ==========================================================================

/// Setup-time data of `linear` sampling (117.7)-(117.8). Built once; holds
/// no field values: the face and point neighbours of every cell, the cell
/// centres, and (117.7)'s fallback classification - a property of the
/// geometry alone, so it is decided here and not per field.
#[derive(Debug, Clone)]
pub struct LinearStencil {
    n_cells: usize,
    /// `C_c` of (117.7)-(117.8), `hm.geom64().c` - f64 in both builds.
    centres: Vec<DVec3>,
    /// `N_f(c)` as a CSR of cell ids, each list ascending, deduped, no `c`.
    f_off: Vec<usize>,
    f_nbr: Vec<u32>,
    /// `N_p(c)` as a CSR of cell ids, each list ascending, deduped, no `c`.
    p_off: Vec<usize>,
    p_nbr: Vec<u32>,
    /// (117.7)'s fallback test per cell: `det G <= 1e-12 (tr G / 3)^3`.
    fallback: Vec<bool>,
    n_fallback: usize,
}

/// `det G` of the 3 x 3 symmetric normal matrix
/// `[[a, b, c], [b, d, e], [c, e, f]]`.
fn det3(a: f64, b: f64, c: f64, d: f64, e: f64, f: f64) -> f64 {
    a * (d * f - e * e) - b * (b * f - c * e) + c * (b * e - d * c)
}

/// A `clamp` that cannot panic: `f64::clamp` asserts when a bound is NaN,
/// and a non-finite field value must reach the output, never abort (117.5).
/// A NaN value fails both tests and passes through.
fn clip(s: f64, lo: f64, hi: f64) -> f64 {
    if s < lo {
        lo
    } else if s > hi {
        hi
    } else {
        s
    }
}

/// Sort, dedup and drop `c` from a neighbour list in place; the kept length.
fn sorted_neighbours(s: &mut [u32], c: u32) -> usize {
    s.sort_unstable();
    // Sorted, so dedup compacts in place.
    let mut w = 0usize;
    for k in 0..s.len() {
        if w == 0 || s[k] != s[w - 1] {
            s[w] = s[k];
            w += 1;
        }
    }
    let k = s[..w].partition_point(|&x| x < c);
    if k < w && s[k] == c {
        s.copy_within(k + 1..w, k);
        w - 1
    } else {
        w
    }
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
    /// The mesh's cell count, `hm.n_cells`: the field length a `sample`
    /// call takes (117.5).
    pub n_cells: usize,
    /// The linear stencil of (117.7)-(117.8) when `spec.sample` is
    /// [`LatticeSample::Linear`], else `None` (117.5).
    pub linear: Option<LinearStencil>,
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

    /// `s_v` for every voxel, `[n_voxels]`, flat index as `self.cell`
    /// (117.5). Dispatches on `self.spec.sample`: `cell` copies `phi_c`;
    /// `linear` writes (117.8) - the cell value plus (117.7)'s gradient,
    /// clipped to the point neighbours' range; a `NONE` voxel holds
    /// `spec.background`. All f64 in both builds; the writer rounds once
    /// (117.2 item 2). One call does no geometric search.
    pub fn sample_f64(&self, phi: &[f64]) -> Result<Vec<f64>> {
        if phi.len() != self.n_cells {
            return Err(Error::Mesh(format!(
                "output.visualisation.lattice.sample: the field has {} values but the map was built on {} cells",
                phi.len(),
                self.n_cells
            )));
        }
        let mut out = vec![self.spec.background; self.cell.len()];
        match self.spec.sample {
            LatticeSample::Cell => {
                for (v, &cv) in self.cell.iter().enumerate() {
                    if cv != NONE {
                        out[v] = phi[cv as usize];
                    }
                }
            }
            LatticeSample::Linear => {
                let st = match &self.linear {
                    Some(st) => st,
                    None => {
                        return Err(Error::Mesh(
                            "output.visualisation.lattice.sample: sample is linear but this map carries no linear stencil - rebuild the map with spec.sample = LatticeSample::Linear"
                                .to_string(),
                        ));
                    }
                };
                // (117.7)'s g_c and (117.8)'s (m_c, M_c) once per CELL, not
                // once per voxel: neither depends on v, and a coarse cell
                // can own hundreds of voxels. One call therefore costs
                // O(n_cells x neighbours + n_voxels), and the voxel loop
                // only reads the two tables - bit-identical to a per-voxel
                // evaluation, since both are the same function of (phi, c).
                let mut grad = Vec::with_capacity(self.n_cells);
                let mut bnd = Vec::with_capacity(self.n_cells);
                for c in 0..self.n_cells {
                    grad.push(st.gradient(phi, c));
                    bnd.push(st.bounds(phi, c));
                }
                let centres = &st.centres;
                for (v, &cv) in self.cell.iter().enumerate() {
                    if cv == NONE {
                        continue;
                    }
                    let c = cv as usize;
                    let x = self.centre(
                        v % self.n[0],
                        v / self.n[0] % self.n[1],
                        v / (self.n[0] * self.n[1]),
                    );
                    let (lo, hi) = bnd[c];
                    out[v] = clip(phi[c] + grad[c].dot(x - centres[c]), lo, hi);
                }
            }
        }
        Ok(out)
    }

    /// The active mask of 117.6, as the Isaac density reads it (117.9,
    /// 117.10): `true` where a cell owns the voxel, `false` where it holds
    /// [`NONE`].
    pub fn fluid_mask(&self) -> Vec<bool> {
        self.cell.iter().map(|&c| c != NONE).collect()
    }

    /// The same for the solver's own `Scalar` cell values, widened exactly
    /// to f64 first (`as f64`, no value rounded back - 117.2 item 2).
    pub fn sample(&self, phi: &[Scalar]) -> Result<Vec<f64>> {
        if phi.len() != self.n_cells {
            return Err(Error::Mesh(format!(
                "output.visualisation.lattice.sample: the field has {} values but the map was built on {} cells",
                phi.len(),
                self.n_cells
            )));
        }
        let wide: Vec<f64> = phi.iter().map(|&s| s as f64).collect();
        self.sample_f64(&wide)
    }
}

// ==========================================================================
//  The Isaac density (SPEC-LIT 117.9, VIO-09)
// ==========================================================================

/// `K_m`, the specific extinction coefficient of flame-generated smoke,
/// m^2/kg (117.14): 8.7 m^2/g from Mulholland & Croarkin (2000), *Fire
/// Mater.* 24(5) 227-230. `isaac_export.py`'s `K = 8700 . smoke_density`.
pub const SMOKE_SPECIFIC_EXTINCTION: f64 = 8700.0;

/// (117.10)'s `f^_v`: `clamp((f_v - f_lo) / (f_hi - f_lo), 0, 1)`, and 0
/// where `M[v] = NONE` (a voxel nothing owns renders nothing). A non-finite
/// `f_v` fails both clamp tests and gives 0, never a NaN in the grid.
/// `range[1] <= range[0]` (a constant field, or a bad explicit range) gives
/// all zeros - the caller sees `D = 0` and writes an all-zero density grid.
pub fn normalise_for_optical_depth(f: &[f64], fluid: Option<&[bool]>, range: [f64; 2]) -> Vec<f64> {
    let [lo, hi] = range;
    let span = hi - lo;
    f.iter()
        .enumerate()
        .map(|(v, &x)| {
            if span <= 0.0 || !span.is_finite() {
                return 0.0;
            }
            if fluid.is_some_and(|m| !m[v]) {
                return 0.0;
            }
            if !x.is_finite() {
                return 0.0; // 117.10: a non-finite f_v gives 0
            }
            if x < lo {
                return 0.0;
            }
            if x > hi {
                return 1.0;
            }
            (x - lo) / span
        })
        .collect()
}

/// The default `range` of 117.9: `(min, max)` of `f` over the FLUID voxels
/// with a finite value. `fluid: None` = every voxel is fluid. `None` when
/// no voxel qualifies - the caller has no range to normalise with, and its
/// write is the all-zero one.
pub fn fluid_range(f: &[f64], fluid: Option<&[bool]>) -> Option<[f64; 2]> {
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for (v, &x) in f.iter().enumerate() {
        if fluid.is_some_and(|m| !m[v]) || !x.is_finite() {
            continue;
        }
        lo = lo.min(x);
        hi = hi.max(x);
    }
    (lo <= hi).then_some([lo, hi])
}

/// (117.11): the largest optical depth of one voxel line parallel to an
/// axis, `max over a and l of h_a sum_{v in l} f^_v`. `n` is `[n_x, n_y,
/// n_z]`, the flat index `i + n0 (j + n1 k)`, and `h` the spacing per axis -
/// one cubic `h` on a lattice, the axis's own `h_a` on a Cartesian box,
/// where the cells' spacing differs per axis.
pub fn max_column_optical_depth(n: [usize; 3], h: [f64; 3], s: &[f64]) -> f64 {
    let at = |i: usize, j: usize, k: usize| i + n[0] * (j + n[1] * k);
    let mut max = 0.0f64;
    // x lines: n_y n_z of them, n_x voxels each.
    for k in 0..n[2] {
        for j in 0..n[1] {
            let mut sum = 0.0f64;
            for i in 0..n[0] {
                sum += s[at(i, j, k)];
            }
            max = max.max(h[0] * sum);
        }
    }
    // y lines.
    for k in 0..n[2] {
        for i in 0..n[0] {
            let mut sum = 0.0f64;
            for j in 0..n[1] {
                sum += s[at(i, j, k)];
            }
            max = max.max(h[1] * sum);
        }
    }
    // z lines.
    for j in 0..n[1] {
        for i in 0..n[0] {
            let mut sum = 0.0f64;
            for k in 0..n[2] {
                sum += s[at(i, j, k)];
            }
            max = max.max(h[2] * sum);
        }
    }
    max
}

/// (117.12): `sigma_v = tau f^_v / D`, in that order, so the densest
/// column's optical depth is exactly `tau` before rounding. `d <= 0` or
/// non-finite (the all-zero write, or the write before `D` is held) gives
/// all zeros - the extinction Beer-Lambert integrates is 0 everywhere.
pub fn optical_depth_sigma(f_hat: &[f64], tau: f64, d: f64) -> Vec<f64> {
    if !(d > 0.0) || !d.is_finite() {
        return vec![0.0; f_hat.len()];
    }
    f_hat.iter().map(|&x| tau * x / d).collect()
}

/// (117.11) + (117.12) together: `(sigma, D)`, with `D` from
/// [`max_column_optical_depth`]. The 117.9 gate "the max column optical
/// depth equals tau within 1 %" holds by construction here, up to the
/// rounding of `sigma` to the voxel precision.
pub fn optical_depth_target(n: [usize; 3], h: [f64; 3], f_hat: &[f64], tau: f64) -> (Vec<f64>, f64) {
    let d = max_column_optical_depth(n, h, f_hat);
    (optical_depth_sigma(f_hat, tau, d), d)
}

/// (117.14): `sigma_v = K_m rho_s,v`, in that order, for the fluid voxels;
/// a negative `rho_s` (an undershoot of a transported mass fraction) is
/// written as 0 and COUNTED - the second return value; [`NONE`] voxels get
/// 0. `fluid: None` = every voxel is fluid.
pub fn physical_extinction(rho_s: &[f64], fluid: Option<&[bool]>, km: f64) -> (Vec<f64>, usize) {
    let mut negatives = 0usize;
    let sigma = rho_s
        .iter()
        .enumerate()
        .map(|(v, &r)| {
            if fluid.is_some_and(|m| !m[v]) {
                return 0.0;
            }
            if r < 0.0 {
                negatives += 1;
                return 0.0;
            }
            km * r
        })
        .collect();
    (sigma, negatives)
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

/// 117.12's spec-only refusals: the ones that need no mesh (VIO-04 raises
/// them when the case is read).
pub fn check_spec(spec: &LatticeSpec) -> Result<()> {
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

/// The lattice of (117.1)-(117.4), decided without rasterising a voxel:
/// the box, the voxel asked for, and the voxel the budget lays.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LatticeShape {
    /// The low corner of the lattice box (117.2).
    pub lo: DVec3,
    /// `L_a = hi_a - lo_a` (117.1).
    pub l: [f64; 3],
    /// `h`, the voxel edge asked for (117.3).
    pub voxel_requested: f64,
    /// `h'` of (117.3).
    pub voxel: f64,
    /// `s = h' / h` of (117.4).
    pub coarsen: f64,
}

impl LatticeShape {
    /// `[n_x, n_y, n_z]` of (117.1) at `h'`.
    pub fn n(&self) -> [usize; 3] {
        lattice_dims(self.l, self.voxel)
    }
}

/// (117.3)-(117.4)'s budget, with 117.12's refusal of an explicit voxel the
/// budget would coarsen by `s > 16`. `explicit` is "the case named `voxel`";
/// the default voxel is coarsened without that limit. Returns `(h', s)`.
pub fn check_budget(l: [f64; 3], h_req: f64, explicit: bool, max_voxels: usize) -> Result<(f64, f64)> {
    let (h, coarsen) = coarsen_to_budget(l, h_req, max_voxels);
    if explicit && coarsen > MAX_EXPLICIT_COARSEN {
        let n_req = match lattice_count(l, h_req) {
            Some(n) => format!("{n}"),
            None => "more than usize can hold".to_string(),
        };
        return Err(Error::Refused(format!(
            "output.visualisation.lattice.voxel: the budget cannot hold the lattice asked for: voxel {h_req} m needs {n_req} voxels, over maxVoxels {max_voxels}; the smallest lattice that fits has h' = {h} m, a coarsening factor s = {coarsen} over the {} limit",
            MAX_EXPLICIT_COARSEN
        )));
    }
    Ok((h, coarsen))
}

/// Every refusal of 117.12 that [`VoxelMap::build`] raises, and nothing it
/// computes after them: the spec, the mesh, the box, the requested voxel
/// and the budget. Linear in the points and cells, no voxel visited - so a
/// driver can ask "can a lattice be laid here?" before the time loop
/// (VIO-04) without building the map. `build` calls it first, so the two
/// can never disagree.
pub fn lattice_shape(raw: &PolyMeshRaw, hm: &HostMesh, spec: &LatticeSpec) -> Result<LatticeShape> {
    check_spec(spec)?;
    check_mesh(raw, hm)?;
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
            let v_min = hm.geom64().v.iter().copied().fold(f64::INFINITY, f64::min);
            if !(v_min > 0.0 && v_min.is_finite()) {
                return Err(Error::Mesh(format!(
                    "output.visualisation.lattice.voxel: the default voxel is (min_c V_c)^(1/3), but the smallest cell volume is {v_min}"
                )));
            }
            v_min.cbrt()
        }
    };

    // ---- the budget (117.3, 117.4) -----------------------------------
    let (h, coarsen) = check_budget(l, h_req, spec.voxel.is_some(), spec.max_voxels)?;
    Ok(LatticeShape { lo, l, voxel_requested: h_req, voxel: h, coarsen })
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

/// `N_f(c)` for every cell: the cells across the INTERNAL faces (`f <
/// n_internal_faces`), a set - a cell that shares two faces with `c` counts
/// once - ascending, never `c`. Boundary faces add nothing.
fn face_neighbours(raw: &PolyMeshRaw, hm: &HostMesh) -> (Vec<usize>, Vec<u32>) {
    let mut off = vec![0usize; hm.n_cells + 1];
    for f in 0..hm.n_internal_faces {
        off[raw.owner[f] as usize + 1] += 1;
        off[raw.neighbour[f] as usize + 1] += 1;
    }
    for c in 0..hm.n_cells {
        off[c + 1] += off[c];
    }
    let mut nbrs = vec![0u32; off[hm.n_cells]];
    let mut fill = off.clone();
    for f in 0..hm.n_internal_faces {
        let (o, n) = (raw.owner[f] as usize, raw.neighbour[f] as usize);
        nbrs[fill[o]] = n as u32;
        fill[o] += 1;
        nbrs[fill[n]] = o as u32;
        fill[n] += 1;
    }
    // A set: sort, drop duplicates and any self-entry, compacting in place -
    // the kept lists only shrink.
    let mut done = vec![0usize; hm.n_cells + 1];
    let mut w = 0usize;
    for c in 0..hm.n_cells {
        let k = sorted_neighbours(&mut nbrs[off[c]..fill[c]], c as u32);
        nbrs.copy_within(off[c]..off[c] + k, w);
        w += k;
        done[c + 1] = w;
    }
    nbrs.truncate(w);
    (done, nbrs)
}

/// The points of every cell - all its faces' points, internal and boundary,
/// straight from `raw.faces` - as a CSR, each list sorted and deduped.
fn cell_points(raw: &PolyMeshRaw, hm: &HostMesh) -> (Vec<usize>, Vec<u32>) {
    let mut off = vec![0usize; hm.n_cells + 1];
    for c in 0..hm.n_cells {
        let mut n = 0usize;
        for k in hm.cf_offset[c] as usize..hm.cf_offset[c + 1] as usize {
            n += raw.faces[hm.cf_face[k] as usize].len();
        }
        for k in hm.bcf_offset[c] as usize..hm.bcf_offset[c + 1] as usize {
            n += raw.faces[hm.n_internal_faces + hm.bcf_face[k] as usize].len();
        }
        off[c + 1] = off[c] + n;
    }
    let mut pts = vec![0u32; off[hm.n_cells]];
    for c in 0..hm.n_cells {
        let mut w = off[c];
        for k in hm.cf_offset[c] as usize..hm.cf_offset[c + 1] as usize {
            for &p in &raw.faces[hm.cf_face[k] as usize] {
                pts[w] = p as u32;
                w += 1;
            }
        }
        for k in hm.bcf_offset[c] as usize..hm.bcf_offset[c + 1] as usize {
            for &p in &raw.faces[hm.n_internal_faces + hm.bcf_face[k] as usize] {
                pts[w] = p as u32;
                w += 1;
            }
        }
        pts[off[c]..w].sort_unstable();
        // Duplicates past `dedup`-style compaction are not needed: a point
        // met twice only re-visits the same point's cells, which the stamp
        // in `point_neighbours` dedups anyway.
    }
    (off, pts)
}

/// The cells of every point, inverted from [`cell_points`]: a CSR over
/// `raw.points`, each list ascending (the cells are visited in order).
fn point_cells(
    cp_off: &[usize],
    cp: &[u32],
    n_points: usize,
    n_cells: usize,
) -> (Vec<usize>, Vec<u32>) {
    let mut off = vec![0usize; n_points + 1];
    for c in 0..n_cells {
        for &p in &cp[cp_off[c]..cp_off[c + 1]] {
            off[p as usize + 1] += 1;
        }
    }
    for i in 0..n_points {
        off[i + 1] += off[i];
    }
    let mut cells = vec![0u32; off[n_points]];
    let mut fill = off.clone();
    for c in 0..n_cells {
        for &p in &cp[cp_off[c]..cp_off[c + 1]] {
            cells[fill[p as usize]] = c as u32;
            fill[p as usize] += 1;
        }
    }
    (off, cells)
}

/// `N_p(c)` for every cell: the cells that share at least one point with
/// `c` - the union over the cell's own points - ascending, deduped, never
/// `c`. `N_f(c)` is a subset of it.
fn point_neighbours(raw: &PolyMeshRaw, hm: &HostMesh) -> (Vec<usize>, Vec<u32>) {
    let (cp_off, cp) = cell_points(raw, hm);
    let (pc_off, pc) = point_cells(&cp_off, &cp, raw.points.len(), hm.n_cells);
    let total: usize = (0..hm.n_cells)
        .map(|c| {
            cp[cp_off[c]..cp_off[c + 1]]
                .iter()
                .map(|&p| pc_off[p as usize + 1] - pc_off[p as usize])
                .sum::<usize>()
        })
        .sum();
    let mut nbrs = vec![0u32; total];
    let mut off = vec![0usize; hm.n_cells + 1];
    let mut stamp = vec![usize::MAX; hm.n_cells];
    let mut w = 0usize;
    for c in 0..hm.n_cells {
        let s0 = w;
        for &p in &cp[cp_off[c]..cp_off[c + 1]] {
            for &n in &pc[pc_off[p as usize]..pc_off[p as usize + 1]] {
                let n = n as usize;
                if n != c && stamp[n] != c {
                    stamp[n] = c;
                    nbrs[w] = n as u32;
                    w += 1;
                }
            }
        }
        nbrs[s0..w].sort_unstable();
        off[c + 1] = w;
    }
    (off, nbrs)
}

impl LinearStencil {
    /// The stencils of (117.7)-(117.8) for every cell of `hm`: `N_f(c)` from
    /// the internal faces, `N_p(c)` from the points the cells share through
    /// `raw.faces`, the centres of `geom64()`, and (117.7)'s fallback test,
    /// which reads the geometry only. Everything is f64 in both builds
    /// (117.2 item 2). Cost `O(cells x neighbours)`.
    pub fn build(raw: &PolyMeshRaw, hm: &HostMesh) -> Result<LinearStencil> {
        let n_cells = hm.n_cells;
        let centres = hm.geom64().c.to_vec();
        let (f_off, f_nbr) = face_neighbours(raw, hm);
        let (p_off, p_nbr) = point_neighbours(raw, hm);
        // (117.7)'s fallback test: the offsets `d_n = C_n - C_c` of the face
        // neighbours must span three dimensions, `G = sum d_n d_n^T`; a cell
        // with no face neighbour has `G = 0` and falls back. Geometry only,
        // so it is decided once, here, and `gradient` inherits it.
        let mut fallback = vec![false; n_cells];
        let mut n_fallback = 0usize;
        for c in 0..n_cells {
            let (mut g00, mut g01, mut g02, mut g11, mut g12, mut g22) =
                (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
            let cc = centres[c];
            for &ni in &f_nbr[f_off[c]..f_off[c + 1]] {
                let d = centres[ni as usize] - cc;
                g00 += d.x * d.x;
                g01 += d.x * d.y;
                g02 += d.x * d.z;
                g11 += d.y * d.y;
                g12 += d.y * d.z;
                g22 += d.z * d.z;
            }
            let t = (g00 + g11 + g22) / 3.0;
            if det3(g00, g01, g02, g11, g12, g22) <= 1e-12 * t * t * t {
                fallback[c] = true;
                n_fallback += 1;
            }
        }
        Ok(LinearStencil {
            n_cells,
            centres,
            f_off,
            f_nbr,
            p_off,
            p_nbr,
            fallback,
            n_fallback,
        })
    }
}

impl LinearStencil {
    /// The cells the stencils were built for.
    pub fn n_cells(&self) -> usize {
        self.n_cells
    }

    /// The number of cells with `det G <= 1e-12 (tr G / 3)^3` (117.7).
    pub fn n_fallback(&self) -> usize {
        self.n_fallback
    }

    /// Whether cell `c` has no 3-D stencil, so (117.7) gives it `g_c = 0`.
    pub fn is_fallback(&self, c: usize) -> bool {
        self.fallback[c]
    }

    /// `N_f(c)`: ascending, no duplicate, never `c`.
    pub fn face_neighbours(&self, c: usize) -> &[u32] {
        &self.f_nbr[self.f_off[c]..self.f_off[c + 1]]
    }

    /// `N_p(c)`: ascending, no duplicate, never `c`.
    pub fn point_neighbours(&self, c: usize) -> &[u32] {
        &self.p_nbr[self.p_off[c]..self.p_off[c + 1]]
    }
}

impl LinearStencil {
    /// `g_c` of (117.7) for the field `phi` (`[n_cells]`, f64): the normal
    /// equations `G g = b`, `G = sum d_n d_n^T`, `b = sum d_n (phi_n -
    /// phi_c)` with `d_n = C_n - C_c`, solved by Cramer's rule. Exact for a
    /// linear field whenever `G` is invertible (Mavriplis 2003). The zero
    /// vector on a fallback cell.
    pub fn gradient(&self, phi: &[f64], c: usize) -> DVec3 {
        if self.fallback[c] {
            return DVec3::new(0.0, 0.0, 0.0);
        }
        let cc = self.centres[c];
        let (mut g00, mut g01, mut g02, mut g11, mut g12, mut g22) =
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        let (mut b0, mut b1, mut b2) = (0.0, 0.0, 0.0);
        for &ni in &self.f_nbr[self.f_off[c]..self.f_off[c + 1]] {
            let n = ni as usize;
            let d = self.centres[n] - cc;
            let dp = phi[n] - phi[c];
            g00 += d.x * d.x;
            g01 += d.x * d.y;
            g02 += d.x * d.z;
            g11 += d.y * d.y;
            g12 += d.y * d.z;
            g22 += d.z * d.z;
            b0 += d.x * dp;
            b1 += d.y * dp;
            b2 += d.z * dp;
        }
        let det = det3(g00, g01, g02, g11, g12, g22);
        DVec3::new(
            (b0 * (g11 * g22 - g12 * g12) - g01 * (b1 * g22 - g12 * b2)
                + g02 * (b1 * g12 - g11 * b2))
                / det,
            (g00 * (b1 * g22 - b2 * g12) - b0 * (g01 * g22 - g12 * g02)
                + g02 * (g01 * b2 - b1 * g02))
                / det,
            (g00 * (g11 * b2 - g12 * b1) - g01 * (g01 * b2 - b1 * g02)
                + b0 * (g01 * g12 - g11 * g02))
                / det,
        )
    }
}

impl LinearStencil {
    /// `(m_c, M_c)` of (117.8): the range of `phi` over `c` and its point
    /// neighbours - values of real cells, so the clip of (117.8) never
    /// creates a new extremum (Barth & Jespersen 1989).
    pub fn bounds(&self, phi: &[f64], c: usize) -> (f64, f64) {
        let (mut lo, mut hi) = (phi[c], phi[c]);
        for &ni in &self.p_nbr[self.p_off[c]..self.p_off[c + 1]] {
            let v = phi[ni as usize];
            lo = lo.min(v);
            hi = hi.max(v);
        }
        (lo, hi)
    }
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
    /// every point are in world coordinates, so `hm` must be a world mesh:
    /// `output_plan::volume_target` builds one from `raw` when the driver's
    /// mesh is stored about F32-10's local origin, and the written transform
    /// stays the world transform (117.2 item 4). Deterministic and
    /// single-threaded; no float is hashed.
    pub fn build(raw: &PolyMeshRaw, hm: &HostMesh, spec: &LatticeSpec) -> Result<VoxelMap> {
        let t0 = Instant::now();
        let shape = lattice_shape(raw, hm, spec)?;
        // The linear stencil, when `linear` sampling is asked for (117.5) -
        // inside the timed section, so `report.build_s` covers it.
        let linear = match spec.sample {
            LatticeSample::Linear => Some(LinearStencil::build(raw, hm)?),
            LatticeSample::Cell => None,
        };
        let g = hm.geom64();
        let LatticeShape { lo, l, voxel_requested: h_req, voxel: h, coarsen } = shape;

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
            n_linear_fallback: linear.as_ref().map_or(0, LinearStencil::n_fallback),
            build_s: t0.elapsed().as_secs_f64(),
            note,
        };
        Ok(VoxelMap {
            n,
            lo,
            voxel: h,
            cell,
            n_cells: hm.n_cells,
            linear,
            spec: spec.clone(),
            report,
        })
    }
}

// ==========================================================================
//  Tests (SPEC-LIT 117.9, VIO-09)
// ==========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::nvdb::lattice_fixture as lf;

    /// The 2:1 mesh of `mesh/refined.rs`, a lattice of `h = 1/24` on its
    /// DEFAULT box (the point bounding box, 24^3 voxels), cell sampling, and
    /// the Gaussian plume `f_c = exp(-|C_c - centre|^2 / 0.05)` sampled onto
    /// it (117.5), with the fluid mask.
    fn plume_voxels() -> (VoxelMap, Vec<f64>, Vec<bool>) {
        let (raw, hm) = lf::refined();
        let spec = LatticeSpec { voxel: Some(1.0 / 24.0), ..Default::default() };
        let vm = VoxelMap::build(&raw, &hm, &spec).expect("VoxelMap::build");
        let g = hm.geom64();
        let centre = DVec3::new(0.5, 0.5, 0.5);
        let f: Vec<Scalar> = (0..hm.n_cells)
            .map(|c| {
                let d = g.c[c] - centre;
                (-(d.dot(d)) / 0.05).exp() as Scalar
            })
            .collect();
        let vox = vm.sample(&f).expect("sample");
        let mask = vm.fluid_mask();
        (vm, vox, mask)
    }

    /// The test's OWN (117.11), deliberately not
    /// [`max_column_optical_depth`]: the same max over the three axes'
    /// lines, written out again here so the check is independent of the
    /// function under test.
    fn own_max_column(n: [usize; 3], h: [f64; 3], s: &[f64]) -> f64 {
        let at = |i: usize, j: usize, k: usize| i + n[0] * (j + n[1] * k);
        let mut best = 0.0f64;
        for k in 0..n[2] {
            for j in 0..n[1] {
                let mut sum = 0.0f64;
                for i in 0..n[0] {
                    sum += s[at(i, j, k)];
                }
                best = best.max(h[0] * sum);
            }
        }
        for k in 0..n[2] {
            for i in 0..n[0] {
                let mut sum = 0.0f64;
                for j in 0..n[1] {
                    sum += s[at(i, j, k)];
                }
                best = best.max(h[1] * sum);
            }
        }
        for j in 0..n[1] {
            for i in 0..n[0] {
                let mut sum = 0.0f64;
                for k in 0..n[2] {
                    sum += s[at(i, j, k)];
                }
                best = best.max(h[2] * sum);
            }
        }
        best
    }

    /// VIO-09's gate, in f64 before any rounding: the max column optical
    /// depth of (117.12)'s sigma is exactly `tau` (within 1e-12 relative),
    /// no column exceeds it, and the returned `D` is the test's own
    /// (117.11) of `f^`. Checked at `tau = 2.0` (the visual default) and
    /// `tau = 0.5`.
    #[test]
    fn optical_depth_target() {
        let (vm, vox, mask) = plume_voxels();
        let n = vm.n;
        let h = [vm.voxel; 3];
        let range = fluid_range(&vox, Some(&mask)).expect("the plume has fluid voxels");
        let f_hat = normalise_for_optical_depth(&vox, Some(&mask), range);
        for tau in [2.0, 0.5] {
            // `super::` - this test's own name shadows the function's.
            let (sigma, d) = super::optical_depth_target(n, h, &f_hat, tau);
            assert_eq!(d, own_max_column(n, h, &f_hat), "D is the test's own (117.11)");
            let got = own_max_column(n, h, &sigma);
            let rel = ((got - tau) / tau).abs();
            assert!(rel < 1e-12, "tau {tau}: densest column {got} vs tau, rel {rel}");
            // No column may exceed tau (1 + 1e-12) - Beer-Lambert's bound;
            // `own_max_column` IS the max over every column of every axis.
            let ceil = tau * (1.0 + 1e-12);
            assert!(got <= ceil, "tau {tau}: a column reaches {got} > {ceil}");
            // The gate itself, AT the file's precision: sigma as f32.
            let s32: Vec<f64> = sigma.iter().map(|&s| s as f32 as f64).collect();
            let got32 = own_max_column(n, h, &s32);
            let rel32 = ((got32 - tau) / tau).abs();
            println!("optical_depth_target: tau {tau}, f32-rounded densest column {got32}, \
                      relative error {rel32:.3e}");
            assert!(rel32 < 1e-2, "tau {tau}: f32 relative error {rel32} >= 1 %");
        }
    }

    /// (117.11) on a box whose spacing differs per axis: the deepest line
    /// is the one whose AXIS spacing multiplies it, not the longest one.
    #[test]
    fn max_column_optical_depth_uses_the_axis_spacing() {
        // Two voxels along x only: the y lines are single voxels at h_y = 2,
        // the z lines single voxels at h_z = 4 - the z line of voxel 0 wins.
        assert_eq!(max_column_optical_depth([2, 1, 1], [1.0, 2.0, 4.0], &[1.0, 1.0]), 4.0);
        // Voxel 1 is not fluid: only voxel 0's z line survives, still 4.0.
        assert_eq!(max_column_optical_depth([2, 1, 1], [1.0, 2.0, 4.0], &[1.0, 0.0]), 4.0);
        // One voxel, all zero: no line has anything to integrate.
        assert_eq!(max_column_optical_depth([1, 1, 1], [1.0, 1.0, 1.0], &[0.0]), 0.0);
    }

    /// A constant field has f_hi = f_lo: f^ is 0 everywhere, so the write
    /// sees D = 0 and writes the all-zero density grid (117.9).
    #[test]
    fn optical_depth_zero_field() {
        let f = [0.5f64, 0.5, 0.5, 0.5];
        let range = fluid_range(&f, None).expect("all fluid");
        assert_eq!(range, [0.5, 0.5]);
        let f_hat = normalise_for_optical_depth(&f, None, range);
        assert!(f_hat.iter().all(|&s| s == 0.0));
        let (sigma, d) = super::optical_depth_target([2, 2, 1], [1.0, 1.0, 1.0], &f_hat, 2.0);
        assert_eq!(d, 0.0);
        assert!(sigma.iter().all(|&s| s == 0.0));
    }

    /// (117.10)'s clamp, the NaN rule and the NONE rule, and the default
    /// range's own two rules.
    #[test]
    fn normalise_clamps() {
        let f = [0.1, 0.5, 0.9, f64::NAN, 0.5];
        let fluid = [true, true, true, true, false];
        let f_hat = normalise_for_optical_depth(&f, Some(&fluid), [0.25, 0.75]);
        assert_eq!(f_hat[0], 0.0);
        assert_eq!(f_hat[1], 0.5, "(0.5 - 0.25) / 0.5 is exact");
        assert_eq!(f_hat[2], 1.0);
        assert_eq!(f_hat[3], 0.0, "a non-finite f_v gives 0");
        assert_eq!(f_hat[4], 0.0, "a NONE voxel gives 0");

        // fluid_range ignores NONE voxels and non-finite values alike.
        let g = [0.1, f64::NAN, 0.9, 0.5];
        assert_eq!(fluid_range(&g, Some(&[true, true, true, true])), Some([0.1, 0.9]));
        assert_eq!(fluid_range(&g, Some(&[false, false, false, true])), Some([0.5, 0.5]));
        assert_eq!(fluid_range(&g, Some(&[true, false, false, false])), Some([0.1, 0.1]));
        assert_eq!(fluid_range(&[f64::NAN, f64::NAN], None), None, "no finite fluid voxel");
        assert_eq!(fluid_range(&[0.5, 0.7], Some(&[false, false])), None);
        // `fluid: None` = every voxel is fluid.
        assert_eq!(fluid_range(&[0.25, 1.0], None), Some([0.25, 1.0]));
    }

    /// (117.14), exactly: `K_m = 8.7 m^2/g = 8700 m^2/kg`, and sigma is
    /// `K_m * rho_s` bit for bit, with the negatives written 0 and counted.
    #[test]
    fn physical_extinction_is_km_rho() {
        assert_eq!(SMOKE_SPECIFIC_EXTINCTION, 8.7 * 1000.0);
        let rho_s = [0.0, 1e-3, 2.5e-4, -1e-5];
        let (sigma, count) = physical_extinction(&rho_s, None, SMOKE_SPECIFIC_EXTINCTION);
        let want = [0.0, SMOKE_SPECIFIC_EXTINCTION * 1e-3, SMOKE_SPECIFIC_EXTINCTION * 2.5e-4, 0.0];
        for (s, w) in sigma.iter().zip(want) {
            assert_eq!(*s, w, "sigma is km * rho_s bit for bit");
        }
        assert_eq!(count, 1, "exactly one negative rho_s, written 0 and counted");
        // A NONE voxel is 0 and its negative rho_s is not counted.
        let (sigma, count) =
            physical_extinction(&rho_s, Some(&[true, true, false, true]), SMOKE_SPECIFIC_EXTINCTION);
        assert_eq!(sigma[2], 0.0, "the NONE voxel is 0 even at a positive rho_s");
        assert_eq!(sigma[3], 0.0, "the negative fluid voxel is 0");
        assert_eq!(count, 1);
    }
}
