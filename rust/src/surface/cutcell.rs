// meteor-cfd - Copyright (c) 2026 주식회사 메테오시뮬레이션 (Meteo Simulation Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

//! Embedded-boundary fractions (SPEC-LIT section 24): supersampled face and
//! volume fractions, the closure-defined cut face, small-cell merging.
//!
//! Written from:
//!   ofgpu `SPEC-LIT.md` section 24 (24.1 classification refresher, 24.2 face
//!     area fractions, 24.3 the cut face by closure, 24.4 volume fractions and
//!     centroids, 24.5 small cells);
//!   Aftosmis, Berger & Melton, *AIAA J.* 36(6) (1998) 952 - the Cartesian
//!     cut-cell EB construction section 24 generalises from castellation;
//!   the AMReX (BSD-3, readable) EB data model - a volume fraction per cell,
//!     an area fraction per face, one boundary face per cut cell - cited by
//!     section 24 as the shape this module's data follows.
//! No GPL-licensed source was consulted.
//!
//! # The supersample lattice (24.1, 24.4)
//!
//! A cell is classified by casting one column of the surface's x-ray index
//! ([`super::TriIndex::crossings_x`], the same watertight intersection
//! [`super::classify`] uses) per `(j, k)` sub-column of an `s x s` tangential
//! grid, and reading off the parity of the `s` sample centres along that
//! column. That is "one crossing sort per (y,z) sub-column" (24.1): a single
//! axis suffices, because every sample point the cell needs - the `s^3`
//! volume centres AND every face's `s x s` layer - comes out of that one
//! `s x s x s` boolean array, sliced. Slicing the array's own boundary layers
//! for `alpha_f` (rather than re-classifying at the face plane) is exactly
//! "the sample points are the face's own supersample columns, so no extra
//! classification pass is needed" (24.2).
//!
//! `classify_one_cell` never runs on a cell the surface cannot reach: a cell
//! is trusted to whatever [`super::classify::classify`] already said about it
//! whenever the nearest surface triangle is farther away than the cell's own
//! half-diagonal, because no point of a cell can be closer to the surface
//! than that without being nearer to the cell's own centre than its
//! half-diagonal - the exact contrapositive of "the nearest triangle is more
//! than a half-diagonal away". That is what keeps supersampling to a thin
//! shell of boundary cells instead of every cell in the block.
//!
//! # The cut face (24.3)
//!
//! `Sf_cut = -sum(alpha_f * Sf_full)` over the six axis directions, by
//! definition, so THIS CELL's own closure `sum_f Sf = 0` holds to round-off
//! against ITS OWN fractions, regardless of how approximate they are (24.3;
//! checked directly, per-cell, by this module's own tests). Two independently
//! classified neighbours can still disagree, by a small amount, on the
//! `alpha_f` of the face between them - 24.4's "one lattice, no seams"
//! promise covers only the degenerate all-fluid/all-solid case - so the mesh
//! ASSEMBLY that turns these fractions into an actual polyhedron (in
//! `blockgen.rs`) recomputes each cell's cut face against whatever its
//! neighbours actually settled the shared faces to, rather than trusting the
//! value computed here in isolation. `cut_sf` and `cut_cf` below are that
//! recomputation's starting point (exact on their own) and its fallback
//! position, not its last word.
//!
//! `cut_cf`, the cut face's centroid, is the mean of every INTERNAL
//! fluid/solid transition inside the cell's own `s^3` array (adjacent
//! samples that disagree, `*DESIGN*` per 24.3 - "adequate at first order"):
//! a transition at the cell's own boundary layer is already accounted for by
//! that face's `alpha_f < 1`, not by the cut face, so only interior
//! transitions are counted.
//!
//! # Small-cell merging (24.5)
//!
//! Implemented as a union-find over the ORIGINAL (pre-merge) cells. Merging a
//! cell `r` into its largest-shared-face fluid neighbour `n` sums volumes and
//! volume-weights the centroid, then simply drops `r`'s identity - every
//! other face `r` owned (internal to a third cell, or `r`'s own cut/wall
//! face) is re-pointed at `n` by resolving through the union-find at
//! mesh-assembly time. An internal face whose two sides resolve to the SAME
//! surviving cell has become interior to the merged polyhedron and is
//! dropped outright: by the divergence theorem its two contributions
//! (`+Sf` from one side, `-Sf` from the other) already cancel, so dropping it
//! costs nothing and is what "the gather-CSR assembly already handles it"
//! (24.5) means - a merged cell is just a polyhedron with more faces, no
//! different in kind from an ordinary one.
//!
//! `theta_min` is checked against each SURVIVING cell's own original
//! `v_full`, forever - even after several slivers have been absorbed into it
//! - because that is the denominator the conditioning concern (24.5, "for
//! THIS solver the harm is conditioning") is about: is the control volume
//! that used to sit at this grid location still adequate.

use std::collections::HashMap;

use crate::error::{Error, Result};
use crate::surface::classify::{classify, BlockAxes};
use crate::surface::{Surface, TriIndex};
use crate::{Scalar, Vec3};

/// `s` in 24.2/24.4: the supersample lattice is `s` samples per axis inside a
/// cell (`s^3` for volume, `s^2` per face). *DESIGN*, per 24.2.
pub const DEFAULT_SUPERSAMPLE: usize = 16;
/// `theta_min` in 24.5. *DESIGN*.
pub const DEFAULT_THETA_MIN: Scalar = 0.2;

// ==========================================================================
//  Per-cell fractions
// ==========================================================================

/// What §24.1's classification refresher calls a cell.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CellState {
    Fluid,
    Solid,
    Cut,
}

/// Everything section 24 computes for one non-solid cell: volume and area
/// fractions, centroids, and (for a `Cut` cell) the closure-defined cut face.
///
/// A `Fluid` cell carries `theta = 1`, every `alpha = 1`, and a zero cut face
/// - the same values a `Cut` cell would carry if its interface receded to
/// nothing, which is exactly the "no seams" consistency 24.4 asks for.
#[derive(Debug, Clone)]
pub struct CellFractions {
    pub state: CellState,
    /// Fluid volume fraction, `V / V_full`.
    pub theta: Scalar,
    /// `theta * V_full`.
    pub volume: Scalar,
    /// Mean of the fluid supersample positions (cell centre for `Fluid`).
    pub centroid: Vec3,
    /// Face area fractions, axis order `-x +x -y +y -z +z` (24.2).
    pub alpha: [Scalar; 6],
    /// This cell's OWN closure-defined cut face area vector (24.3), computed
    /// from its OWN `alpha` array alone: `-sum(alpha[d] * Sf_full[d])`. Zero
    /// unless `Cut`.
    ///
    /// Exact for THIS cell in isolation (checked directly in this module's
    /// own tests), but two neighbouring cells classified independently can
    /// each compute a slightly different `alpha_f` for the face they share
    /// (24.4's "one lattice, no seams" guarantee is stated only for the
    /// degenerate all-fluid/all-solid case) - so mesh ASSEMBLY, which decides
    /// one shared value per face and therefore knows what a cell actually
    /// ended up with, is where the field this struct feeds recomputes the
    /// cut face that closes the cell exactly. See `blockgen.rs`'s cut-cell
    /// section for that reconciliation; this field is its starting point and
    /// its position (`cut_cf`, below), not its last word.
    pub cut_sf: Vec3,
    /// The cut face's centroid (24.3). Equals `centroid` unless `Cut`.
    pub cut_cf: Vec3,
    /// Index into the surface's triangle array of the triangle nearest
    /// `cut_cf` - the patch a carved wall face would take (23.4). Meaningful
    /// only when `state == Cut`.
    pub cut_tri: usize,
    /// The cell's full (uncut) volume - the `theta_min` denominator (24.5).
    pub v_full: Scalar,
}

// ==========================================================================
//  The field
// ==========================================================================

/// Per-cell fractions over a whole block: `None` where the cell is `Solid`.
#[derive(Debug, Clone)]
pub struct CutCellField {
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    pub xn: Vec<Scalar>,
    pub yn: Vec<Scalar>,
    pub zn: Vec<Scalar>,
    /// `i + nx*(j + ny*k)` indexed, same rule as `blockgen`/`classify`.
    pub cells: Vec<Option<CellFractions>>,
    pub n_solid: usize,
    /// Cells classified `Fluid` (whether trusted from the coarse pass or
    /// found fully fluid by the fine lattice).
    pub n_fluid: usize,
    pub n_cut: usize,
    pub s: usize,
}

impl CutCellField {
    #[inline]
    pub fn cell_index(&self, i: usize, j: usize, k: usize) -> usize {
        i + self.nx * (j + self.ny * k)
    }

    #[inline]
    pub fn decompose(&self, c: usize) -> (usize, usize, usize) {
        let i = c % self.nx;
        let t = c / self.nx;
        (i, t % self.ny, t / self.ny)
    }

    /// The neighbour of `(i,j,k)` in direction `dir` (`-x +x -y +y -z +z`),
    /// or `None` at the domain edge.
    pub fn neighbour_index(&self, i: usize, j: usize, k: usize, dir: usize) -> Option<usize> {
        match dir {
            0 => (i > 0).then(|| self.cell_index(i - 1, j, k)),
            1 => (i + 1 < self.nx).then(|| self.cell_index(i + 1, j, k)),
            2 => (j > 0).then(|| self.cell_index(i, j - 1, k)),
            3 => (j + 1 < self.ny).then(|| self.cell_index(i, j + 1, k)),
            4 => (k > 0).then(|| self.cell_index(i, j, k - 1)),
            _ => (k + 1 < self.nz).then(|| self.cell_index(i, j, k + 1)),
        }
    }

    pub fn cell_bounds(&self, i: usize, j: usize, k: usize) -> (Vec3, Vec3) {
        (
            Vec3::new(self.xn[i], self.yn[j], self.zn[k]),
            Vec3::new(self.xn[i + 1], self.yn[j + 1], self.zn[k + 1]),
        )
    }

    pub fn full_face_area(&self, i: usize, j: usize, k: usize, dir: usize) -> Scalar {
        let hx = self.xn[i + 1] - self.xn[i];
        let hy = self.yn[j + 1] - self.yn[j];
        let hz = self.zn[k + 1] - self.zn[k];
        match dir {
            0 | 1 => hy * hz,
            2 | 3 => hx * hz,
            _ => hx * hy,
        }
    }

    /// The full outward area vector of `(i,j,k)`'s face `dir`, axis order
    /// `-x +x -y +y -z +z` - what `alpha_f` scales (24.2).
    pub fn full_sf(&self, i: usize, j: usize, k: usize, dir: usize) -> Vec3 {
        let a = self.full_face_area(i, j, k, dir);
        match dir {
            0 => Vec3::new(-a, 0.0, 0.0),
            1 => Vec3::new(a, 0.0, 0.0),
            2 => Vec3::new(0.0, -a, 0.0),
            3 => Vec3::new(0.0, a, 0.0),
            4 => Vec3::new(0.0, 0.0, -a),
            _ => Vec3::new(0.0, 0.0, a),
        }
    }
}

// ==========================================================================
//  24.1-24.4: classification, fractions, the cut face
// ==========================================================================

/// One sample position along an axis: the centre of sub-interval `idx` of
/// `s`, as a fraction of the interval `[0,1]`.
#[inline]
fn sub_frac(idx: usize, s: usize) -> Scalar {
    (idx as Scalar + 0.5) / s as Scalar
}

/// The full section 24 pass over one candidate ("might be cut") cell: an
/// `s x s x s` array built from `s x s` column casts along x
/// ([`TriIndex::crossings_x`], reused from 23.3's watertight intersection),
/// then reduced to `theta`, `alpha_f` (by slicing the array's own boundary
/// layers, 24.2) and, if the cell turns out mixed, the closure-defined cut
/// face (24.3). `cut_tri` is left `0`; the caller fills it in because it
/// needs the shared [`TriIndex`] for a query this function has no other use
/// for.
// The six cube faces below are written with one index expression shape,
// `fluid[ii + s * (jj + s * kk)]`, and the fixed coordinate substituted in
// place: `0` for the low face and `s - 1` for the high one. That leaves
// literal `0 +` and `* 0` terms, which clippy reads as an identity and an
// erasing operation - correctly, in isolation. They are kept because the
// symmetry is the point: each low/high pair sits on adjacent lines and
// differs only in the substituted coordinate, so a transposed axis is
// visible by eye. Folding the constants away would make the six faces read
// as six different expressions and hide exactly that class of bug.
#[allow(clippy::erasing_op, clippy::identity_op)]
fn classify_one_cell(
    idx: &TriIndex,
    xn: &[Scalar],
    yn: &[Scalar],
    zn: &[Scalar],
    i: usize,
    j: usize,
    k: usize,
    s: usize,
) -> CellFractions {
    let (x0, x1) = (xn[i], xn[i + 1]);
    let (y0, y1) = (yn[j], yn[j + 1]);
    let (z0, z1) = (zn[k], zn[k + 1]);
    let (hx, hy, hz) = (x1 - x0, y1 - y0, z1 - z0);
    let v_full = hx * hy * hz;

    // Flat `s^3` array, `ii + s*(jj + s*kk)`, filled one x-column per (jj,kk)
    // - "one crossing sort per (y,z) sub-column" (24.1).
    //
    // Parity convention matches `classify::parity_fill`/`cast_axis` exactly:
    // an ODD number of crossings before a sample means it is INSIDE the
    // surface, i.e. SOLID (`SolidMask::solid`'s own doc: "true = inside the
    // surface = removed from the fluid mesh"). A sample is fluid on the EVEN
    // parity.
    let mut fluid = vec![false; s * s * s];
    for kk in 0..s {
        let z = z0 + sub_frac(kk, s) * hz;
        for jj in 0..s {
            let y = y0 + sub_frac(jj, s) * hy;
            let hits = idx.crossings_x(y, z);
            let mut h = 0usize;
            for ii in 0..s {
                let x = x0 + sub_frac(ii, s) * hx;
                while h < hits.len() && hits[h].0 < x {
                    h += 1;
                }
                fluid[ii + s * (jj + s * kk)] = (h & 1) == 0;
            }
        }
    }

    let n_fluid = fluid.iter().filter(|&&f| f).count();
    let n_total = s * s * s;
    let theta = n_fluid as Scalar / n_total as Scalar;

    if n_fluid == 0 {
        return CellFractions {
            state: CellState::Solid,
            theta: 0.0,
            volume: 0.0,
            centroid: Vec3::ZERO,
            alpha: [0.0; 6],
            cut_sf: Vec3::ZERO,
            cut_cf: Vec3::ZERO,
            cut_tri: 0,
            v_full,
        };
    }

    let centre = Vec3::new(0.5 * (x0 + x1), 0.5 * (y0 + y1), 0.5 * (z0 + z1));
    if n_fluid == n_total {
        // 24.4 consistency: an all-fluid cell gets theta = 1, alpha = 1 on
        // every face, no seams.
        return CellFractions {
            state: CellState::Fluid,
            theta: 1.0,
            volume: v_full,
            centroid: centre,
            alpha: [1.0; 6],
            cut_sf: Vec3::ZERO,
            cut_cf: centre,
            cut_tri: 0,
            v_full,
        };
    }

    // ---- CUT: volume centroid -------------------------------------------
    let pos = |ii: usize, jj: usize, kk: usize| -> Vec3 {
        Vec3::new(
            x0 + sub_frac(ii, s) * hx,
            y0 + sub_frac(jj, s) * hy,
            z0 + sub_frac(kk, s) * hz,
        )
    };
    let mut csum = Vec3::ZERO;
    for kk in 0..s {
        for jj in 0..s {
            for ii in 0..s {
                if fluid[ii + s * (jj + s * kk)] {
                    csum += pos(ii, jj, kk);
                }
            }
        }
    }
    let centroid = csum / n_fluid as Scalar;

    // ---- 24.2: face fractions, sliced from the same array -----------------
    let mut face_fluid = [0usize; 6];
    for jj in 0..s {
        for kk in 0..s {
            if fluid[0 + s * (jj + s * kk)] {
                face_fluid[0] += 1;
            }
            if fluid[(s - 1) + s * (jj + s * kk)] {
                face_fluid[1] += 1;
            }
        }
    }
    for ii in 0..s {
        for kk in 0..s {
            if fluid[ii + s * (0 + s * kk)] {
                face_fluid[2] += 1;
            }
            if fluid[ii + s * ((s - 1) + s * kk)] {
                face_fluid[3] += 1;
            }
        }
    }
    for ii in 0..s {
        for jj in 0..s {
            if fluid[ii + s * (jj + s * 0)] {
                face_fluid[4] += 1;
            }
            if fluid[ii + s * (jj + s * (s - 1))] {
                face_fluid[5] += 1;
            }
        }
    }
    let s2 = (s * s) as Scalar;
    let alpha = [
        face_fluid[0] as Scalar / s2,
        face_fluid[1] as Scalar / s2,
        face_fluid[2] as Scalar / s2,
        face_fluid[3] as Scalar / s2,
        face_fluid[4] as Scalar / s2,
        face_fluid[5] as Scalar / s2,
    ];

    // ---- 24.3: the cut face, by closure ------------------------------------
    let a_yz = hy * hz;
    let a_xz = hx * hz;
    let a_xy = hx * hy;
    let full_sf = [
        Vec3::new(-a_yz, 0.0, 0.0),
        Vec3::new(a_yz, 0.0, 0.0),
        Vec3::new(0.0, -a_xz, 0.0),
        Vec3::new(0.0, a_xz, 0.0),
        Vec3::new(0.0, 0.0, -a_xy),
        Vec3::new(0.0, 0.0, a_xy),
    ];
    let mut sum = Vec3::ZERO;
    for d in 0..6 {
        sum += full_sf[d] * alpha[d];
    }
    let cut_sf = -sum;

    // Centroid: mean of every INTERNAL fluid/solid transition midpoint. A
    // transition at the array's own outer boundary is already accounted for
    // by that face's alpha < 1, not by the cut face - see the module doc.
    let mut isum = Vec3::ZERO;
    let mut icount = 0usize;
    for kk in 0..s {
        for jj in 0..s {
            for ii in 0..s {
                let f0 = fluid[ii + s * (jj + s * kk)];
                if ii + 1 < s && f0 != fluid[(ii + 1) + s * (jj + s * kk)] {
                    isum += (pos(ii, jj, kk) + pos(ii + 1, jj, kk)) * 0.5;
                    icount += 1;
                }
                if jj + 1 < s && f0 != fluid[ii + s * ((jj + 1) + s * kk)] {
                    isum += (pos(ii, jj, kk) + pos(ii, jj + 1, kk)) * 0.5;
                    icount += 1;
                }
                if kk + 1 < s && f0 != fluid[ii + s * (jj + s * (kk + 1))] {
                    isum += (pos(ii, jj, kk) + pos(ii, jj, kk + 1)) * 0.5;
                    icount += 1;
                }
            }
        }
    }
    // `icount == 0` cannot happen for a genuinely mixed array: the fluid and
    // solid samples are both non-empty in a connected s^3 grid, so a shortest
    // path between one of each must cross an internal fluid/solid edge. The
    // fallback exists only so a future change to the connectivity argument
    // fails soft instead of dividing by zero.
    let cut_cf = if icount > 0 { isum / icount as Scalar } else { centroid };

    CellFractions {
        state: CellState::Cut,
        theta,
        volume: theta * v_full,
        centroid,
        alpha,
        cut_sf,
        cut_cf,
        cut_tri: 0,
        v_full,
    }
}

/// Classify every cell of the block into fluid/solid/cut fractions (section
/// 24). `s` is the supersample lattice size ([`DEFAULT_SUPERSAMPLE`]).
///
/// Reuses [`classify`] (23.3, cell-centre column parity) for the coarse pass,
/// then refines only the cells the surface can actually reach - see the
/// module doc for the half-diagonal argument that makes that exact rather
/// than a heuristic.
pub fn classify_cutcells(axes: &BlockAxes, surf: &Surface, s: usize) -> Result<CutCellField> {
    if s == 0 {
        return Err(Error::Config(format!(
            "cutcell: supersample s must be at least 1, got {s}"
        )));
    }

    let mask = classify(axes, surf)?;
    let (nx, ny, nz) = (mask.nx, mask.ny, mask.nz);
    let n_cells = nx * ny * nz;

    // Bucket size ~ the mesh spacing, the same rule `classify` uses.
    let mut hint: Scalar = 0.0;
    hint += (axes.xn[axes.xn.len() - 1] - axes.xn[0]).abs() / nx as Scalar;
    hint += (axes.yn[axes.yn.len() - 1] - axes.yn[0]).abs() / ny as Scalar;
    hint += (axes.zn[axes.zn.len() - 1] - axes.zn[0]).abs() / nz as Scalar;
    hint = (hint / 3.0).max(Scalar::MIN_POSITIVE);
    let idx = TriIndex::new(surf, hint)?;

    let mut cells: Vec<Option<CellFractions>> = vec![None; n_cells];
    let (mut n_solid, mut n_fluid, mut n_cut) = (0usize, 0usize, 0usize);

    let xn = axes.xn;
    let yn = axes.yn;
    let zn = axes.zn;

    for c in 0..n_cells {
        let i = c % nx;
        let t = c / nx;
        let (j, k) = (t % ny, t / ny);

        let lo = Vec3::new(xn[i], yn[j], zn[k]);
        let hi = Vec3::new(xn[i + 1], yn[j + 1], zn[k + 1]);
        let centre = (lo + hi) * 0.5;
        let half_diag = 0.5 * (hi - lo).mag();
        let (_, dist) = idx.nearest_triangle(centre);

        if dist > half_diag {
            // The surface cannot reach this cell (module doc); trust §23.3.
            if mask.solid[c] {
                n_solid += 1;
            } else {
                n_fluid += 1;
                let v_full = (xn[i + 1] - xn[i]) * (yn[j + 1] - yn[j]) * (zn[k + 1] - zn[k]);
                cells[c] = Some(CellFractions {
                    state: CellState::Fluid,
                    theta: 1.0,
                    volume: v_full,
                    centroid: centre,
                    alpha: [1.0; 6],
                    cut_sf: Vec3::ZERO,
                    cut_cf: centre,
                    cut_tri: 0,
                    v_full,
                });
            }
            continue;
        }

        let mut cf = classify_one_cell(&idx, xn, yn, zn, i, j, k, s);
        match cf.state {
            CellState::Solid => n_solid += 1,
            CellState::Fluid => {
                n_fluid += 1;
                cells[c] = Some(cf);
            }
            CellState::Cut => {
                cf.cut_tri = idx.nearest_triangle(cf.cut_cf).0;
                n_cut += 1;
                cells[c] = Some(cf);
            }
        }
    }

    Ok(CutCellField {
        nx,
        ny,
        nz,
        xn: xn.to_vec(),
        yn: yn.to_vec(),
        zn: zn.to_vec(),
        cells,
        n_solid,
        n_fluid,
        n_cut,
        s,
    })
}

// ==========================================================================
//  24.5: small-cell merging
// ==========================================================================

/// Union-find `find`, with path compression (SPEC section 1's family of
/// tricks does not cover this, but the shape is the standard one).
fn find(parent: &mut [u32], x: u32) -> u32 {
    let mut r = x;
    while parent[r as usize] != r {
        r = parent[r as usize];
    }
    let mut c = x;
    while parent[c as usize] != r {
        let next = parent[c as usize];
        parent[c as usize] = r;
        c = next;
    }
    r
}

/// The result of merging every cell with `theta_c < theta_min` into its
/// largest-shared-face fluid neighbour, iterated until none remain (24.5).
///
/// `root`, `volume` and `centroid` are indexed by the ORIGINAL cell id;
/// `root[c]` is the live cell `c` now belongs to (itself, if it was never
/// merged), and `volume`/`centroid` are valid at root positions only
/// (`root[r] == r`).
#[derive(Debug, Clone)]
pub struct MergeResult {
    pub root: Vec<u32>,
    pub volume: Vec<Scalar>,
    pub centroid: Vec<Vec3>,
    pub n_merged: usize,
}

/// Merge every cell below `theta_min` into the fluid neighbour it shares the
/// largest total face area with, repeating until every surviving cell clears
/// the threshold (24.5). Refuses (naming the cell) a sliver with no fluid
/// neighbour at all to merge into.
pub fn merge_small_cells(field: &CutCellField, theta_min: Scalar) -> Result<MergeResult> {
    let n = field.cells.len();
    let mut parent: Vec<u32> = (0..n as u32).collect();
    let mut volume = vec![0.0 as Scalar; n];
    let mut centroid = vec![Vec3::ZERO; n];
    let mut v_full = vec![0.0 as Scalar; n];

    for (c, cf) in field.cells.iter().enumerate() {
        if let Some(cf) = cf {
            volume[c] = cf.volume;
            centroid[c] = cf.centroid;
            v_full[c] = cf.v_full;
        }
    }

    let mut n_merged = 0usize;

    loop {
        // First live root below threshold, in cell-index order - any order
        // converges (every merge strictly increases its survivor's theta and
        // strictly reduces the live cell count), so the choice only affects
        // which neighbour absorbs which sliver, not whether the loop ends.
        let mut victim: Option<usize> = None;
        for c in 0..n {
            if field.cells[c].is_none() {
                continue;
            }
            let r = find(&mut parent, c as u32) as usize;
            if r != c || v_full[r] <= 0.0 {
                continue;
            }
            if volume[r] / v_full[r] < theta_min {
                victim = Some(r);
                break;
            }
        }
        let Some(r) = victim else { break };

        // Total shared (already alpha-scaled) face area to every OTHER live
        // group, summed over every original cell still resolving to `r` -
        // "a merged cell can absorb several slivers" (24.5) means a group can
        // present more than one contact face to the same neighbour.
        let mut area_by_group: HashMap<u32, Scalar> = HashMap::new();
        for c in 0..n {
            if field.cells[c].is_none() || find(&mut parent, c as u32) as usize != r {
                continue;
            }
            let (i, j, k) = field.decompose(c);
            for dir in 0..6 {
                let Some(nb) = field.neighbour_index(i, j, k, dir) else { continue };
                if field.cells[nb].is_none() {
                    continue;
                }
                let ng = find(&mut parent, nb as u32);
                if ng as usize == r {
                    continue;
                }
                let alpha = field.cells[c].as_ref().unwrap().alpha[dir];
                if alpha <= 0.0 {
                    continue;
                }
                *area_by_group.entry(ng).or_insert(0.0) += alpha * field.full_face_area(i, j, k, dir);
            }
        }

        let survivor = area_by_group
            .iter()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(&g, _)| g);

        let Some(survivor) = survivor else {
            let theta = volume[r] / v_full[r];
            let (i, j, k) = field.decompose(r);
            return Err(Error::Mesh(format!(
                "cutcell: cell ({i},{j},{k}) has theta_c = {theta:.4} < theta_min = \
                 {theta_min} and no fluid neighbour to merge into - the geometry is \
                 an isolated sliver too thin for this grid; refine the mesh near it"
            )));
        };

        let (vr, cr) = (volume[r], centroid[r]);
        let (vs, cs) = (volume[survivor as usize], centroid[survivor as usize]);
        let vt = vr + vs;
        centroid[survivor as usize] = if vt > 0.0 { (cs * vs + cr * vr) / vt } else { cs };
        volume[survivor as usize] = vt;
        parent[r] = survivor;
        n_merged += 1;
    }

    let mut root = vec![0u32; n];
    for c in 0..n {
        if field.cells[c].is_some() {
            root[c] = find(&mut parent, c as u32);
        }
    }

    Ok(MergeResult { root, volume, centroid, n_merged })
}

// ==========================================================================
//  Tests
// ==========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::surface::SoupTri;

    const EPS: Scalar = 1.0e-9;

    fn uniform(lo: Scalar, hi: Scalar, n: usize) -> Vec<Scalar> {
        (0..=n).map(|i| lo + (hi - lo) * i as Scalar / n as Scalar).collect()
    }

    /// An axis-aligned closed cuboid, outward wound.
    fn cuboid_surface(lo: Vec3, hi: Vec3) -> Surface {
        let p = [
            Vec3::new(lo.x, lo.y, lo.z),
            Vec3::new(hi.x, lo.y, lo.z),
            Vec3::new(hi.x, hi.y, lo.z),
            Vec3::new(lo.x, hi.y, lo.z),
            Vec3::new(lo.x, lo.y, hi.z),
            Vec3::new(hi.x, lo.y, hi.z),
            Vec3::new(hi.x, hi.y, hi.z),
            Vec3::new(lo.x, hi.y, hi.z),
        ];
        const T: [[usize; 3]; 12] = [
            [0, 3, 2], [0, 2, 1],
            [4, 5, 6], [4, 6, 7],
            [0, 4, 7], [0, 7, 3],
            [1, 2, 6], [1, 6, 5],
            [0, 1, 5], [0, 5, 4],
            [3, 7, 6], [3, 6, 2],
        ];
        let soup: Vec<SoupTri> = T.iter().map(|&[a, b, c]| (0u32, [p[a], p[b], p[c]])).collect();
        Surface::from_soup(soup, vec!["box".into()]).expect("cuboid surface")
    }

    /// A closed UV sphere, matching `blockgen`'s test fixture in shape.
    fn sphere_surface(centre: Vec3, r: Scalar, n_theta: usize, n_phi: usize) -> Surface {
        let pi = std::f64::consts::PI as Scalar;
        let rings: Vec<Vec<Vec3>> = (1..n_theta)
            .map(|t| {
                let th = pi * t as Scalar / n_theta as Scalar;
                (0..n_phi)
                    .map(|p| {
                        let ph = 2.0 * pi * p as Scalar / n_phi as Scalar;
                        centre
                            + Vec3::new(
                                th.sin() * ph.cos(),
                                th.sin() * ph.sin(),
                                th.cos(),
                            ) * r
                    })
                    .collect()
            })
            .collect();
        let north = centre + Vec3::new(0.0, 0.0, r);
        let south = centre + Vec3::new(0.0, 0.0, -r);

        let mut soup: Vec<SoupTri> = Vec::new();
        for p in 0..n_phi {
            let p1 = (p + 1) % n_phi;
            soup.push((0, [north, rings[0][p], rings[0][p1]]));
        }
        for t in 0..rings.len() - 1 {
            for p in 0..n_phi {
                let p1 = (p + 1) % n_phi;
                soup.push((0, [rings[t][p], rings[t + 1][p], rings[t + 1][p1]]));
                soup.push((0, [rings[t][p], rings[t + 1][p1], rings[t][p1]]));
            }
        }
        let last = rings.len() - 1;
        for p in 0..n_phi {
            let p1 = (p + 1) % n_phi;
            soup.push((0, [south, rings[last][p1], rings[last][p]]));
        }
        Surface::from_soup(soup, vec!["sphere".into()]).expect("sphere surface")
    }

    fn unit_axes(n: usize) -> (Vec<Scalar>, Vec<Scalar>, Vec<Scalar>) {
        let xn = uniform(0.0, 1.0, n);
        (xn.clone(), xn.clone(), xn)
    }

    /// 24.6 row 1 (closure): `cut_sf + sum(alpha_f * Sf_full)` must vanish to
    /// round-off on EVERY cut cell, by construction, whatever the fractions
    /// came out to be.
    #[test]
    fn cut_face_closes_every_cell_to_round_off() {
        let (xn, yn, zn) = unit_axes(40);
        let axes = BlockAxes { xn: &xn, yn: &yn, zn: &zn };
        let s = sphere_surface(Vec3::new(0.5, 0.5, 0.5), 0.3, 48, 96);

        let field = classify_cutcells(&axes, &s, DEFAULT_SUPERSAMPLE).expect("classify");
        assert!(field.n_cut > 0, "the sphere must actually cut some cells");

        let mut max_err: Scalar = 0.0;
        for c in field.cells.iter().flatten() {
            if c.state != CellState::Cut {
                continue;
            }
            let mut sum = c.cut_sf;
            let full = [
                Vec3::new(-1.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(0.0, -1.0, 0.0), Vec3::new(0.0, 1.0, 0.0),
                Vec3::new(0.0, 0.0, -1.0), Vec3::new(0.0, 0.0, 1.0),
            ];
            // Reconstruct each axis face's own full area from theta's cell
            // size (uniform grid here, h = 1/40) times alpha.
            let h = 1.0 / 40.0;
            for d in 0..6 {
                sum += full[d] * (c.alpha[d] * h * h);
            }
            max_err = max_err.max(sum.mag());
        }
        println!("cutcell closure: max |sum Sf| over {} cut cells = {max_err:e}", field.n_cut);
        assert!(max_err < 1e-10, "closure error {max_err} exceeds round-off");
    }

    /// 24.6 row 2: a grid-aligned cuboid must reproduce castellation exactly
    /// - every non-solid cell is `theta in {0,1}`, none `Cut`.
    #[test]
    fn grid_aligned_cuboid_reproduces_castellation() {
        // [0.25,0.75]^3 in a 20-cell unit box: faces land exactly on nodes.
        let (xn, yn, zn) = unit_axes(20);
        let axes = BlockAxes { xn: &xn, yn: &yn, zn: &zn };
        let surf = cuboid_surface(Vec3::new(0.25, 0.25, 0.25), Vec3::new(0.75, 0.75, 0.75));

        let field = classify_cutcells(&axes, &surf, DEFAULT_SUPERSAMPLE).expect("classify");
        assert_eq!(field.n_cut, 0, "a grid-aligned cuboid must cut nothing");
        assert_eq!(field.n_solid, 1000, "10^3 analytic solid cells");
        assert_eq!(field.n_fluid, 7000, "8000 - 1000 analytic fluid cells");

        for c in field.cells.iter().flatten() {
            assert!(
                (c.theta - 1.0).abs() < EPS,
                "a non-cut cell must be theta = 1, got {}",
                c.theta
            );
        }
    }

    /// 24.6 row 3: a cuboid whose face sits at the MIDPLANE of one row of
    /// cells must give exactly `theta = 0.5` there and nowhere else, with the
    /// total volume exact to the sample resolution.
    #[test]
    fn mid_cell_plane_cuboid_gives_theta_one_half() {
        let n = 20usize;
        let (xn, yn, zn) = unit_axes(n);
        let axes = BlockAxes { xn: &xn, yn: &yn, zn: &zn };

        // Solid for x >= 0.525 (the midplane of cell i=10, which spans
        // [0.5, 0.55)); y and z faces land on nodes so only the x-row at
        // i = 10 is cut.
        let surf = cuboid_surface(Vec3::new(0.525, -1.0, -1.0), Vec3::new(2.0, 2.0, 2.0));

        let field = classify_cutcells(&axes, &surf, DEFAULT_SUPERSAMPLE).expect("classify");

        let mut cut_rows: std::collections::BTreeSet<usize> = Default::default();
        let h = 1.0 / n as Scalar;
        let mut total_theta_vol: Scalar = 0.0;
        for c in field.cells.iter().flatten() {
            total_theta_vol += c.volume;
        }
        for c in 0..field.cells.len() {
            let Some(cf) = &field.cells[c] else { continue };
            let (i, _, _) = field.decompose(c);
            if cf.state == CellState::Cut {
                cut_rows.insert(i);
                assert!(
                    (cf.theta - 0.5).abs() < 1.0 / DEFAULT_SUPERSAMPLE as Scalar,
                    "row {i}: theta {} should be 0.5 to the sample resolution",
                    cf.theta
                );
            }
        }
        assert_eq!(cut_rows, std::collections::BTreeSet::from([10]), "only row 10 is cut");

        // Analytic fluid volume: everything x < 0.525.
        let expect = 0.525 * 1.0 * 1.0;
        assert!(
            (total_theta_vol - expect).abs() < h * h * h,
            "total fluid volume {total_theta_vol} vs analytic {expect}"
        );
    }

    /// 24.6 row 4: the cut-cell sphere's volume error must be MUCH smaller
    /// than castellation's (both reported).
    #[test]
    fn sphere_volume_error_is_much_smaller_than_castellation() {
        let pi = std::f64::consts::PI as Scalar;
        let n = 40usize;
        let (xn, yn, zn) = unit_axes(n);
        let axes = BlockAxes { xn: &xn, yn: &yn, zn: &zn };
        let r: Scalar = 0.3;
        let surf = sphere_surface(Vec3::new(0.5, 0.5, 0.5), r, 48, 96);

        let field = classify_cutcells(&axes, &surf, DEFAULT_SUPERSAMPLE).expect("classify");
        let expect = 1.0 - 4.0 / 3.0 * pi * r * r * r;

        let mut cut_vol: Scalar = 0.0;
        let mut castellated_vol: Scalar = 0.0;
        let h = 1.0 / n as Scalar;
        for c in field.cells.iter().flatten() {
            cut_vol += c.volume;
            castellated_vol += h * h * h; // every non-solid cell counts fully
        }

        let err_cut = (cut_vol - expect).abs();
        let err_castellated = (castellated_vol - expect).abs();
        println!(
            "sphere volume: cut-cell {cut_vol} (err {err_cut:e}), castellated \
             {castellated_vol} (err {err_castellated:e}), analytic {expect}"
        );
        assert!(
            err_cut < 0.2 * err_castellated,
            "cut-cell error {err_cut} is not much smaller than castellation's {err_castellated}"
        );
    }

    /// 24.6 row 5: after merging, no surviving cell is below `theta_min`.
    #[test]
    fn merging_leaves_no_cell_below_theta_min() {
        let n = 30usize;
        let (xn, yn, zn) = unit_axes(n);
        let axes = BlockAxes { xn: &xn, yn: &yn, zn: &zn };
        let r: Scalar = 0.3;
        // An off-grid centre so slivers of every size occur near the shell.
        let surf = sphere_surface(Vec3::new(0.501_3, 0.499_7, 0.502_1), r, 32, 64);

        let field = classify_cutcells(&axes, &surf, DEFAULT_SUPERSAMPLE).expect("classify");
        let theta_min = DEFAULT_THETA_MIN;

        let n_below_before = field
            .cells
            .iter()
            .flatten()
            .filter(|c| c.theta > 0.0 && c.theta < theta_min)
            .count();
        assert!(n_below_before > 0, "the test needs at least one sliver to merge");

        let merged = merge_small_cells(&field, theta_min).expect("merge");
        println!(
            "merge: {} slivers below theta_min = {theta_min} merged away ({} were below \
             threshold before merging)",
            merged.n_merged, n_below_before
        );
        assert!(merged.n_merged > 0);

        for c in 0..field.cells.len() {
            if field.cells[c].is_none() {
                continue;
            }
            let r = merged.root[c] as usize;
            if r != c {
                continue; // not a surviving root
            }
            let theta = merged.volume[r] / field.cells[r].as_ref().unwrap().v_full;
            assert!(
                theta >= theta_min - 1e-12,
                "surviving cell {r} has theta {theta} < theta_min {theta_min} after merging"
            );
        }
    }

    /// A cell state must be exactly one of the three - the fixture surfaces
    /// above only exercise Fluid/Solid/Cut through geometry, this pins the
    /// enum's derived equality down directly.
    #[test]
    fn cell_state_equality() {
        assert_eq!(CellState::Fluid, CellState::Fluid);
        assert_ne!(CellState::Fluid, CellState::Solid);
        assert_ne!(CellState::Cut, CellState::Solid);
    }
}
