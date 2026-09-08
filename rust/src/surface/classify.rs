// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

//! Inside/outside classification of a structured block against a surface.
//!
//! Written from:
//!   ofgpu `SPEC-LIT.md` §23.3 (column-parity ray casting, jittered-ray
//!     retry, 3-axis majority vote, winding-number arbitration);
//!   Aftosmis, Berger & Melton, *AIAA J.* 36(6) (1998) 952 - the
//!     castellation stage this feeds;
//!   Barill, Dickson, Schmidt, Levin & Jacobson, *ACM TOG* 37(4) (2018) -
//!     the generalized winding number used as the arbiter, evaluated per
//!     triangle by the exact solid-angle formula their §2 builds on.
//! No GPL-licensed source was consulted.
//!
//! The pipeline, in the order §23.3 states it:
//!
//! 1. **Column parity.** For each grid column (fixed pair of tangential cell
//!    centres) cast a line along the axis, collect the watertight crossings
//!    ([`super::TriIndex::crossings_x`]), sort, and classify every cell
//!    centre in the column by crossing parity.
//! 2. **Jittered retry.** A second ray, offset inside the same column of
//!    cells by an irrational fraction of the cell size, detects the
//!    ambiguous hits (rays through vertices, edges, or coplanar features):
//!    where the two parities disagree, a third ray with a different
//!    irrational offset breaks the tie. *DESIGN*: if the two jittered rays
//!    agree the base ray was the fluke and their answer is taken as firm;
//!    if they disagree the column's answer is kept but marked unsure.
//! 3. **3-axis majority vote.** Steps 1-2 run along x, y and z. A cell whose
//!    three column classifications agree is done. A 2-1 split is settled by
//!    the majority when both majority columns are firm.
//! 4. **Winding number.** The remaining cells - a 2-1 split resting on an
//!    unsure column - are arbitrated by the exact solid-angle winding
//!    number, O(tris) per cell. They are counted, because §23.3 accepts
//!    that cost only because the cells are rare; a surface that arbitrates
//!    half the domain deserves a visible number saying so.
//!
//! Cells in columns the surface never crosses get identical parities from
//! every ray, so they classify in step 1 at no extra cost - no retry ray is
//! even consulted per cell, and neither the vote nor the winding number
//! does any work for them.

use crate::error::{Error, Result};
use crate::{Scalar, Vec3};

use super::{SoupTri, Surface, TriIndex};

// ==========================================================================
//  Inputs and outputs
// ==========================================================================

/// The node coordinates of the three block axes, `n + 1` entries each,
/// strictly increasing - exactly what `blockgen::graded_nodes` produces.
/// Cell centres are the interval midpoints; on a rectilinear block that is
/// the exact centroid.
#[derive(Debug, Clone, Copy)]
pub struct BlockAxes<'a> {
    pub xn: &'a [Scalar],
    pub yn: &'a [Scalar],
    pub zn: &'a [Scalar],
}

/// Per-cell solid flags, i fastest (`cell = i + nx*(j + ny*k)`, the same
/// index rule as `blockgen`), plus the classification statistics §23.3 asks
/// to be reported.
#[derive(Debug, Clone)]
pub struct SolidMask {
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    /// `true` = inside the surface = removed from the fluid mesh.
    pub solid: Vec<bool>,
    pub n_solid: usize,
    /// Cells settled by the 3-axis majority vote (columns disagreed).
    pub voted: usize,
    /// Cells the vote could not settle, decided by the exact solid-angle
    /// winding number.
    pub arbitrated: usize,
}

impl SolidMask {
    pub fn n_cells(&self) -> usize {
        self.solid.len()
    }

    pub fn n_fluid(&self) -> usize {
        self.solid.len() - self.n_solid
    }

    #[inline]
    pub fn is_solid(&self, i: usize, j: usize, k: usize) -> bool {
        self.solid[i + self.nx * (j + self.ny * k)]
    }
}

// ==========================================================================
//  Classification
// ==========================================================================

/// Classify every cell of the block as solid (inside `surf`) or fluid, per
/// §23.3. See the module doc for the pipeline.
///
/// The surface's orientation is deliberately irrelevant: parity does not
/// read normals, and the winding-number arbiter takes `|w|`, so a surface
/// wound inward classifies identically to one wound outward.
///
/// The per-cell combine pass is parallel: contiguous chunks of `c` on
/// `std::thread::scope` threads when the block is big enough to pay for them
/// (`PARALLEL_MIN_CELLS`), one serial chunk otherwise. Both shapes run
/// `combine_cell_range`, so the mask is bit-identical either way.
pub fn classify(axes: &BlockAxes, surf: &Surface) -> Result<SolidMask> {
    let nodes: [&[Scalar]; 3] = [axes.xn, axes.yn, axes.zn];
    let mut n = [0usize; 3];
    for (ax, nd) in nodes.iter().enumerate() {
        if nd.len() < 2 {
            return Err(Error::Mesh(format!(
                "classify: axis {ax} has {} node(s); a block axis needs at least 2",
                nd.len()
            )));
        }
        n[ax] = nd.len() - 1;
    }
    let n_cells = n[0]
        .checked_mul(n[1])
        .and_then(|t| t.checked_mul(n[2]))
        .ok_or_else(|| Error::Mesh("classify: cell count overflows".to_string()))?;

    // Bucket size ~ the mesh spacing (§23.4): the mean cell edge over the
    // three axes. The grid is an accelerator, so the mean is good enough
    // even on a graded block.
    let mut hint: Scalar = 0.0;
    for ax in 0..3 {
        let nd = nodes[ax];
        hint += (nd[nd.len() - 1] - nd[0]).abs() / n[ax] as Scalar;
    }
    hint = (hint / 3.0).max(Scalar::MIN_POSITIVE);

    // Bit `ax` of `vote[c]` is the axis-`ax` column's answer (set = solid);
    // bit `ax` of `sure[c]` says whether that answer was firm.
    let mut vote = vec![0u8; n_cells];
    let mut sure = vec![0u8; n_cells];

    for ax in 0..3 {
        cast_axis(surf, ax, n, nodes, hint, &mut vote, &mut sure)?;
    }

    // ---- combine ----------------------------------------------------------
    let mut solid = vec![false; n_cells];
    let mut n_solid = 0usize;
    let mut voted = 0usize;
    let mut arbitrated = 0usize;

    // The pass is embarrassingly parallel: each cell reads only its own
    // `vote[c]`/`sure[c]` - fixed once the three `cast_axis` passes are done -
    // and the immutable surface, and writes exactly its own `solid[c]`, so it
    // splits into contiguous chunks of `c`, one per thread, under
    // `std::thread::scope` (not rayon: the READMEs publish this crate's
    // dependency list and it must stay true). Both shapes below run the same
    // [`combine_cell_range`] on the same per-cell inputs and the per-chunk
    // counters are summed in chunk order, so the mask and the three counts
    // come out exactly what the old serial loop produced. The threads are
    // for the arbiter tail: an arbitrated cell is O(tris) (module doc), and
    // one slow cell must not serialise the rest.
    let n_threads = std::thread::available_parallelism().map_or(1, |p| p.get()).min(n_cells);
    if n_threads <= 1 || n_cells < PARALLEL_MIN_CELLS {
        let (solid_n, voted_n, arbitrated_n) =
            combine_cell_range(&vote, &sure, &mut solid, surf, n, nodes, 0);
        n_solid += solid_n;
        voted += voted_n;
        arbitrated += arbitrated_n;
    } else {
        // `div_ceil` keeps the chunk count at or under `n_threads` - one
        // non-empty chunk per spawned thread, none left idle, none doubled up.
        let chunk_len = n_cells.div_ceil(n_threads);
        // The shared inputs go to the workers as copied references: naming
        // `&vote` inside the `move` closure below would capture `vote` itself
        // BY MOVE, and the outer `map` closure is `FnMut`, so the second
        // worker would see a moved-out vector. The references are bound here,
        // where Copy makes each worker carry its own copy of the borrow. Only
        // `slab` is exclusive, and each worker's chunk is a disjoint
        // `chunks_mut` piece of `solid`.
        let vote_ref = &vote;
        let sure_ref = &sure;
        let per_chunk: Vec<(usize, usize, usize)> = std::thread::scope(|scope| {
            let handles: Vec<_> = solid
                .chunks_mut(chunk_len)
                .enumerate()
                .map(|(chunk, slab)| {
                    let c0 = chunk * chunk_len;
                    scope
                        .spawn(move || combine_cell_range(vote_ref, sure_ref, slab, surf, n, nodes, c0))
                })
                .collect();
            // The per-cell body is infallible - `combine_cell_range` returns a
            // counter triple, not a `Result` - so a worker can only die by
            // panicking; `join` re-raises that panic here, exactly what the
            // serial loop would have done in place.
            handles
                .into_iter()
                .map(|h| h.join().expect("solid-mask combine worker panicked"))
                .collect()
        });
        for (solid_n, voted_n, arbitrated_n) in per_chunk {
            n_solid += solid_n;
            voted += voted_n;
            arbitrated += arbitrated_n;
        }
    }

    Ok(SolidMask { nx: n[0], ny: n[1], nz: n[2], solid, n_solid, voted, arbitrated })
}

/// Whether the 2-1 split in `vote` rests on an unsure majority column and
/// must go to the winding number.
#[inline]
fn needs_arbiter(vote: u8, sure: u8) -> bool {
    debug_assert!(vote != 0 && vote != 0b111);
    let majority_solid = vote.count_ones() >= 2;
    let majority_mask = if majority_solid { vote } else { !vote & 0b111 };
    (majority_mask & sure) != majority_mask
}

/// Below this many cells the whole pass runs serially. The common cell costs
/// two byte loads, a match and a store - only an arbitrated cell reaches the
/// O(tris) winding number, and §23.3 keeps those rare - so this many cells is
/// microseconds of combine work, under what spawning and joining the threads
/// costs unless an arbiter tail lands in the block. That per-cell cost is
/// well under the cut-cell pass's (a `nearest_triangle` query every cell,
/// `cutcell::PARALLEL_MIN_CELLS` at 1 << 12), so the crossover sits higher
/// here. *DESIGN*.
const PARALLEL_MIN_CELLS: usize = 1 << 16;

/// One contiguous chunk of the combine pass: block cells `c0 .. c0 +
/// solid.len()` in the same `c` order the serial loop used, written into
/// `solid` in place. Returns the chunk's own `(n_solid, voted, arbitrated)`.
///
/// This is deliberately the ONLY copy of the per-cell logic - both the serial
/// and the parallel entry in [`classify`] call it - so the mask is
/// bit-identical to the serial one by construction, not by coincidence: every
/// cell is computed by the same instructions on the same inputs (`vote`,
/// `sure` and `surf` are read-only for the whole pass, and the winding number
/// is the same pure function of `(surf, cell_centre)` per cell), each `solid` slot
/// is written by exactly one worker, and the three counters are plain integer
/// totals summed in chunk order.
fn combine_cell_range(
    vote: &[u8],
    sure: &[u8],
    solid: &mut [bool],
    surf: &Surface,
    n: [usize; 3],
    nodes: [&[Scalar]; 3],
    c0: usize,
) -> (usize, usize, usize) {
    let (mut n_solid, mut voted, mut arbitrated) = (0usize, 0usize, 0usize);
    for (local, slot) in solid.iter_mut().enumerate() {
        let c = c0 + local;
        let v = vote[c];
        let s = match v {
            0 => false,
            0b111 => true,
            _ if !needs_arbiter(v, sure[c]) => {
                voted += 1;
                v.count_ones() >= 2
            }
            _ => {
                arbitrated += 1;
                // Cell centres are only needed for the (rare) arbitrated
                // cells, so they are recomputed on demand rather than
                // materialised for the whole block.
                winding_number(surf, cell_centre(n, nodes, c)).abs() >= 0.5
            }
        };
        if s {
            n_solid += 1;
        }
        *slot = s;
    }
    (n_solid, voted, arbitrated)
}

/// The centre of flat cell id `c`: the interval midpoints of its three axes.
fn cell_centre(n: [usize; 3], nodes: [&[Scalar]; 3], c: usize) -> Vec3 {
    let i = c % n[0];
    let t = c / n[0];
    let (j, k) = (t % n[1], t / n[1]);
    Vec3::new(
        0.5 * (nodes[0][i] + nodes[0][i + 1]),
        0.5 * (nodes[1][j] + nodes[1][j + 1]),
        0.5 * (nodes[2][k] + nodes[2][k + 1]),
    )
}

// ==========================================================================
//  Per-axis column casting
// ==========================================================================

/// Irrational jitter fractions of the local cell size, per §23.3's
/// simulation-of-simplicity: an irrational offset cannot land the retry ray
/// on the same rational grid feature the base ray hit.
///
/// *DESIGN* - the constants are ours, and their SIZE matters: about `1e-3`
/// of a cell. That is many orders of magnitude above rounding noise, so a
/// degenerate hit (ray through a vertex or an edge) is escaped reliably -
/// but small enough that the jittered ray almost never samples genuinely
/// different geometry. A large offset would turn every silhouette-grazing
/// column into a fake ambiguity and send perfectly ordinary near-surface
/// cells to the O(tris) winding-number arbiter.
fn jitter1() -> (Scalar, Scalar) {
    (
        ((2.0 as Scalar).sqrt() - 1.0) / 1024.0,
        ((3.0 as Scalar).sqrt() - 1.0) / 1024.0,
    )
}
fn jitter2() -> (Scalar, Scalar) {
    (
        ((5.0 as Scalar).sqrt() - 2.0) / 512.0,
        (std::f64::consts::PI as Scalar - 3.0) / 512.0,
    )
}

/// Cell-centre coordinates of one axis: the interval midpoints.
fn centres(nodes: &[Scalar]) -> Vec<Scalar> {
    (0..nodes.len() - 1).map(|i| 0.5 * (nodes[i] + nodes[i + 1])).collect()
}

/// Parity fill of one column: `out[i]` = is centre `i` inside, from the
/// sorted crossings. One merge walk - both lists are ascending.
fn parity_fill(hits: &[(Scalar, usize)], centres: &[Scalar], out: &mut [bool]) {
    let mut h = 0usize;
    for (i, &c) in centres.iter().enumerate() {
        while h < hits.len() && hits[h].0 < c {
            h += 1;
        }
        out[i] = (h & 1) == 1;
    }
}

/// The surface with its coordinates cyclically rotated so original axis
/// `ax` becomes x. A cyclic permutation is a proper rotation: windings,
/// areas and closedness are exactly preserved, so `crossings_x` on the
/// rotated copy is the axis-`ax` column cast on the original.
fn rotate_surface(surf: &Surface, ax: usize) -> Result<Surface> {
    let rot = |p: Vec3| -> Vec3 {
        match ax {
            1 => Vec3::new(p.y, p.z, p.x),
            2 => Vec3::new(p.z, p.x, p.y),
            _ => p,
        }
    };
    let soup: Vec<SoupTri> = surf
        .tris
        .iter()
        .enumerate()
        .map(|(t, tri)| {
            (
                surf.tri_patch[t],
                [
                    rot(surf.points[tri[0] as usize]),
                    rot(surf.points[tri[1] as usize]),
                    rot(surf.points[tri[2] as usize]),
                ],
            )
        })
        .collect();
    Surface::from_soup(soup, surf.patch_names.clone())
}

/// Cast every column along axis `ax` and record one vote (and one sureness
/// bit) per cell.
fn cast_axis(
    surf: &Surface,
    ax: usize,
    n: [usize; 3],
    nodes: [&[Scalar]; 3],
    hint: Scalar,
    vote: &mut [u8],
    sure: &mut [u8],
) -> Result<()> {
    let rotated;
    let rs: &Surface = if ax == 0 {
        surf
    } else {
        rotated = rotate_surface(surf, ax)?;
        &rotated
    };
    let idx = TriIndex::new(rs, hint)?;

    let a1 = (ax + 1) % 3;
    let a2 = (ax + 2) % 3;
    let cast_c = centres(nodes[ax]);
    let col1 = centres(nodes[a1]);
    let col2 = centres(nodes[a2]);

    let (j1y, j1z) = jitter1();
    let (j2y, j2z) = jitter2();

    let n0 = n[ax];
    let mut p0 = vec![false; n0];
    let mut p1 = vec![false; n0];
    let mut p2 = vec![false; n0];

    for k in 0..n[a2] {
        for j in 0..n[a1] {
            let (y, z) = (col1[j], col2[k]);
            let hy = nodes[a1][j + 1] - nodes[a1][j];
            let hz = nodes[a2][k + 1] - nodes[a2][k];

            parity_fill(&idx.crossings_x(y, z), &cast_c, &mut p0);
            parity_fill(&idx.crossings_x(y + j1y * hy, z + j1z * hz), &cast_c, &mut p1);

            let mut have_retry = false;
            for i in 0..n0 {
                let (v, firm) = if p0[i] == p1[i] {
                    (p0[i], true)
                } else {
                    if !have_retry {
                        parity_fill(
                            &idx.crossings_x(y + j2y * hy, z + j2z * hz),
                            &cast_c,
                            &mut p2,
                        );
                        have_retry = true;
                    }
                    if p2[i] == p1[i] {
                        // The two jittered rays agree against the base ray:
                        // the base ray grazed a feature. Firm.
                        (p1[i], true)
                    } else {
                        // Base and retry against the first jitter: the
                        // column itself is unstable here.
                        (p0[i], false)
                    }
                };

                let mut ii = [0usize; 3];
                ii[ax] = i;
                ii[a1] = j;
                ii[a2] = k;
                let c = ii[0] + n[0] * (ii[1] + n[1] * ii[2]);
                if v {
                    vote[c] |= 1 << ax;
                }
                if firm {
                    sure[c] |= 1 << ax;
                }
            }
        }
    }

    Ok(())
}

// ==========================================================================
//  Winding-number arbiter
// ==========================================================================

/// Generalized winding number of `p` with respect to the surface: the summed
/// signed solid angle of every triangle over `4*pi` (Barill et al. 2018,
/// eq. 5; the per-triangle solid angle is the classical exact
/// `2*atan2` form their §2 quotes).
///
/// Exactly `+-1` inside a closed consistently wound surface, `0` outside,
/// and fractional through holes - which is what makes it the right arbiter
/// for the cells parity could not settle.
pub(crate) fn winding_number(surf: &Surface, p: Vec3) -> Scalar {
    let mut sum: Scalar = 0.0;
    for tri in &surf.tris {
        let a = surf.points[tri[0] as usize] - p;
        let b = surf.points[tri[1] as usize] - p;
        let c = surf.points[tri[2] as usize] - p;
        let (la, lb, lc) = (a.mag(), b.mag(), c.mag());

        let num = a.dot(b.cross(c));
        let den = la * lb * lc + a.dot(b) * lc + b.dot(c) * la + c.dot(a) * lb;
        // atan2 handles every quadrant, including den <= 0 (the triangle
        // subtending more than a hemisphere), where a plain atan would be
        // off by pi.
        sum += 2.0 * num.atan2(den);
    }
    sum / (4.0 * std::f64::consts::PI as Scalar)
}

// ==========================================================================
//  Tests
// ==========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Uniform nodes on [lo, hi] with n cells.
    fn uniform(lo: Scalar, hi: Scalar, n: usize) -> Vec<Scalar> {
        (0..=n).map(|i| lo + (hi - lo) * i as Scalar / n as Scalar).collect()
    }

    /// The unit cube [0,1]^3, outward wound, from the fixtures in the parent
    /// module's tests.
    fn unit_cube() -> Surface {
        let p = super::super::tests::cube_points();
        let soup: Vec<SoupTri> = super::super::tests::CUBE_TRIS
            .iter()
            .map(|&[a, b, c]| (0u32, [p[a], p[b], p[c]]))
            .collect();
        match Surface::from_soup(soup, vec!["cube".into()]) {
            Ok(s) => s,
            Err(e) => panic!("cube build failed: {e}"),
        }
    }

    #[test]
    fn winding_number_is_one_inside_and_zero_outside() {
        let s = unit_cube();
        let w_in = winding_number(&s, Vec3::new(0.5, 0.5, 0.5));
        let w_out = winding_number(&s, Vec3::new(1.7, 0.3, 0.4));
        assert!((w_in - 1.0).abs() < 1e-9, "inside w = {w_in}");
        assert!(w_out.abs() < 1e-9, "outside w = {w_out}");
    }

    /// Grid-aligned cube in a 20^3 block over [-0.5, 1.5]^3: the cell
    /// centres never touch the surface, so parity alone must classify every
    /// cell - the analytic 10^3 solid count, with no vote and no arbitration.
    #[test]
    fn parity_alone_classifies_an_untouched_cube_exactly() {
        let s = unit_cube();
        let xn = uniform(-0.5, 1.5, 20);
        let axes = BlockAxes { xn: &xn, yn: &xn, zn: &xn };

        let m = match classify(&axes, &s) {
            Ok(m) => m,
            Err(e) => panic!("classify failed: {e}"),
        };

        assert_eq!(m.n_cells(), 8000);
        assert_eq!(m.n_solid, 1000, "10 cells per axis are inside");
        assert_eq!(m.voted, 0, "no column may disagree on this geometry");
        assert_eq!(m.arbitrated, 0, "zero arbitration cost, per the contract");

        // Spot checks: dead centre solid, a corner cell fluid.
        assert!(m.is_solid(10, 10, 10));
        assert!(!m.is_solid(0, 0, 0));
        assert!(!m.is_solid(19, 19, 19));
    }

    /// Orientation must not matter: the same cube with every triangle
    /// flipped classifies identically (parity reads no normals, the arbiter
    /// takes |w|).
    #[test]
    fn inverted_winding_classifies_identically() {
        let p = super::super::tests::cube_points();
        let soup: Vec<SoupTri> = super::super::tests::CUBE_TRIS
            .iter()
            .map(|&[a, b, c]| (0u32, [p[a], p[c], p[b]]))
            .collect();
        let s = match Surface::from_soup(soup, vec!["cube".into()]) {
            Ok(s) => s,
            Err(e) => panic!("build failed: {e}"),
        };

        let xn = uniform(-0.5, 1.5, 20);
        let axes = BlockAxes { xn: &xn, yn: &xn, zn: &xn };
        let m = match classify(&axes, &s) {
            Ok(m) => m,
            Err(e) => panic!("classify failed: {e}"),
        };
        assert_eq!(m.n_solid, 1000);
        assert_eq!(m.arbitrated, 0);
    }
}
