// meteor-cfd - Copyright (c) 2026 주식회사 메테오시뮬레이션 (Meteo Simulation Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

//! Structured hex ("one block") mesh generator, and the ready-to-run cases
//! built on top of it.
//!
//! This exists so the project has real test meshes without a working OpenFOAM
//! installation to run `blockMesh`. The output is byte-for-byte readable by
//! OpenFOAM-12: same `FoamFile` headers, same upper-triangular face ordering,
//! same face winding.
//!
//! Guarantees the rest of ofgpu relies on, and which the tests at the bottom
//! pin down because a violation of any of them is invisible in the files
//! themselves:
//!
//! * `cell(i, j, k) = i + nx*(j + ny*k)` - i fastest;
//! * `point(i, j, k) = i + (nx+1)*(j + (ny+1)*k)`;
//! * internal faces sorted by owner, then neighbour, with `owner < neighbour`
//!   for every one of them - `lduAddressing` order;
//! * the area vector of an internal face points from owner to neighbour, the
//!   area vector of a boundary face points out of the domain;
//! * boundary faces follow all internal faces and every patch occupies a
//!   contiguous `[startFace, startFace + nFaces)` range - including when one
//!   of the six sides is split in two by a [`PatchWindow`], which is what puts
//!   a burner inlet in the middle of the plume case's floor.
//!
//! Provenance: carried across from this project's own earlier C++ mesh
//! generator when the crate moved to Rust. It writes a file format; it
//! implements no discretisation. No GPL-licensed source was consulted.
//!
//! Two things are done deliberately:
//!
//! 1. Internal faces are stored as `(owner, neighbour, direction)`, **not** as
//!    four point labels. The winding is regenerated from the owner cell at
//!    write time, which makes "the area vector points from owner to neighbour"
//!    true by construction rather than by bookkeeping: whatever the sort does
//!    to the face order, the four points still come from the owner's
//!    `+x`/`+y`/`+z` face. A geometric check re-verifies every face anyway.
//!
//! 2. Integers are rendered by hand into a megabyte buffer rather than through
//!    the formatting machinery one token at a time. A 4 M cell block has ~12 M
//!    internal faces, and the difference there is minutes against seconds.

use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::Write as _;
use std::path::{Path, PathBuf};

use crate::error::{Error, IoContext, Result};
use crate::io::fields::{
    write_scalar_field, write_vector_field, PatchFieldSpec, RawScalarField, RawVectorField,
};
use crate::mesh::PatchKind;
use crate::{Scalar, Vec3};

// ==========================================================================
//  Public specification types
// ==========================================================================

/// One axis of the block: `[lo, hi]` cut into `n` cells, optionally graded.
#[derive(Debug, Clone, PartialEq)]
pub struct GradedAxis {
    pub lo: Scalar,
    pub hi: Scalar,
    pub n: usize,
    /// Expansion ratio, last cell / first cell (1 = uniform).
    ///
    /// With `two_sided` it is the largest cell (at the centre) divided by the
    /// smallest (at either wall), which for an even `n` is the same thing as
    /// grading each half by this ratio.
    pub expansion: Scalar,
    /// `true`: grade symmetrically toward BOTH ends (channel walls).
    pub two_sided: bool,
}

impl Default for GradedAxis {
    fn default() -> Self {
        Self { lo: 0.0, hi: 1.0, n: 10, expansion: 1.0, two_sided: false }
    }
}

/// The block itself plus the six patches that bound it.
///
/// Patch slots are ordered `-x +x -y +y -z +z`. That is also the order the
/// patches appear in `constant/polyMesh/boundary`, and therefore the order
/// their faces occupy in the flattened boundary-face arrays every kernel
/// indexes with.
#[derive(Debug, Clone, PartialEq)]
pub struct BlockSpec {
    pub x: GradedAxis,
    pub y: GradedAxis,
    pub z: GradedAxis,
    pub patch_name: [String; 6],
    pub patch_type: [String; 6],
    /// Optionally split one slot in two; see [`PatchWindow`].
    pub window: Option<PatchWindow>,
}

/// A rectangular window of one slot's faces, carved out into a patch of its
/// own - the plume case's burner inlet sitting in the middle of its floor.
///
/// The two indices are the slot's own tangential cell indices, in the order
/// `boundary_quad` decomposes a slot index: `(j, k)` on `-x`/`+x`, `(i, k)` on
/// `-y`/`+y`, `(i, j)` on `-z`/`+z`. Both ranges are half-open, `[lo, hi)`.
///
/// The faces the window does not take keep the slot's own name and type, so a
/// window turns one patch into two. They are written window first, then the
/// remainder, because OpenFOAM stores a patch as nothing but a `startFace` and
/// an `nFaces`: a patch whose faces are interleaved with another's cannot be
/// expressed at all, and a boundary file that claims otherwise is read back as
/// a different mesh with no error anywhere.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PatchWindow {
    /// Which of the six slots is split, in `-x +x -y +y -z +z` order.
    pub slot: usize,
    /// First cell index of the window, `[fast, slow]`.
    pub lo: [usize; 2],
    /// One past the last cell index, `[fast, slow]`.
    pub hi: [usize; 2],
    /// Name and OpenFOAM type of the patch made from the window itself.
    pub name: String,
    pub type_name: String,
}

impl Default for BlockSpec {
    fn default() -> Self {
        Self {
            x: GradedAxis::default(),
            y: GradedAxis::default(),
            z: GradedAxis::default(),
            patch_name: ["xMin", "xMax", "yMin", "yMax", "zMin", "zMax"].map(String::from),
            patch_type: ["patch", "patch", "wall", "wall", "empty", "empty"].map(String::from),
            window: None,
        }
    }
}

// ==========================================================================
//  Grading
// ==========================================================================

/// `n + 1` nodes of a geometric progression on `[lo, hi]` whose last cell
/// divided by its first cell is `r_ratio`:
///
/// ```text
/// r = R^(1/(n-1)),    x_i = lo + L*(r^i - 1)/(r^n - 1)
/// ```
///
/// The endpoints are *set* afterwards rather than accumulated, so no round-off
/// drift can leave the block slightly the wrong size.
fn fill_graded(v: &mut [Scalar], lo: Scalar, hi: Scalar, n: usize, r_ratio: Scalar) {
    let l = hi - lo;

    // `!(r > 0)` rather than `r <= 0` so a NaN ratio falls back to uniform
    // instead of producing a mesh full of NaN coordinates.
    if n <= 1 || !(r_ratio > 0.0) || (r_ratio - 1.0).abs() < 1e-10 {
        for i in 0..=n {
            v[i] = lo + l * (i as Scalar) / (n as Scalar);
        }
    } else {
        let r = r_ratio.powf(1.0 / ((n - 1) as Scalar));
        let den = r.powf(n as Scalar) - 1.0;

        for i in 0..=n {
            v[i] = lo + l * (r.powf(i as Scalar) - 1.0) / den;
        }
    }

    v[0] = lo;
    v[n] = hi;
}

/// Node coordinates of one graded axis: `n + 1` values with `[0] == lo` and
/// `[n] == hi` exactly.
///
/// A degenerate axis (`n == 0`) yields the single node `lo`. The writers reject
/// it up front, so this only has to avoid panicking on the way there.
pub fn graded_nodes(a: &GradedAxis) -> Vec<Scalar> {
    if a.n == 0 {
        return vec![a.lo];
    }

    let n = a.n;
    let mut v = vec![0.0 as Scalar; n + 1];

    let half = n / 2;
    // Geometric steps between the wall cell and the centre cell.
    let k = (n - 1) / 2;

    if a.two_sided && k >= 1 && a.expansion > 0.0 && (a.expansion - 1.0).abs() > 1e-10 {
        // Symmetric grading. Cell i gets the weight r^min(i, n-1-i), so the
        // sizes grow geometrically from BOTH walls toward the centre and the
        // ratio of the largest cell to the smallest is exactly `expansion`.
        // For even n this is identical to grading the lower half with ratio
        // `expansion` and mirroring it, which is what the two-sided option
        // means; unlike a literal two-halves split it also stays symmetric for
        // odd n, where the extra cell straddles the centre instead of landing
        // lopsidedly in one half.
        let r = a.expansion.powf(1.0 / (k as Scalar));

        let mut w = vec![0.0 as Scalar; n];
        let mut sum = 0.0 as Scalar;
        for i in 0..n {
            let lev = if i < n - 1 - i { i } else { n - 1 - i };
            w[i] = r.powf(lev as Scalar);
            sum += w[i];
        }

        // Only the lower half is accumulated; the upper half is its exact
        // reflection, so no round-off can break the symmetry of the mesh.
        let l = a.hi - a.lo;
        let mut acc = 0.0 as Scalar;
        v[0] = a.lo;
        for i in 1..=half {
            acc += w[i - 1];
            v[i] = a.lo + l * (acc / sum);
        }
        for i in (half + 1)..=n {
            v[i] = a.lo + a.hi - v[n - i];
        }
        v[n] = a.hi;
    } else {
        // Two-sided with fewer than three cells has nowhere to grade: both
        // halves would be a single cell, and mirroring makes them equal.
        let r = if a.two_sided { 1.0 } else { a.expansion };
        fill_graded(&mut v, a.lo, a.hi, n, r);
    }

    v
}

// ==========================================================================
//  Sizes
// ==========================================================================

/// Every count of the block, validated to fit a 32-bit `Label`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Counts {
    cells: usize,
    points: usize,
    internal: usize,
    boundary: usize,
}

/// Check the requested block size before anything is allocated.
///
/// Counted in `u64` so an overflow is caught rather than wrapped, and refused
/// loudly above the 32-bit label range rather than emitting a mesh whose
/// indices have silently wrapped negative.
fn counts_of(nx: usize, ny: usize, nz: usize) -> Result<Counts> {
    if nx < 1 || ny < 1 || nz < 1 {
        return Err(Error::Mesh(format!(
            "blockgen: every axis needs at least one cell, got {nx} x {ny} x {nz}"
        )));
    }

    let over = || Error::Mesh("blockgen: block size overflows a 64-bit count".to_string());
    let (nx, ny, nz) = (nx as u64, ny as u64, nz as u64);
    let mul = |a: u64, b: u64| a.checked_mul(b).ok_or_else(over);
    let add = |a: u64, b: u64| a.checked_add(b).ok_or_else(over);

    let cells = mul(mul(nx, ny)?, nz)?;
    let points = mul(mul(add(nx, 1)?, add(ny, 1)?)?, add(nz, 1)?)?;
    let internal = add(
        add(mul(mul(nx - 1, ny)?, nz)?, mul(mul(nx, ny - 1)?, nz)?)?,
        mul(mul(nx, ny)?, nz - 1)?,
    )?;
    let boundary = add(
        add(mul(mul(2, ny)?, nz)?, mul(mul(2, nx)?, nz)?)?,
        mul(mul(2, nx)?, ny)?,
    )?;
    let faces = add(internal, boundary)?;

    if points.max(faces) > i32::MAX as u64 {
        return Err(Error::Mesh(format!(
            "blockgen: mesh too large - {points} points / {faces} faces exceeds \
             the 32-bit label range"
        )));
    }

    Ok(Counts {
        cells: cells as usize,
        points: points as usize,
        internal: internal as usize,
        boundary: boundary as usize,
    })
}

// ==========================================================================
//  The structured lattice
// ==========================================================================

/// Node coordinates plus the index arithmetic every other part of this file
/// agrees on.
struct Grid {
    nx: usize,
    ny: usize,
    nz: usize,
    xn: Vec<Scalar>,
    yn: Vec<Scalar>,
    zn: Vec<Scalar>,
}

impl Grid {
    fn new(b: &BlockSpec) -> Result<Self> {
        // Validated before the node arrays are allocated, so an absurd request
        // is an error rather than an out-of-memory abort.
        counts_of(b.x.n, b.y.n, b.z.n)?;

        Ok(Self {
            nx: b.x.n,
            ny: b.y.n,
            nz: b.z.n,
            xn: graded_nodes(&b.x),
            yn: graded_nodes(&b.y),
            zn: graded_nodes(&b.z),
        })
    }

    #[inline]
    fn point(&self, i: usize, j: usize, k: usize) -> usize {
        i + (self.nx + 1) * (j + (self.ny + 1) * k)
    }

    #[inline]
    fn cell(&self, i: usize, j: usize, k: usize) -> usize {
        i + self.nx * (j + self.ny * k)
    }

    #[inline]
    fn decompose_cell(&self, c: usize) -> (usize, usize, usize) {
        let i = c % self.nx;
        let t = c / self.nx;
        (i, t % self.ny, t / self.ny)
    }

    fn point_coord(&self, p: usize) -> Vec3 {
        let i = p % (self.nx + 1);
        let t = p / (self.nx + 1);
        let j = t % (self.ny + 1);
        let k = t / (self.ny + 1);
        Vec3::new(self.xn[i], self.yn[j], self.zn[k])
    }

    /// Exact for a rectilinear hex: the centre is the node midpoint.
    fn cell_centre(&self, c: usize) -> Vec3 {
        let (i, j, k) = self.decompose_cell(c);
        Vec3::new(
            0.5 * (self.xn[i] + self.xn[i + 1]),
            0.5 * (self.yn[j] + self.yn[j + 1]),
            0.5 * (self.zn[k] + self.zn[k + 1]),
        )
    }

    fn n_cells(&self) -> usize {
        self.nx * self.ny * self.nz
    }
}

/// Face counts of the six patches, in `-x +x -y +y -z +z` order.
fn patch_sizes(g: &Grid) -> [usize; 6] {
    [
        g.ny * g.nz,
        g.ny * g.nz,
        g.nx * g.nz,
        g.nx * g.nz,
        g.nx * g.ny,
        g.nx * g.ny,
    ]
}

// ==========================================================================
//  Faces
// ==========================================================================

/// An internal face, stored by topology only. `dir` is 0/1/2 for the owner's
/// `+x`/`+y`/`+z` face; the four corner points are regenerated from it on
/// write.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct IFace {
    own: usize,
    nei: usize,
    dir: u8,
}

#[derive(Debug, Clone, Copy)]
struct Quad {
    p: [usize; 4],
    own: usize,
    /// `None` on a boundary face.
    nei: Option<usize>,
}

/// The four corners of an internal face, wound as the OWNER's `+x`/`+y`/`+z`
/// face of the standard OpenFOAM hex (points 1-2-6-5, 3-7-6-2, 4-5-6-7). Those
/// windings have outward normals for the owner, i.e. owner -> neighbour.
fn internal_quad(g: &Grid, f: IFace) -> Quad {
    let (i, j, k) = g.decompose_cell(f.own);

    let p = match f.dir {
        0 => [
            // +x
            g.point(i + 1, j, k),
            g.point(i + 1, j + 1, k),
            g.point(i + 1, j + 1, k + 1),
            g.point(i + 1, j, k + 1),
        ],
        1 => [
            // +y
            g.point(i, j + 1, k),
            g.point(i, j + 1, k + 1),
            g.point(i + 1, j + 1, k + 1),
            g.point(i + 1, j + 1, k),
        ],
        _ => [
            // +z
            g.point(i, j, k + 1),
            g.point(i + 1, j, k + 1),
            g.point(i + 1, j + 1, k + 1),
            g.point(i, j + 1, k + 1),
        ],
    };

    Quad { p, own: f.own, nei: Some(f.nei) }
}

/// The `idx`'th face of patch `p` (0..5 = `-x +x -y +y -z +z`), wound so the
/// normal points OUT of the domain.
///
/// Within a patch the index runs with the lower-numbered tangential direction
/// fastest, which keeps a patch field in the same i-fastest order as the cells
/// behind it.
fn boundary_quad(g: &Grid, p: usize, idx: usize) -> Quad {
    let (nx, ny, nz) = (g.nx, g.ny, g.nz);

    let (own, pts) = match p {
        0 => {
            // xMin, outward -x
            let (j, k) = (idx % ny, idx / ny);
            (
                g.cell(0, j, k),
                [
                    g.point(0, j, k),
                    g.point(0, j, k + 1),
                    g.point(0, j + 1, k + 1),
                    g.point(0, j + 1, k),
                ],
            )
        }
        1 => {
            // xMax, outward +x
            let (j, k) = (idx % ny, idx / ny);
            (
                g.cell(nx - 1, j, k),
                [
                    g.point(nx, j, k),
                    g.point(nx, j + 1, k),
                    g.point(nx, j + 1, k + 1),
                    g.point(nx, j, k + 1),
                ],
            )
        }
        2 => {
            // yMin, outward -y
            let (i, k) = (idx % nx, idx / nx);
            (
                g.cell(i, 0, k),
                [
                    g.point(i, 0, k),
                    g.point(i + 1, 0, k),
                    g.point(i + 1, 0, k + 1),
                    g.point(i, 0, k + 1),
                ],
            )
        }
        3 => {
            // yMax, outward +y
            let (i, k) = (idx % nx, idx / nx);
            (
                g.cell(i, ny - 1, k),
                [
                    g.point(i, ny, k),
                    g.point(i, ny, k + 1),
                    g.point(i + 1, ny, k + 1),
                    g.point(i + 1, ny, k),
                ],
            )
        }
        4 => {
            // zMin, outward -z
            let (i, j) = (idx % nx, idx / nx);
            (
                g.cell(i, j, 0),
                [
                    g.point(i, j, 0),
                    g.point(i, j + 1, 0),
                    g.point(i + 1, j + 1, 0),
                    g.point(i + 1, j, 0),
                ],
            )
        }
        _ => {
            // zMax, outward +z
            let (i, j) = (idx % nx, idx / nx);
            (
                g.cell(i, j, nz - 1),
                [
                    g.point(i, j, nz),
                    g.point(i + 1, j, nz),
                    g.point(i + 1, j + 1, nz),
                    g.point(i, j + 1, nz),
                ],
            )
        }
    };

    Quad { p: pts, own, nei: None }
}

// ==========================================================================
//  Split patches
// ==========================================================================

/// `(fast, slow)` face counts of one slot, in the order `boundary_quad`
/// decomposes its index.
fn slot_dims(g: &Grid, p: usize) -> (usize, usize) {
    match p {
        0 | 1 => (g.ny, g.nz),
        2 | 3 => (g.nx, g.nz),
        _ => (g.nx, g.ny),
    }
}

/// Node arrays of a slot's two tangential axes, `(fast, slow)`.
fn slot_axes(g: &Grid, p: usize) -> (&[Scalar], &[Scalar]) {
    match p {
        0 | 1 => (&g.yn, &g.zn),
        2 | 3 => (&g.xn, &g.zn),
        _ => (&g.xn, &g.yn),
    }
}

/// Which of its slot's faces an output patch owns.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SlotPart {
    /// All of them, in slot order - the ordinary, unsplit case.
    All,
    /// The window only.
    Window,
    /// The slot minus the window.
    Rest,
}

/// One entry of `constant/polyMesh/boundary`, plus the rule that turns a
/// patch-local face index into a slot-local one.
#[derive(Debug, Clone)]
struct OutPatch {
    name: String,
    type_name: String,
    /// Slot (`-x +x -y +y -z +z`) this patch draws its faces from.
    slot: usize,
    part: SlotPart,
    /// `[a_lo, a_hi, b_lo, b_hi]` of the window in slot coordinates; unused
    /// when `part` is `All`.
    win: [usize; 4],
    /// `startFace`: the patch's first face in the GLOBAL face list.
    start: usize,
    size: usize,
}

impl OutPatch {
    /// Slot-local index of this patch's `idx`'th face, `idx < self.size`.
    ///
    /// A patch is one contiguous `[startFace, startFace + nFaces)` run and
    /// OpenFOAM records nothing else, so a split slot cannot be emitted in
    /// plain (fast, slow) order: every window face first, then every remaining
    /// face. This is the inverse of that regrouping, closed-form so that no
    /// per-face index list has to be materialised for a big mesh.
    fn slot_index(&self, na: usize, idx: usize) -> usize {
        let [a0, a1, b0, b1] = self.win;

        match self.part {
            SlotPart::All => idx,

            SlotPart::Window => {
                let w = a1 - a0;
                (a0 + idx % w) + na * (b0 + idx / w)
            }

            SlotPart::Rest => {
                // Three runs: the full rows below the window, the punched rows
                // beside it, then the full rows above.
                let below = b0 * na;
                if idx < below {
                    return idx;
                }

                let w = a1 - a0;
                let per = na - w;
                let middle = (b1 - b0) * per;

                if idx < below + middle {
                    // `per == 0` cannot reach here: it makes `middle` zero.
                    let t = idx - below;
                    let a = t % per;
                    return if a < a0 { a } else { a + w } + na * (b0 + t / per);
                }

                b1 * na + (idx - below - middle)
            }
        }
    }
}

/// Validate a window against the slot it splits and return it as
/// `[a_lo, a_hi, b_lo, b_hi]`.
fn window_rect(g: &Grid, w: &PatchWindow) -> Result<[usize; 4]> {
    let bad = |m: String| Error::Mesh(format!("blockgen: patch window '{}': {m}", w.name));

    if w.slot >= 6 {
        return Err(bad(format!("slot {} is not one of the six", w.slot)));
    }
    if w.name.is_empty() || w.type_name.is_empty() {
        return Err(bad("needs both a name and a type".to_string()));
    }

    let (na, nb) = slot_dims(g, w.slot);
    let ([a0, b0], [a1, b1]) = (w.lo, w.hi);

    if a0 >= a1 || b0 >= b1 {
        return Err(bad(format!("is empty, [{a0},{a1}) x [{b0},{b1})")));
    }
    if a1 > na || b1 > nb {
        return Err(bad(format!(
            "[{a0},{a1}) x [{b0},{b1}) does not fit the {na} x {nb} faces of slot {}",
            w.slot
        )));
    }
    if (a1 - a0) * (b1 - b0) == na * nb {
        return Err(bad("covers the whole slot, leaving its host patch empty".to_string()));
    }

    Ok([a0, a1, b0, b1])
}

/// The patches as they will be written, in face order.
///
/// `size` and `start` are per SLOT; a split slot contributes two patches whose
/// ranges partition it, so the global face order is unchanged.
fn build_patches(
    g: &Grid,
    b: &BlockSpec,
    size: &[usize; 6],
    start: &[usize; 6],
) -> Result<Vec<OutPatch>> {
    let mut out: Vec<OutPatch> = Vec::with_capacity(7);

    // Validated once, up front: a window naming a slot outside 0..6 would
    // otherwise match nothing in the loop and be dropped without a word.
    let win = match &b.window {
        Some(w) => Some((w, window_rect(g, w)?)),
        None => None,
    };

    for p in 0..6 {
        match win {
            Some((w, rect)) if w.slot == p => {
                let n_win = (rect[1] - rect[0]) * (rect[3] - rect[2]);

                out.push(OutPatch {
                    name: w.name.clone(),
                    type_name: w.type_name.clone(),
                    slot: p,
                    part: SlotPart::Window,
                    win: rect,
                    start: start[p],
                    size: n_win,
                });
                out.push(OutPatch {
                    name: b.patch_name[p].clone(),
                    type_name: b.patch_type[p].clone(),
                    slot: p,
                    part: SlotPart::Rest,
                    win: rect,
                    start: start[p] + n_win,
                    size: size[p] - n_win,
                });
            }
            _ => out.push(OutPatch {
                name: b.patch_name[p].clone(),
                type_name: b.patch_type[p].clone(),
                slot: p,
                part: SlotPart::All,
                win: [0; 4],
                start: start[p],
                size: size[p],
            }),
        }
    }

    // Two patches of the same name is not an error OpenFOAM reports; it just
    // resolves every lookup to the first one.
    for i in 1..out.len() {
        if out[..i].iter().any(|q| q.name == out[i].name) {
            return Err(Error::Mesh(format!(
                "blockgen: two patches are both called '{}'",
                out[i].name
            )));
        }
    }

    Ok(out)
}

/// Physical extent of a window, `([fast_lo, fast_hi], [slow_lo, slow_hi])`.
fn window_extent(g: &Grid, w: &PatchWindow) -> ([Scalar; 2], [Scalar; 2]) {
    let (fa, sa) = slot_axes(g, w.slot);
    let at = |v: &[Scalar], i: usize| v[i.min(v.len() - 1)];
    (
        [at(fa, w.lo[0]), at(fa, w.hi[0])],
        [at(sa, w.lo[1]), at(sa, w.hi[1])],
    )
}

/// Area vector of a planar quad. For a planar face the triangulated area of
/// SPEC section 2 reduces to half the cross product of the diagonals, which is
/// what the geometry module will compute from these same four points.
fn quad_area(g: &Grid, q: &Quad) -> Vec3 {
    let p0 = g.point_coord(q.p[0]);
    let p1 = g.point_coord(q.p[1]);
    let p2 = g.point_coord(q.p[2]);
    let p3 = g.point_coord(q.p[3]);
    (p2 - p0).cross(p3 - p1) * 0.5
}

fn quad_centre(g: &Grid, q: &Quad) -> Vec3 {
    (g.point_coord(q.p[0]) + g.point_coord(q.p[1]) + g.point_coord(q.p[2]) + g.point_coord(q.p[3]))
        * 0.25
}

/// Verify, rather than assume, that the winding gives the right sign.
///
/// A sign error here poisons every flux, gradient and matrix coefficient
/// downstream and is invisible in the file itself, so every face is checked on
/// every write instead of the construction being trusted.
fn winding_ok(g: &Grid, q: &Quad) -> bool {
    let sf = quad_area(g, q);
    let c_own = g.cell_centre(q.own);

    // Internal: must point at the neighbour. Boundary: must point out of the
    // owner, i.e. from the cell centre toward the face centre.
    let target = match q.nei {
        Some(n) => g.cell_centre(n),
        None => quad_centre(g, q),
    };

    sf.dot(target - c_own) > 0.0
}

// ==========================================================================
//  The assembled block
// ==========================================================================

/// Everything the writers need, built once and shared between the polyMesh
/// writer and the initial-field writer so a big mesh is not built twice.
struct Block {
    g: Grid,
    faces: Vec<IFace>,
    patch_size: [usize; 6],
    /// The patches as they are written. Six of them unless a slot is split by
    /// a [`PatchWindow`], in which case that slot contributes two whose ranges
    /// partition it.
    patches: Vec<OutPatch>,
    n_cells: usize,
    n_points: usize,
    n_internal: usize,
    n_boundary: usize,
}

impl Block {
    fn new(b: &BlockSpec) -> Result<Self> {
        let c = counts_of(b.x.n, b.y.n, b.z.n)?;
        let g = Grid::new(b)?;

        let patch_size = patch_sizes(&g);
        let mut patch_start = [0usize; 6];
        let mut acc = c.internal;
        for (p, start) in patch_start.iter_mut().enumerate() {
            *start = acc;
            acc += patch_size[p];
        }

        // Emitted owner-ascending and, within one owner, +x then +y then +z,
        // whose neighbours are c+1 < c+nx < c+nx*ny. That is already
        // upper-triangular order; the sort below is what makes it true rather
        // than assumed, and costs one pass when the construction is right.
        let mut faces: Vec<IFace> = Vec::with_capacity(c.internal);
        for k in 0..g.nz {
            for j in 0..g.ny {
                for i in 0..g.nx {
                    let cell = g.cell(i, j, k);
                    if i + 1 < g.nx {
                        faces.push(IFace { own: cell, nei: cell + 1, dir: 0 });
                    }
                    if j + 1 < g.ny {
                        faces.push(IFace { own: cell, nei: cell + g.nx, dir: 1 });
                    }
                    if k + 1 < g.nz {
                        faces.push(IFace { own: cell, nei: cell + g.nx * g.ny, dir: 2 });
                    }
                }
            }
        }

        if faces.len() != c.internal {
            return Err(Error::Mesh(format!(
                "blockgen: internal face count mismatch, built {} expected {}",
                faces.len(),
                c.internal
            )));
        }

        // The keys are unique, so an unstable sort is still deterministic.
        faces.sort_unstable_by(|a, b| (a.own, a.nei).cmp(&(b.own, b.nei)));

        for f in 0..faces.len() {
            if faces[f].own >= faces[f].nei {
                return Err(Error::Mesh(format!(
                    "blockgen: owner >= neighbour after sorting, at face {f}"
                )));
            }
            if f > 0 && (faces[f - 1].own, faces[f - 1].nei) >= (faces[f].own, faces[f].nei) {
                return Err(Error::Mesh(format!(
                    "blockgen: internal faces are not upper-triangular, at face {f}"
                )));
            }
        }

        let patches = build_patches(&g, b, &patch_size, &patch_start)?;

        Ok(Self {
            g,
            faces,
            patch_size,
            patches,
            n_cells: c.cells,
            n_points: c.points,
            n_internal: c.internal,
            n_boundary: c.boundary,
        })
    }

    fn n_faces(&self) -> usize {
        self.n_internal + self.n_boundary
    }
}

// ==========================================================================
//  Buffered ASCII output
// ==========================================================================

/// The banner and the fixed part of every `FoamFile` header.
///
/// The layout was read off data files, which is what a file format is. Nothing
/// here was taken from another program's source.
const BANNER: &str = r#"/*---------------------------------------------------------------------------*\
| ofgpu  --  GPU-native finite volume CFD                                     |
|                                                                             |
| Written in the OpenFOAM ASCII case format so that existing pre- and         |
| post-processing tools can read it. A file format is not a work: ofgpu is    |
| an independent implementation, neither derived from nor affiliated with     |
| OpenFOAM.                                                                   |
\*---------------------------------------------------------------------------*/
FoamFile
{
    version     2.0;
    format      ascii;
    class       "#;

const SEPARATOR: &str =
    "// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n";

const FOOTER_RULE: &str =
    "// ************************************************************************* //\n";

/// C's `%.*g` with `sig` significant digits.
///
/// Rust's own float `Display` is the shortest decimal that round-trips, which
/// is *more* accurate but writes `0.00001` where OpenFOAM writes `1e-05` and
/// `0.0023437500000000003` where it writes `0.00234375`. Reproducing printf
/// keeps a regenerated case a no-op diff against the checked-in reference
/// cases, which is worth more than the last two bits of a mesh coordinate.
fn fmt_g_prec(v: Scalar, sig: usize) -> String {
    let x = v as f64;
    if x == 0.0 {
        // printf keeps the sign of a negative zero; so does this.
        return if x.is_sign_negative() { "-0".to_string() } else { "0".to_string() };
    }
    if !x.is_finite() {
        return format!("{x}");
    }

    // The decimal exponent comes from the formatter rather than from `log10`,
    // which is off by one at exact powers of ten on some libm builds.
    let sci = format!("{:.*e}", sig - 1, x);
    let (mant, exp) = match sci.split_once('e') {
        Some((m, e)) => (m, e.parse::<i32>().unwrap_or(0)),
        None => (sci.as_str(), 0),
    };

    // printf switches to the exponent style below 1e-4 and at or above 10^sig.
    if exp < -4 || exp >= sig as i32 {
        format!(
            "{}e{}{:02}",
            trim_trailing_zeros(mant),
            if exp < 0 { '-' } else { '+' },
            exp.abs()
        )
    } else {
        let dec = (sig as i32 - 1 - exp).max(0) as usize;
        trim_trailing_zeros(&format!("{:.*}", dec, x))
    }
}

/// `%g` at printf's default precision, for the dimensioned constants in
/// `constant/` and for the console summary.
fn fmt_g(v: Scalar) -> String {
    fmt_g_prec(v, 6)
}

fn trim_trailing_zeros(s: &str) -> String {
    if s.contains('.') {
        s.trim_end_matches('0').trim_end_matches('.').to_string()
    } else {
        s.to_string()
    }
}

/// Buffered ASCII writer.
///
/// The buffer is flushed by hand rather than left to a `BufWriter` so integer
/// rendering can write straight into it.
struct TextOut {
    file: File,
    buf: String,
    cap: usize,
    path: PathBuf,
}

impl TextOut {
    fn create(path: &Path) -> Result<Self> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).path(parent)?;
        }
        // Rust never translates line endings, so the files stay LF on Windows
        // too - OpenFOAM reads either, but LF keeps them diffable.
        let file = File::create(path).path(path)?;
        let cap = 1 << 20;
        Ok(Self { file, buf: String::with_capacity(cap), cap, path: path.to_path_buf() })
    }

    fn flush(&mut self) -> Result<()> {
        if !self.buf.is_empty() {
            self.file.write_all(self.buf.as_bytes()).path(&self.path)?;
            self.buf.clear();
        }
        Ok(())
    }

    fn reserve(&mut self, k: usize) -> Result<()> {
        if self.buf.len() + k > self.cap {
            self.flush()?;
        }
        Ok(())
    }

    fn s(&mut self, t: &str) -> Result<()> {
        if t.len() >= self.cap {
            self.flush()?;
            return self.file.write_all(t.as_bytes()).path(&self.path);
        }
        self.reserve(t.len())?;
        self.buf.push_str(t);
        Ok(())
    }

    fn c(&mut self, ch: char) -> Result<()> {
        self.reserve(4)?;
        self.buf.push(ch);
        Ok(())
    }

    /// Decimal integer, formatted by hand: there are O(10^8) of these in a big
    /// mesh and the formatting machinery dominates the profile if it is used.
    fn num(&mut self, v: usize) -> Result<()> {
        self.reserve(24)?;
        let mut tmp = [0u8; 20];
        let mut k = 0;
        let mut x = v;
        loop {
            tmp[k] = b'0' + (x % 10) as u8;
            k += 1;
            x /= 10;
            if x == 0 {
                break;
            }
        }
        while k > 0 {
            k -= 1;
            self.buf.push(tmp[k] as char);
        }
        Ok(())
    }

    /// 15 significant digits, far more than OpenFOAM's default
    /// `writePrecision` of 6: a graded mesh loses real geometric accuracy at 6.
    fn real(&mut self, v: Scalar) -> Result<()> {
        self.reserve(48)?;
        self.buf.push_str(&fmt_g_prec(v, 15));
        Ok(())
    }

    fn finish(mut self) -> Result<()> {
        self.flush()?;
        self.file.flush().path(&self.path)
    }
}

fn write_foam_header(os: &mut TextOut, cls: &str, object: &str, note: &str) -> Result<()> {
    os.s(BANNER)?;
    os.s(cls)?;
    os.s(";\n")?;

    if !note.is_empty() {
        os.s("    note        \"")?;
        os.s(note)?;
        os.s("\";\n")?;
    }

    os.s("    location    \"constant/polyMesh\";\n    object      ")?;
    os.s(object)?;
    os.s(";\n}\n")?;
    os.s(SEPARATOR)?;
    os.s("\n")
}

fn write_foam_footer(os: &mut TextOut) -> Result<()> {
    os.s("\n\n")?;
    os.s(FOOTER_RULE)
}

fn write_quad(os: &mut TextOut, q: &Quad) -> Result<()> {
    os.s("4(")?;
    os.num(q.p[0])?;
    os.c(' ')?;
    os.num(q.p[1])?;
    os.c(' ')?;
    os.num(q.p[2])?;
    os.c(' ')?;
    os.num(q.p[3])?;
    os.s(")\n")
}

fn bad_winding(what: &str, f: usize, q: &Quad) -> Error {
    let nei: i64 = match q.nei {
        Some(n) => n as i64,
        None => -1,
    };
    Error::Mesh(format!(
        "blockgen: {what} face {f} (owner {}, neighbour {nei}) is wound the wrong way",
        q.own
    ))
}

// ==========================================================================
//  polyMesh
// ==========================================================================

/// Generate and write
/// `case_dir/constant/polyMesh/{points,faces,owner,neighbour,boundary}`.
pub fn write_block_mesh(case_dir: &Path, b: &BlockSpec) -> Result<()> {
    let block = Block::new(b)?;
    write_poly_mesh(case_dir, &block)
}

fn write_poly_mesh(case_dir: &Path, block: &Block) -> Result<()> {
    let g = &block.g;
    let dir = case_dir.join("constant").join("polyMesh");
    fs::create_dir_all(&dir).path(&dir)?;

    let note = format!(
        "nPoints:{}  nCells:{}  nFaces:{}  nInternalFaces:{}",
        block.n_points,
        block.n_cells,
        block.n_faces(),
        block.n_internal
    );

    // ---- points ----------------------------------------------------------
    {
        let mut os = TextOut::create(&dir.join("points"))?;
        write_foam_header(&mut os, "vectorField", "points", "")?;

        os.num(block.n_points)?;
        os.s("\n(\n")?;

        for k in 0..=g.nz {
            for j in 0..=g.ny {
                for i in 0..=g.nx {
                    os.c('(')?;
                    os.real(g.xn[i])?;
                    os.c(' ')?;
                    os.real(g.yn[j])?;
                    os.c(' ')?;
                    os.real(g.zn[k])?;
                    os.s(")\n")?;
                }
            }
        }

        os.s(")")?;
        write_foam_footer(&mut os)?;
        os.finish()?;
    }

    // ---- faces -----------------------------------------------------------
    {
        let mut os = TextOut::create(&dir.join("faces"))?;
        write_foam_header(&mut os, "faceList", "faces", "")?;

        os.num(block.n_faces())?;
        os.s("\n(\n")?;

        for (f, iface) in block.faces.iter().enumerate() {
            let q = internal_quad(g, *iface);
            if !winding_ok(g, &q) {
                return Err(bad_winding("internal", f, &q));
            }
            write_quad(&mut os, &q)?;
        }

        for patch in &block.patches {
            let (na, _) = slot_dims(g, patch.slot);
            for idx in 0..patch.size {
                let q = boundary_quad(g, patch.slot, patch.slot_index(na, idx));
                if !winding_ok(g, &q) {
                    return Err(bad_winding(&patch.name, idx, &q));
                }
                write_quad(&mut os, &q)?;
            }
        }

        os.s(")")?;
        write_foam_footer(&mut os)?;
        os.finish()?;
    }

    // ---- owner -----------------------------------------------------------
    {
        let mut os = TextOut::create(&dir.join("owner"))?;
        write_foam_header(&mut os, "labelList", "owner", &note)?;

        os.num(block.n_faces())?;
        os.s("\n(\n")?;

        for f in &block.faces {
            os.num(f.own)?;
            os.c('\n')?;
        }

        for patch in &block.patches {
            let (na, _) = slot_dims(g, patch.slot);
            for idx in 0..patch.size {
                os.num(boundary_quad(g, patch.slot, patch.slot_index(na, idx)).own)?;
                os.c('\n')?;
            }
        }

        os.s(")")?;
        write_foam_footer(&mut os)?;
        os.finish()?;
    }

    // ---- neighbour -------------------------------------------------------
    {
        let mut os = TextOut::create(&dir.join("neighbour"))?;
        write_foam_header(&mut os, "labelList", "neighbour", &note)?;

        os.num(block.n_internal)?;
        os.s("\n(\n")?;

        for f in &block.faces {
            os.num(f.nei)?;
            os.c('\n')?;
        }

        os.s(")")?;
        write_foam_footer(&mut os)?;
        os.finish()?;
    }

    // ---- boundary --------------------------------------------------------
    {
        let mut os = TextOut::create(&dir.join("boundary"))?;
        write_foam_header(&mut os, "polyBoundaryMesh", "boundary", "")?;

        os.num(block.patches.len())?;
        os.s("\n(\n")?;

        for patch in &block.patches {
            os.s("    ")?;
            os.s(&patch.name)?;
            os.s("\n    {\n        type            ")?;
            os.s(&patch.type_name)?;
            os.s(";\n        nFaces          ")?;
            os.num(patch.size)?;
            os.s(";\n        startFace       ")?;
            os.num(patch.start)?;
            os.s(";\n    }\n")?;
        }

        os.s(")")?;
        write_foam_footer(&mut os)?;
        os.finish()?;
    }

    println!(
        "[blockgen] {}: {} cells ({}x{}x{}), {} points, {} faces ({} internal)",
        dir.display(),
        block.n_cells,
        g.nx,
        g.ny,
        g.nz,
        block.n_points,
        block.n_faces(),
        block.n_internal
    );

    Ok(())
}

// ==========================================================================
//  Cases
// ==========================================================================

/// The ready-to-run cases `generate_cases` knows how to build.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CaseKind {
    /// Plane channel, graded to both walls, 2-D.
    Channel,
    /// Lid-driven cavity, 2-D.
    Cavity,
    /// Backward-facing-step BOX, 2-D. See `case_block_spec` for why it is not
    /// a true step.
    Step,
    /// Uniform benchmark box.
    Big,
    /// Buoyant plume in a room-sized box, with a burner inlet let into the
    /// middle of the floor and one open face at `xMax` for everything the
    /// burner injects to leave through.
    Plume,
    /// Two-dimensional dam break: a column of water of width `a` and height
    /// `2a` released in a tank open at the top, in the geometry Martin & Moyce
    /// (1952) measured the surge front of. See [`DAM_BREAK_A`].
    DamBreak,
}

impl CaseKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Channel => "channel",
            Self::Cavity => "cavity",
            Self::Step => "step",
            Self::Big => "big",
            Self::Plume => "plume",
            Self::DamBreak => "damBreak",
        }
    }

    pub fn from_name(s: &str) -> Option<Self> {
        match s {
            "channel" => Some(Self::Channel),
            "cavity" => Some(Self::Cavity),
            "step" => Some(Self::Step),
            "big" => Some(Self::Big),
            "plume" => Some(Self::Plume),
            "damBreak" | "dambreak" => Some(Self::DamBreak),
            _ => None,
        }
    }

    /// The resolution the C++ `generate_cases` uses when none is given.
    pub fn default_resolution(self) -> (usize, usize, usize) {
        match self {
            Self::Channel => (200, 120, 1),
            Self::Cavity => (128, 128, 1),
            Self::Step => (300, 100, 1),
            Self::Big => (160, 160, 160),
            // 14.64 x 6.24 x 3 m at ~0.15 m: the cell count of the published
            // FDS-vs-GPU fire benchmark, so timings are comparable to it.
            Self::Plume => (98, 42, 20),
            // 5a x 3a at a/30, so the column is 30 cells wide and 60 tall -
            // enough for the surge front to be resolved to a few per cent of
            // `a` without the case taking minutes.
            Self::DamBreak => (150, 90, 1),
        }
    }
}

impl std::str::FromStr for CaseKind {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self> {
        Self::from_name(s)
            .ok_or_else(|| {
                Error::Config(format!(
                    "unknown case '{s}' (channel|cavity|step|big|plume|damBreak)"
                ))
            })
    }
}

/// Column width of the dam-break case, metres.
///
/// Martin & Moyce (1952) report the collapse of a column of width `a` and
/// height `n^2 a` in the dimensionless variables `Z = z/a` and
/// `T = t sqrt(g/a)`, so the ABSOLUTE size is a free choice and only the
/// aspect ratio and the dimensionless groups matter. 0.05 m puts the case at
/// laboratory scale with a Bond number `rho g a^2/sigma` of about 340, i.e.
/// gravity-dominated and surface tension a small correction, which is the
/// regime their experiment was in.
pub const DAM_BREAK_A: Scalar = 0.05;

/// Column height as a multiple of the width - Martin & Moyce's `n^2`.
pub const DAM_BREAK_ASPECT: Scalar = 2.0;

/// Side of the plume burner opening before it is snapped to whole cells.
const PLUME_INLET_SIDE: Scalar = 1.2;

/// Velocity out of the burner.
const PLUME_INLET_U: Scalar = 2.0;

/// Ambient air, 20 C.
const PLUME_T_AMBIENT: Scalar = 293.15;

/// Burner outlet, 900 C.
const PLUME_T_INLET: Scalar = 1173.15;

/// The half-open cell range whose CENTRES fall inside `centre +- width/2`.
///
/// Snapping by cell centre rather than by node is what keeps the opening
/// symmetric about `centre` to within half a cell on a grid that does not
/// happen to have a node there - the plume's does not, on either axis.
///
/// A window narrower than one cell would come out empty, so it collapses to
/// the single cell whose centre is nearest instead: an inlet patch with no
/// faces is a mesh the solver cannot use.
fn centred_cell_range(nodes: &[Scalar], centre: Scalar, width: Scalar) -> (usize, usize) {
    let n = nodes.len().saturating_sub(1);
    if n == 0 {
        return (0, 0);
    }

    let mid = |i: usize| 0.5 * (nodes[i] + nodes[i + 1]);
    let (lo, hi) = (centre - 0.5 * width, centre + 0.5 * width);

    let mut a = n;
    let mut b = 0;
    for i in 0..n {
        let c = mid(i);
        if c >= lo && c <= hi {
            a = a.min(i);
            b = i + 1;
        }
    }

    if a < b {
        return (a, b);
    }

    let mut best = 0;
    let mut best_d = Scalar::MAX;
    for i in 0..n {
        let d = (mid(i) - centre).abs();
        if d < best_d {
            best_d = d;
            best = i;
        }
    }
    (best, best + 1)
}

/// The burner opening, as a window of the `zMin` slot.
fn plume_inlet_window(b: &BlockSpec) -> PatchWindow {
    let (i0, i1) = centred_cell_range(&graded_nodes(&b.x), 0.0, PLUME_INLET_SIDE);
    let (j0, j1) = centred_cell_range(&graded_nodes(&b.y), 0.0, PLUME_INLET_SIDE);

    PatchWindow {
        slot: 4,
        lo: [i0, j0],
        hi: [i1, j1],
        name: "inlet".to_string(),
        type_name: "patch".to_string(),
    }
}

/// Geometry, grading and patch naming per case, carried across from
/// `generate_cases.cu`'s `main`.
fn case_block_spec(kind: CaseKind, nx: usize, ny: usize, nz: usize) -> BlockSpec {
    let mut b = BlockSpec::default();

    let (names, types): ([&str; 6], [&str; 6]) = match kind {
        CaseKind::Channel => {
            // Plane channel, Re_tau style geometry. A streamwise cyclic pair
            // would be the physical choice but cyclics need a coupled patch
            // pair, so this uses inlet/outlet instead.
            b.x.lo = 0.0;
            b.x.hi = 4.0;
            b.y.lo = -1.0;
            b.y.hi = 1.0;
            b.y.expansion = 20.0;
            b.y.two_sided = true;
            b.z.lo = 0.0;
            b.z.hi = 0.1;
            (
                ["inlet", "outlet", "bottomWall", "topWall", "back", "front"],
                ["patch", "patch", "wall", "wall", "empty", "empty"],
            )
        }
        CaseKind::Cavity => {
            // Lid-driven cavity: the classic case, but at 128^2 instead of
            // 20^2. Cube 0..0.1 on all three axes; the z thickness is
            // irrelevant to a 2-D solution, it only scales V and magSf
            // uniformly.
            b.x.lo = 0.0;
            b.x.hi = 0.1;
            b.y.lo = 0.0;
            b.y.hi = 0.1;
            b.z.lo = 0.0;
            b.z.hi = 0.1;
            (
                ["leftWall", "rightWall", "fixedWall", "movingWall", "back", "front"],
                ["wall", "wall", "wall", "wall", "empty", "empty"],
            )
        }
        CaseKind::Step => {
            // NOT a true backward-facing step. blockgen makes exactly one
            // rectangular block, so there is no step face and no inlet-channel
            // offset: this is the BFS *outlet* box (30h x 2h) on its own,
            // graded toward both walls and refined toward the inlet in x. It
            // exercises the same cell counts and aspect ratios as a real BFS
            // mesh and is useful for performance and BC plumbing, but its
            // solution is a plane channel developing from a uniform inlet, not
            // a separating flow.
            b.x.lo = 0.0;
            b.x.hi = 30.0;
            b.x.expansion = 4.0;
            b.y.lo = 0.0;
            b.y.hi = 2.0;
            b.y.expansion = 10.0;
            b.y.two_sided = true;
            b.z.lo = 0.0;
            b.z.hi = 1.0;
            (
                ["inlet", "outlet", "lowerWall", "upperWall", "back", "front"],
                ["patch", "patch", "wall", "wall", "empty", "empty"],
            )
        }
        CaseKind::Big => {
            // Uniform unit cube for benchmarking. 160^3 = 4 096 000 cells.
            b.x.lo = 0.0;
            b.x.hi = 1.0;
            b.y.lo = 0.0;
            b.y.hi = 1.0;
            b.z.lo = 0.0;
            b.z.hi = 1.0;
            (
                ["inlet", "outlet", "bottomWall", "topWall", "backWall", "frontWall"],
                ["patch", "patch", "wall", "wall", "wall", "wall"],
            )
        }
        CaseKind::DamBreak => {
            // A tank 5a wide and 3a tall, one cell thick, with the column of
            // water in the bottom-left corner. Gravity is along -y because the
            // resolved plane of a blockgen 2-D case is x-y: the front and back
            // are the `empty` slots.
            //
            // 5a of run-out is what the surge front needs before it reaches
            // the far wall at about T = 3, which is past the range Martin &
            // Moyce's early-time data covers; 3a of headroom keeps the
            // atmosphere patch out of the collapsing column's way.
            b.x.lo = 0.0;
            b.x.hi = 5.0 * DAM_BREAK_A;
            b.y.lo = 0.0;
            b.y.hi = 3.0 * DAM_BREAK_A;
            b.z.lo = 0.0;
            b.z.hi = DAM_BREAK_A / 20.0;
            (
                ["leftWall", "rightWall", "lowerWall", "atmosphere", "back", "front"],
                ["wall", "wall", "wall", "patch", "empty", "empty"],
            )
        }
        CaseKind::Plume => {
            // Fire plume in a room-sized box with exactly ONE opening: +x.
            //
            // Every other face is solid. That is not a modelling preference,
            // it is what makes the case solvable. Mass entering at the burner
            // has to leave somewhere, and with several open faces there is no
            // unique split of the outflow between them - the potential-flow
            // solve in `potential_flow.rs` would then need a pressure level
            // per opening rather than the one Dirichlet reference a single
            // `outlet` supplies. One opening also makes the imbalance test
            // meaningful: `inlet_flux + outlet_flux` is a two-term sum whose
            // cancellation is exactly the conservation claim.
            //
            // The burner is a window cut out of the floor further down, once
            // the cell size is known.
            b.x.lo = -8.32;
            b.x.hi = 6.32;
            b.y.lo = -2.62;
            b.y.hi = 3.62;
            b.z.lo = 0.0;
            b.z.hi = 3.0;
            (
                ["wallXMin", "outlet", "wallYMin", "wallYMax", "floor", "ceiling"],
                ["wall", "patch", "wall", "wall", "wall", "wall"],
            )
        }
    };

    b.patch_name = names.map(String::from);
    b.patch_type = types.map(String::from);

    b.x.n = nx;
    b.y.n = ny;
    b.z.n = nz;

    // An "empty" patch is only legal with a single cell in that direction.
    if nz > 1 {
        for p in 4..6 {
            if b.patch_type[p] == "empty" {
                b.patch_type[p] = "wall".to_string();
            }
        }
    }

    // Needs the axes to be sized first: the opening is snapped to whole cells.
    if kind == CaseKind::Plume {
        b.window = Some(plume_inlet_window(&b));
    }

    b
}

/// Write a complete runnable case: polyMesh, `constant/`, `system/` and a `0/`
/// directory whose fields are real per-cell profiles rather than a uniform
/// guess.
pub fn write_case(case_dir: &Path, kind: CaseKind, nx: usize, ny: usize, nz: usize) -> Result<()> {
    if nx < 1 || ny < 1 || nz < 1 {
        return Err(Error::Config(format!(
            "generate_cases: bad resolution {nx} x {ny} x {nz}"
        )));
    }

    let b = case_block_spec(kind, nx, ny, nz);
    let block = Block::new(&b)?;

    write_poly_mesh(case_dir, &block)?;

    // BEFORE write_system, which is what makes this work: `write_dict` never
    // overwrites a file that already exists, so the two-phase dictionaries
    // written here survive and `write_system` fills in only what is missing
    // (controlDict, and nothing else for this case).
    if kind == CaseKind::DamBreak {
        write_dam_break_system(case_dir)?;
    }

    write_system(case_dir)?;

    // The dam break is a two-phase case and shares almost nothing with the
    // single-phase ones below: its `constant/transportProperties` names two
    // fluids and has no single `nu`, its gravity is along -y rather than -z,
    // and its `0/` holds `alpha.water` and `p_rgh` rather than `p`, `T` and
    // four turbulence fields. Diverting here keeps all of that out of the
    // single-phase path, rather than putting a `kind == DamBreak` branch in
    // each of its four writers.
    if kind == CaseKind::DamBreak {
        write_two_phase_constant(case_dir)?;
        write_gravity_vector(case_dir, Vec3::new(0.0, -9.81, 0.0))?;
        write_dam_break_fields(case_dir, &block)?;

        println!(
            "damBreak: {} x {} x {} = {} cells -> {}",
            nx,
            ny,
            nz,
            block.n_cells,
            case_dir.display()
        );
        println!(
            "  column {} x {} m in a {} x {} m tank; Martin & Moyce a = {} m",
            fmt_g(DAM_BREAK_A),
            fmt_g(DAM_BREAK_ASPECT * DAM_BREAK_A),
            fmt_g(5.0 * DAM_BREAK_A),
            fmt_g(3.0 * DAM_BREAK_A),
            fmt_g(DAM_BREAK_A)
        );
        return Ok(());
    }

    // Air for the plume; every other case keeps the 1e-5 it has always had.
    let nu: Scalar = if kind == CaseKind::Plume { 1.5e-5 } else { 1e-5 };

    // Only the plume carries a temperature equation, so only its dictionary
    // gets the two Prandtl numbers that equation reads. Writing them into the
    // other cases would change files nothing there looks at.
    let extra = if kind == CaseKind::Plume {
        // `TRef` is the buoyancy REFERENCE, not a Boussinesq expansion point:
        // `momentum::BuoyancyCoeffs` builds the body force from the ideal-gas
        // ratio `TRef/T`, which stays exact over this case's 293 K to 1173 K
        // where a `beta*(T - TRef)` would be wrong by a factor of three. No
        // `beta` is written, and that omission is deliberate - BUOYANT.md
        // section 2.
        "\n\n// Air at ambient. Read by the temperature equation.\n\
         Pr              0.71;\n\
         Prt             0.85;\n\n\
         // Buoyancy reference: b = g*(TRef/T - 1). Deliberately no beta - this\n\
         // solver does not use the Boussinesq approximation.\n\
         TRef            293.15;"
    } else {
        ""
    };
    write_constant(case_dir, nu, "kEpsilon", extra)?;

    // Gravity is what makes the plume a plume. Every other case here is
    // isothermal, and a `constant/g` in one of those would be read by nothing.
    if kind == CaseKind::Plume {
        write_gravity(case_dir)?;
    }

    // Wall-normal direction and half height per case, so the initial profile is
    // oriented the way the geometry expects.
    let inlet = b.window.as_ref().map(|w| window_extent(&block.g, w));
    let (u_ref, half_height): (Scalar, Scalar) = match kind {
        CaseKind::Channel => (1.0, 1.0),
        CaseKind::Step => (10.0, 1.0),
        CaseKind::Cavity => (1.0, 0.05),
        CaseKind::Big => (1.0, 0.5),
        // Unreachable: the two-phase case returned above, before any of this.
        // Present because the compiler counts arms, not reachability.
        CaseKind::DamBreak => (0.0, DAM_BREAK_A),
        // The plume has no wall-normal profile; `half_height` only feeds the
        // inlet mixing length, which scales with the burner, not the room. The
        // two snapped sides differ by well under a percent, so their mean is
        // the honest single number to hand it.
        CaseKind::Plume => {
            let side = match inlet {
                Some((fx, fy)) => 0.25 * ((fx[1] - fx[0]) + (fy[1] - fy[0])),
                None => 0.5 * PLUME_INLET_SIDE,
            };
            (PLUME_INLET_U, side)
        }
    };

    write_initial_fields(case_dir, kind, &block, nu, u_ref, half_height, 1)?;

    if let (Some(w), Some((fx, fy))) = (b.window.as_ref(), inlet) {
        println!(
            "  {}: {} x {} m over x [{}, {}] y [{}, {}] - {} of the {} {} faces",
            w.name,
            fmt_g(fx[1] - fx[0]),
            fmt_g(fy[1] - fy[0]),
            fmt_g(fx[0]),
            fmt_g(fx[1]),
            fmt_g(fy[0]),
            fmt_g(fy[1]),
            (w.hi[0] - w.lo[0]) * (w.hi[1] - w.lo[1]),
            block.patch_size[w.slot],
            b.patch_name[w.slot],
        );
    }

    println!(
        "{}: {} x {} x {} = {} cells -> {}",
        kind.as_str(),
        nx,
        ny,
        nz,
        block.n_cells,
        case_dir.display()
    );

    Ok(())
}

// --------------------------------------------------------------------------
//  system/ and constant/ dictionaries
// --------------------------------------------------------------------------

/// Written only when absent, so regenerating a mesh never clobbers a case
/// someone has set up by hand.
///
/// `location` is `"system"` for every dictionary, including the ones that land
/// under `constant/`. That is what the C++ writes and what the reference cases
/// in `cases/` contain; OpenFOAM ignores the field on read, so it is kept
/// verbatim rather than "fixed" into a needless diff.
///
/// Unlike the C++, a failed write is reported instead of shrugged off: a case
/// missing half its dictionaries is not a convenience, it is a broken case.
fn write_dict(dir: &Path, object: &str, body: &str) -> Result<()> {
    let path = dir.join(object);
    if path.exists() {
        return Ok(());
    }

    fs::create_dir_all(dir).path(dir)?;

    let mut os = TextOut::create(&path)?;
    os.s(BANNER)?;
    os.s("dictionary;\n    location    \"system\";\n    object      ")?;
    os.s(object)?;
    os.s(";\n}\n")?;
    os.s(SEPARATOR)?;
    os.s("\n")?;
    os.s(body)?;
    os.s("\n\n")?;
    os.s(FOOTER_RULE)?;
    os.finish()
}

fn write_system(case_dir: &Path) -> Result<()> {
    let dir = case_dir.join("system");

    write_dict(
        &dir,
        "controlDict",
        "application     foamRun;\n\
         startFrom       startTime;\n\
         startTime       0;\n\
         stopAt          endTime;\n\
         endTime         1;\n\
         deltaT          1;\n\
         writeControl    timeStep;\n\
         writeInterval   1;\n\
         purgeWrite      0;\n\
         writeFormat     ascii;\n\
         writePrecision  6;\n\
         writeCompression off;\n\
         timeFormat      general;\n\
         timePrecision   6;\n\
         runTimeModifiable true;",
    )?;

    write_dict(
        &dir,
        "fvSchemes",
        "ddtSchemes\n{\n    default         steadyState;\n}\n\n\
         gradSchemes\n{\n    default         Gauss linear;\n}\n\n\
         divSchemes\n{\n    default         none;\n\
         \x20   div(phi,U)       Gauss linearUpwind grad(U);\n\
         \x20   div(phi,T)       bounded Gauss upwind;\n\
         \x20   div(phi,k)       bounded Gauss upwind;\n\
         \x20   div(phi,epsilon) bounded Gauss upwind;\n\
         \x20   div(phi,omega)   bounded Gauss upwind;\n}\n\n\
         laplacianSchemes\n{\n    default         Gauss linear corrected;\n}\n\n\
         interpolationSchemes\n{\n    default         linear;\n}\n\n\
         snGradSchemes\n{\n    default         corrected;\n}",
    )?;

    // ofgpu looks up solvers/<var>/... by exact name, so the sub-dictionaries
    // are written out one per variable rather than behind a regex key.
    //
    // `p` and `U` are here for `ofgpu-buoyant`; the frozen-flow drivers never
    // look them up. `p` is the one entry whose settings actually matter: the
    // pressure Poisson equation is diagonally weak and takes hundreds of
    // sweeps where every other equation here takes three, so it gets DIC
    // rather than Jacobi and a budget to match. `relTol 0.01` is deliberate
    // too - inside a SIMPLE loop the pressure only has to be solved as well as
    // the coefficients it was assembled from deserve, and driving it to 1e-8
    // on an outer iteration that is about to change `rAUf` is wasted work.
    write_dict(
        &dir,
        "fvSolution",
        "solvers\n{\n\
         \x20   p\n    {\n\
         \x20       solver          PBiCGStab;\n\
         \x20       preconditioner  DIC;\n\
         \x20       tolerance       1e-08;\n\
         \x20       relTol          0.01;\n\
         \x20       maxIter         1000;\n\
         \x20   }\n\n\
         \x20   Phi\n    {\n\
         \x20       solver          PBiCGStab;\n\
         \x20       preconditioner  DIC;\n\
         \x20       tolerance       1e-12;\n\
         \x20       relTol          0;\n\
         \x20       maxIter         5000;\n\
         \x20   }\n\n\
         \x20   U\n    {\n\
         \x20       solver          PBiCGStab;\n\
         \x20       preconditioner  diagonal;\n\
         \x20       tolerance       1e-08;\n\
         \x20       relTol          0.1;\n\
         \x20       maxIter         200;\n\
         \x20   }\n\n\
         \x20   T\n    {\n\
         \x20       solver          PBiCGStab;\n\
         \x20       preconditioner  diagonal;\n\
         \x20       tolerance       1e-08;\n\
         \x20       relTol          0.01;\n\
         \x20       maxIter         200;\n\
         \x20   }\n\n\
         \x20   k\n    {\n\
         \x20       solver          PBiCGStab;\n\
         \x20       preconditioner  diagonal;\n\
         \x20       tolerance       1e-08;\n\
         \x20       relTol          0.01;\n\
         \x20       maxIter         200;\n\
         \x20   }\n\n\
         \x20   epsilon\n    {\n\
         \x20       solver          PBiCGStab;\n\
         \x20       preconditioner  diagonal;\n\
         \x20       tolerance       1e-08;\n\
         \x20       relTol          0.01;\n\
         \x20       maxIter         200;\n\
         \x20   }\n\n\
         \x20   omega\n    {\n\
         \x20       solver          PBiCGStab;\n\
         \x20       preconditioner  diagonal;\n\
         \x20       tolerance       1e-08;\n\
         \x20       relTol          0.01;\n\
         \x20       maxIter         200;\n\
         \x20   }\n}\n\n\
         SIMPLE\n{\n    nNonOrthogonalCorrectors 0;\n}\n\n\
         // U 0.7 with p 0.3 is what OpenFOAM's buoyant cases use, and the two\n\
         // summing to one is the usual rule of thumb behind it.\n\
         relaxationFactors\n{\n    fields\n    {\n\
         \x20       p               0.3;\n\
         \x20   }\n\n\
         \x20   equations\n    {\n\
         \x20       U               0.7;\n\
         \x20       T               0.7;\n\
         \x20       k               0.7;\n\
         \x20       epsilon         0.7;\n\
         \x20       omega           0.7;\n\
         \x20   }\n}",
    )
}

/// `extra` is appended verbatim to `physicalProperties`, for the entries a
/// particular case's physics needs and the others have no use for.
fn write_constant(case_dir: &Path, nu: Scalar, model: &str, extra: &str) -> Result<()> {
    let dir = case_dir.join("constant");

    write_dict(
        &dir,
        "physicalProperties",
        &format!(
            "viscosityModel  constant;\n\nnu              [0 2 -1 0 0 0 0] {};{extra}",
            fmt_g(nu)
        ),
    )?;

    // No `blended` entry. ofgpu always blends the two branches of the law
    // of the wall, so there is nothing for a switch to select; see the note
    // beside `WallFunctionCoeffs` in `io::case` for why the switch was
    // removed rather than implemented.
    write_dict(
        &dir,
        "momentumTransport",
        &format!(
            "simulationType  RAS;\n\nRAS\n{{\n\
             \x20   model           {model};\n\
             \x20   turbulence      on;\n\
             \x20   printCoeffs     on;\n\
}}"
        ),
    )
}

/// `constant/g`, the one dictionary whose absence silently removes the physics.
///
/// Written through [`write_dict`] like every other dictionary, so it carries
/// the same header. OpenFOAM's own `g` file declares
/// `class uniformDimensionedVectorField` rather than `dictionary`; the class
/// line is cosmetic to every reader in this repository - `FoamDict` never looks
/// at it - and keeping one writer is worth more than matching a string
/// OpenFOAM also ignores on read.
///
/// The entry is spelled `g` rather than `value` because that is what
/// `BUOYANT.md` section 2 asks for; [`crate::momentum::BuoyancyCoeffs::from_case`]
/// accepts either.
fn write_gravity(case_dir: &Path) -> Result<()> {
    write_gravity_vector(case_dir, Vec3::new(0.0, 0.0, -9.81))
}

/// `constant/g` with gravity along whichever axis the case's resolved plane
/// leaves free.
///
/// The plume is a 3-D box and falls down `-z`; a blockgen 2-D case resolves
/// `x-y` and puts its `empty` patches on the `z` slots, so its gravity has to
/// be along `-y` - or the whole body force lands in the direction the mesh
/// does not resolve and the case sits there doing nothing.
fn write_gravity_vector(case_dir: &Path, g: Vec3) -> Result<()> {
    write_dict(
        &case_dir.join("constant"),
        "g",
        &format!(
            "dimensions      [0 1 -2 0 0 0 0];\n\n\
             g               [0 1 -2 0 0 0 0] ({} {} {});",
            fmt_g(g.x),
            fmt_g(g.y),
            fmt_g(g.z)
        ),
    )
}

/// `system/fvSchemes`, `system/fvSolution` and `system/controlDict` for the
/// two-phase case.
///
/// Written separately from [`write_system`] rather than as extra entries in
/// it, because a `PIMPLE` dictionary is NOT inert to the other drivers:
/// `io::case::AlgorithmControls::read` takes the last of `SIMPLE`, `PISO` and
/// `PIMPLE` that the file defines, so putting one in the shared dictionary
/// would have given every single-phase case generated here `nCorrectors 3`
/// and a start-up line reading PIMPLE. The two-phase settings belong to the
/// two-phase case.
///
/// Every entry is one SPEC-LIT S20 names; the comments in the file say which.
fn write_dam_break_system(case_dir: &Path) -> Result<()> {
    let dir = case_dir.join("system");

    write_dict(
        &dir,
        "controlDict",
        "application     ofgpu-vof;\n\
         startFrom       startTime;\n\
         startTime       0;\n\
         stopAt          endTime;\n\
         endTime         0.5;\n\
         deltaT          0.0002;\n\
         maxDeltaT       0.001;\n\
         writeControl    runTime;\n\
         writeInterval   0.05;\n\
         purgeWrite      0;\n\
         writeFormat     ascii;\n\
         writePrecision  6;\n\
         writeCompression off;\n\
         timeFormat      general;\n\
         timePrecision   6;\n\
         runTimeModifiable true;",
    )?;

    write_dict(
        &dir,
        "fvSchemes",
        "// SPEC-LIT S20.2 fixes the alpha equation's discretisation - it is\n\
         // explicit, flux-corrected and sub-cycled - so there is deliberately\n\
         // no div(phi,alpha) entry here for a reader to set and be ignored.\n\
         ddtSchemes\n{\n    default         Euler;\n}\n\n\
         gradSchemes\n{\n    default         Gauss linear;\n}\n\n\
         divSchemes\n{\n    default         none;\n\
         // The convecting flux of the two-phase momentum equation is the MASS\n\
         // flux rhoPhi (SPEC-LIT S20.3), and it is not phi.\n\
         \x20   div(rhoPhi,U)    Gauss upwind;\n}\n\n\
         laplacianSchemes\n{\n    default         Gauss linear uncorrected;\n}\n\n\
         interpolationSchemes\n{\n    default         linear;\n}\n\n\
         // The mesh is orthogonal, so `uncorrected` and `corrected` give the\n\
         // same answer and `uncorrected` says so without paying for a pass.\n\
         snGradSchemes\n{\n    default         uncorrected;\n}",
    )?;

    write_dict(
        &dir,
        "fvSolution",
        "solvers\n{\n\
         \x20   // The two-phase pressure, SPEC-LIT S20.5. Diagonally weak\n\
         \x20   // and made weaker by a density ratio of eight hundred, so\n\
         \x20   // it gets DIC and a budget the momentum equation does not\n\
         \x20   // need.\n\
         \x20   p_rgh\n    {\n\
         \x20       solver          PBiCGStab;\n\
         \x20       preconditioner  DIC;\n\
         \x20       tolerance       1e-09;\n\
         \x20       relTol          0.001;\n\
         \x20       maxIter         2000;\n\
         \x20   }\n\n\
         \x20   U\n    {\n\
         \x20       solver          PBiCGStab;\n\
         \x20       preconditioner  diagonal;\n\
         \x20       tolerance       1e-08;\n\
         \x20       relTol          0;\n\
         \x20       maxIter         200;\n\
         \x20   }\n}\n\n\
         // SPEC-LIT S20: `cAlpha` is S20.1, `maxAlphaCo` and\n\
         // `maxAlphaSubCycles` are S20.2's sub-cycling, `nAlphaLimiterIters`\n\
         // is the *DESIGN* iteration count of the Zalesak limiter, and\n\
         // `nCorrectors` is PISO (S5.4).\n\
         PIMPLE\n{\n\
         \x20   momentumPredictor yes;\n\
         \x20   nCorrectors     3;\n\
         \x20   nNonOrthogonalCorrectors 0;\n\
         \x20   cAlpha          1;\n\
         \x20   maxAlphaCo      0.5;\n\
         \x20   maxAlphaSubCycles 100;\n\
         \x20   nAlphaLimiterIters 3;\n\
         }\n\n\
         // A transient run with no outer iterations relaxes nothing.\n\
         relaxationFactors\n{\n    equations\n    {\n\
         \x20       U               1;\n\
         \x20   }\n}",
    )
}

/// `constant/transportProperties` for two immiscible phases - SPEC-LIT S20.3.
///
/// Water and air at 20 C, with `rho` and the DYNAMIC viscosity `mu` per phase
/// rather than a kinematic `nu`: with two fluids there is no single density to
/// divide by, and a kinematic formulation would have to pick one of them and
/// be wrong about the other. [`crate::vof::VofProperties::from_case`] accepts
/// either spelling and converts, but this is the one it is written for.
fn write_two_phase_constant(case_dir: &Path) -> Result<()> {
    write_dict(
        &case_dir.join("constant"),
        "transportProperties",
        "// Two immiscible phases. Phase 1 is the one `alpha.<name>` counts,\n\
         // and it is the first in the list.\n\
         phases          (water air);\n\n\
         water\n{\n\
         \x20   rho             [1 -3 0 0 0 0 0] 998.2;\n\
         \x20   mu              [1 -1 -1 0 0 0 0] 1.002e-03;\n\
         }\n\n\
         air\n{\n\
         \x20   rho             [1 -3 0 0 0 0 0] 1.2;\n\
         \x20   mu              [1 -1 -1 0 0 0 0] 1.8e-05;\n\
         }\n\n\
         // Surface tension, water against air at 20 C.\n\
         sigma           [1 0 -2 0 0 0 0] 0.0728;",
    )
}

// --------------------------------------------------------------------------
//  0/ fields
// --------------------------------------------------------------------------

/// A `PatchFieldSpec` with nothing set but the type.
///
/// Every list stays empty, which under the field-file contract means "the entry
/// is absent from the file"; each caller below fills in only the one its BC
/// actually carries.
fn patch_spec(type_name: &str) -> PatchFieldSpec {
    PatchFieldSpec {
        type_name: type_name.to_string(),
        neighbour_patch: None,
        value: Vec::new(),
        value_v: Vec::new(),
        gradient: Vec::new(),
        gradient_v: Vec::new(),
        ref_value: Vec::new(),
        ref_value_v: Vec::new(),
        ref_gradient: Vec::new(),
        ref_gradient_v: Vec::new(),
        inlet_value: Vec::new(),
        inlet_value_v: Vec::new(),
        value_fraction: Vec::new(),
        extra: std::collections::BTreeMap::new(),
    }
}

/// `0/alpha.water`, `0/U` and `0/p_rgh` for the dam break.
///
/// Written from SPEC-LIT S20 and the geometry of Martin & Moyce (1952): a
/// column of water of width [`DAM_BREAK_A`] and height
/// `DAM_BREAK_ASPECT * DAM_BREAK_A` standing in the bottom-left corner of a
/// tank, released at `t = 0`.
///
/// `alpha` is initialised as a SHARP 0/1 split by cell centre rather than as a
/// cell volume fraction. The column edges land on cell faces exactly, so on
/// this mesh the two agree; the sharp form is used because it makes the initial
/// condition independent of the resolution the case is generated at.
fn write_dam_break_fields(case_dir: &Path, block: &Block) -> Result<()> {
    let g = &block.g;
    let n_cells = g.n_cells();

    let column_w = DAM_BREAK_A;
    let column_h = DAM_BREAK_ASPECT * DAM_BREAK_A;

    // ---- alpha.water -----------------------------------------------------
    //
    // Walls get zeroGradient, which is the ninety-degree contact angle
    // `cuda/vof.cu` documents as this solver's one contact-angle choice: with
    // no wall adhesion model there is no other honest boundary value.
    //
    // The atmosphere gets inletOutlet with an inlet value of 0: whatever
    // leaves, leaves, and anything drawn back in is air. zeroGradient there
    // would let the domain re-inject whatever phase happened to be leaving,
    // which is how a dam break quietly gains water.
    let internal: Vec<Scalar> = (0..n_cells)
        .map(|c| {
            let p = g.cell_centre(c);
            if p.x < column_w && p.y < column_h {
                1.0
            } else {
                0.0
            }
        })
        .collect();

    let mut alpha = RawScalarField {
        name: "alpha.water".to_string(),
        dimensions: "[0 0 0 0 0 0 0]".to_string(),
        internal,
        boundary: BTreeMap::new(),
        boundary_patterns: Vec::new(),
    };

    for patch in &block.patches {
        let pk = PatchKind::from_type(&patch.type_name);
        let s = if pk == PatchKind::Empty {
            patch_spec("empty")
        } else if patch.name == "atmosphere" {
            let mut s = patch_spec("inletOutlet");
            s.inlet_value = vec![0.0];
            s.value = vec![0.0];
            s
        } else {
            patch_spec("zeroGradient")
        };
        alpha.boundary.insert(patch.name.clone(), s);
    }

    write_scalar_field(&case_dir.join("0").join("alpha.water"), &alpha, "0")?;

    // ---- U ---------------------------------------------------------------
    //
    // At rest. Anything else would be an initial condition arguing with the
    // solution: the column is held and released, and a velocity in the file
    // satisfies neither continuity nor the momentum balance.
    let mut u = RawVectorField {
        name: "U".to_string(),
        dimensions: "[0 1 -1 0 0 0 0]".to_string(),
        internal: vec![Vec3::ZERO; n_cells],
        boundary: BTreeMap::new(),
        boundary_patterns: Vec::new(),
    };

    for patch in &block.patches {
        let pk = PatchKind::from_type(&patch.type_name);
        let s = if pk == PatchKind::Empty {
            patch_spec("empty")
        } else if patch.name == "atmosphere" {
            // The velocity at an open boundary next to a prescribed pressure:
            // the flux sets the normal component on inflow and the condition
            // is zero-gradient on outflow.
            let mut s = patch_spec("pressureInletOutletVelocity");
            s.value_v = vec![Vec3::ZERO];
            s
        } else {
            let mut s = patch_spec("noSlip");
            s.value_v = vec![Vec3::ZERO];
            s
        };
        u.boundary.insert(patch.name.clone(), s);
    }

    write_vector_field(&case_dir.join("0").join("U"), &u, "0")?;

    // ---- p_rgh -----------------------------------------------------------
    //
    // `p_rgh = p - rho (g.x)` - SPEC-LIT S20.5 - in Pa, not the kinematic
    // pressure the single-phase cases carry: with two densities there is no
    // one density to divide by.
    //
    // The walls are zeroGradient rather than `fixedFluxPressure`, and the two
    // are the same thing here. A wall's flux is prescribed, so `vof.rs` gives
    // that face no body force and leaves `phi_HbyA` alone there; the pressure
    // equation then never reads a wall gradient at all, and writing a
    // condition whose gradient nothing computes would be a name with no
    // meaning behind it.
    //
    // `atmosphere` gets the one Dirichlet in the case, which is also what
    // makes the pressure matrix non-singular.
    let mut p = RawScalarField {
        name: "p_rgh".to_string(),
        dimensions: "[1 -1 -2 0 0 0 0]".to_string(),
        internal: vec![0.0 as Scalar; n_cells],
        boundary: BTreeMap::new(),
        boundary_patterns: Vec::new(),
    };

    for patch in &block.patches {
        let pk = PatchKind::from_type(&patch.type_name);
        let s = if pk == PatchKind::Empty {
            patch_spec("empty")
        } else if patch.name == "atmosphere" {
            let mut s = patch_spec("fixedValue");
            s.value = vec![0.0];
            s
        } else {
            patch_spec("zeroGradient")
        };
        p.boundary.insert(patch.name.clone(), s);
    }

    write_scalar_field(&case_dir.join("0").join("p_rgh"), &p, "0")
}

/// The initial velocity of one cell.
///
/// A frozen `U` of zero - the honest initial condition for a lid-driven cavity
/// - would leave the model with no shear anywhere except the top row of cells,
/// and the run would be a slow pure-diffusion problem rather than a test of
/// anything. So the cavity gets a single recirculating cell from the stream
/// function
///
/// ```text
/// psi = A sin(pi x/Lx) sin(pi y/Ly),   U = (dpsi/dy, -dpsi/dx)
/// ```
///
/// which is analytically divergence free and has zero normal velocity on all
/// four walls. Everything else gets a 1/7 power law in the wall-normal
/// direction, which gives the turbulence model a realistic shear to work on
/// without needing a momentum solver.
///
/// `lo`, `lx` and `ly` are the bounding box of the CELL CENTRES, not of the
/// points. That is what the C++ measures, and it is what makes the stream
/// function vanish exactly on the first and last cell centre.
#[allow(clippy::too_many_arguments)]
fn initial_velocity(
    cavity: bool,
    c: Vec3,
    lo: Vec3,
    lx: Scalar,
    ly: Scalar,
    u_ref: Scalar,
    half_height: Scalar,
    wall_normal: usize,
) -> Vec3 {
    const PI: f64 = std::f64::consts::PI;

    if cavity {
        let x = ((c.x - lo.x) / lx) as f64;
        let y = ((c.y - lo.y) / ly) as f64;
        let a = u_ref * lx / (PI as Scalar);

        Vec3::new(
            a * (PI as Scalar) / ly * ((PI * x).sin() as Scalar) * ((PI * y).cos() as Scalar),
            -a * (PI as Scalar) / lx * ((PI * x).cos() as Scalar) * ((PI * y).sin() as Scalar),
            0.0,
        )
    } else {
        // The clamp keeps the wall value just inside the profile: pow(0, 1/7)
        // is exactly zero, and a first cell with no velocity at all gives the
        // wall functions nothing to key off.
        let axis = c.component(wall_normal);
        let yy = (1.0 as Scalar - 1e-9).min((axis / half_height).abs());
        let u = u_ref * ((1.0 - yy as f64).powf(1.0 / 7.0) as Scalar);
        Vec3::new(u, 0.0, 0.0)
    }
}

/// Bounding box of the CELL CENTRES, from the 1-D node arrays.
///
/// Scanning the per-axis centres is exactly the min/max over the full 3-D set
/// because the block is rectilinear, and it avoids materialising an
/// `n_cells`-long centre array just to take six extrema.
fn centre_bounds(g: &Grid) -> (Vec3, Vec3) {
    let extrema = |n: &[Scalar], m: usize| {
        let mut lo = 0.5 * (n[0] + n[1]);
        let mut hi = lo;
        for i in 1..m {
            let c = 0.5 * (n[i] + n[i + 1]);
            lo = lo.min(c);
            hi = hi.max(c);
        }
        (lo, hi)
    };

    let (xlo, xhi) = extrema(&g.xn, g.nx);
    let (ylo, yhi) = extrema(&g.yn, g.ny);
    let (zlo, zhi) = extrema(&g.zn, g.nz);

    (Vec3::new(xlo, ylo, zlo), Vec3::new(xhi, yhi, zhi))
}

#[allow(clippy::too_many_arguments)]
fn write_initial_fields(
    case_dir: &Path,
    kind: CaseKind,
    block: &Block,
    nu: Scalar,
    u_ref: Scalar,
    half_height: Scalar,
    wall_normal: usize,
) -> Result<()> {
    let g = &block.g;
    let n_cells = g.n_cells();

    // Turbulence intensity 5 %, mixing length 7 % of the half height: the
    // standard OpenFOAM inlet estimate.
    let i_turb = 0.05 as Scalar;
    let cmu = 0.09 as Scalar;
    let k0 = 1.5 * (i_turb * u_ref) * (i_turb * u_ref);
    let l = 0.07 * half_height * 2.0;
    let eps0 = cmu.powf(0.75) * k0.powf(1.5) / l;
    let omega0 = k0.sqrt() / (cmu.powf(0.25) * l);

    let cavity = kind == CaseKind::Cavity;
    let plume = kind == CaseKind::Plume;

    let (lo, hi) = centre_bounds(g);
    let lx = (hi.x - lo.x).max(1e-30);
    let ly = (hi.y - lo.y).max(1e-30);

    // ---- U ---------------------------------------------------------------
    //
    // The plume starts from rest. It used to start from a prescribed column of
    // rising air over the burner, because the velocity was FROZEN and a plume
    // with no jet in the initial file was a plume with no jet at all.
    // `ofgpu-buoyant` solves the momentum equation, so the same column would be
    // an initial condition arguing with the solution: it satisfies neither
    // continuity nor the momentum balance, and the first pressure correction
    // spends itself undoing it. Zero satisfies continuity exactly.
    let mut internal = Vec::with_capacity(n_cells);
    for c in 0..n_cells {
        let centre = g.cell_centre(c);

        internal.push(if plume {
            Vec3::ZERO
        } else {
            initial_velocity(cavity, centre, lo, lx, ly, u_ref, half_height, wall_normal)
        });
    }

    let mut u = RawVectorField {
        name: "U".to_string(),
        dimensions: "[0 1 -1 0 0 0 0]".to_string(),
        internal,
        boundary: BTreeMap::new(),
        boundary_patterns: Vec::new(),
    };

    for patch in &block.patches {
        let name = patch.name.as_str();
        let pk = PatchKind::from_type(&patch.type_name);

        let s = if pk == PatchKind::Empty {
            patch_spec("empty")
        } else if cavity && name == "movingWall" {
            let mut s = patch_spec("fixedValue");
            s.value_v = vec![Vec3::new(u_ref, 0.0, 0.0)];
            s
        } else if pk == PatchKind::Wall {
            let mut s = patch_spec("noSlip");
            s.value_v = vec![Vec3::ZERO];
            s
        } else if name == "inlet" {
            let mut s = patch_spec("fixedValue");
            s.value_v = if plume {
                // A flat top hat: the taper is in z and this patch is at
                // z = 0, where the jet is at full strength.
                vec![Vec3::new(0.0, 0.0, u_ref)]
            } else {
                // The inlet carries the same profile as the interior, so the
                // solution does not have to develop one from a top hat.
                let (na, _) = slot_dims(g, patch.slot);
                (0..patch.size)
                    .map(|idx| {
                        let q = boundary_quad(g, patch.slot, patch.slot_index(na, idx));
                        u.internal[q.own]
                    })
                    .collect()
            };
            s
        } else if plume {
            // The one opening, at +x. Whatever leaves, leaves, and anything
            // that momentarily reverses comes back in at rest - which is what
            // inletOutlet says; zeroGradient would instead let the domain draw
            // in momentum it invented itself.
            let mut s = patch_spec("inletOutlet");
            s.inlet_value_v = vec![Vec3::ZERO];
            s.value_v = vec![Vec3::ZERO];
            s
        } else {
            patch_spec("zeroGradient")
        };

        u.boundary.insert(name.to_string(), s);
    }

    write_vector_field(&case_dir.join("0").join("U"), &u, "0")?;

    // ---- p ---------------------------------------------------------------
    //
    // Kinematic pressure, `[0 2 -2 0 0 0 0]` - p/rho, which is what a
    // constant-density incompressible solver carries and what `simple.rs`
    // assembles. Only the plume gets one: it is the only case here with a
    // momentum equation to solve, and the others' drivers hold `U` frozen and
    // would read a `0/p` nothing writes back.
    //
    // `fixedValue 0` on the single opening and `zeroGradient` everywhere else
    // is the whole boundary specification, and it is also what makes the
    // pressure matrix non-singular: with every patch Neumann the constant is a
    // null space and `Simple::initialise` would have to pin a cell to remove
    // it. One Dirichlet face is cheaper and physical - it is the level the room
    // is open to.
    if plume {
        let mut pf = RawScalarField {
            name: "p".to_string(),
            dimensions: "[0 2 -2 0 0 0 0]".to_string(),
            internal: vec![0.0 as Scalar; n_cells],
            boundary: BTreeMap::new(),
            boundary_patterns: Vec::new(),
        };

        for patch in &block.patches {
            let pk = PatchKind::from_type(&patch.type_name);

            let s = if pk == PatchKind::Empty {
                patch_spec("empty")
            } else if patch.name == "outlet" {
                let mut s = patch_spec("fixedValue");
                s.value = vec![0.0];
                s
            } else {
                // Walls and the burner alike. A wall imposes no pressure, and
                // neither does an inlet whose velocity is prescribed: fixing
                // both `U` and `p` on the same face over-specifies the face.
                patch_spec("zeroGradient")
            };

            pf.boundary.insert(patch.name.clone(), s);
        }

        write_scalar_field(&case_dir.join("0").join("p"), &pf, "0")?;
    }

    // ---- T ---------------------------------------------------------------
    //
    // Only the plume has one. The other cases are isothermal, and a `0/T`
    // there would be a field nothing solves and nothing reads.
    if plume {
        let mut t = RawScalarField {
            name: "T".to_string(),
            dimensions: "[0 0 0 1 0 0 0]".to_string(),
            internal: vec![PLUME_T_AMBIENT; n_cells],
            boundary: BTreeMap::new(),
            boundary_patterns: Vec::new(),
        };

        for patch in &block.patches {
            let pk = PatchKind::from_type(&patch.type_name);

            let s = if pk == PatchKind::Empty {
                patch_spec("empty")
            } else if patch.name == "inlet" {
                let mut s = patch_spec("fixedValue");
                s.value = vec![PLUME_T_INLET];
                s
            } else if pk == PatchKind::Wall {
                // Adiabatic floor and ceiling: the case is about the plume, not
                // about how much heat the room absorbs.
                patch_spec("zeroGradient")
            } else {
                let mut s = patch_spec("inletOutlet");
                s.inlet_value = vec![PLUME_T_AMBIENT];
                s.value = vec![PLUME_T_AMBIENT];
                s
            };

            t.boundary.insert(patch.name.clone(), s);
        }

        write_scalar_field(&case_dir.join("0").join("T"), &t, "0")?;
    }

    // ---- scalars ---------------------------------------------------------
    let specs: [(&str, &str, Scalar, &str); 4] = [
        ("k", "[0 2 -2 0 0 0 0]", k0, "kqRWallFunction"),
        ("epsilon", "[0 2 -3 0 0 0 0]", eps0, "epsilonWallFunction"),
        ("omega", "[0 0 -1 0 0 0 0]", omega0, "omegaWallFunction"),
        ("nut", "[0 2 -1 0 0 0 0]", 0.0, "nutkWallFunction"),
    ];

    for (name, dims, value, wall_type) in specs {
        let mut f = RawScalarField {
            name: name.to_string(),
            dimensions: dims.to_string(),
            internal: vec![value; n_cells],
            boundary: BTreeMap::new(),
            boundary_patterns: Vec::new(),
        };

        for patch in &block.patches {
            let pname = patch.name.as_str();
            let pk = PatchKind::from_type(&patch.type_name);

            let s = if pk == PatchKind::Empty {
                patch_spec("empty")
            } else if pk == PatchKind::Wall {
                let mut s = patch_spec(wall_type);
                s.value = vec![value];
                s
            } else if name == "nut" {
                // nut is never prescribed on a non-wall patch - the model
                // evaluates it there. Checked before the inlet branch on
                // purpose: fixing nut = 0 at an inlet would silently delete the
                // turbulent diffusivity of everything entering.
                let mut s = patch_spec("calculated");
                s.value = vec![0.0];
                s
            } else if pname == "inlet" {
                let mut s = patch_spec("fixedValue");
                s.value = vec![value];
                s
            } else {
                patch_spec("zeroGradient")
            };

            f.boundary.insert(pname.to_string(), s);
        }

        write_scalar_field(&case_dir.join("0").join(name), &f, "0")?;
    }

    println!(
        "  0/ fields: Uref {}  k {}  epsilon {}  omega {}  (nu {})",
        fmt_g(u_ref),
        fmt_g(k0),
        fmt_g(eps0),
        fmt_g(omega0),
        fmt_g(nu)
    );

    Ok(())
}

// ==========================================================================
//  Tests
// ==========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    /// Tolerances are expressed in units of `Scalar::EPSILON` so the same
    /// assertions still mean something under the `single` feature.
    const EPS: Scalar = Scalar::EPSILON;

    fn axis(lo: Scalar, hi: Scalar, n: usize, expansion: Scalar, two_sided: bool) -> GradedAxis {
        GradedAxis { lo, hi, n, expansion, two_sided }
    }

    /// A block that is graded on two axes and uniform on the third, so nothing
    /// below can pass by accident on a cube of identical cells.
    fn spec(nx: usize, ny: usize, nz: usize) -> BlockSpec {
        BlockSpec {
            x: axis(0.0, 2.0, nx, 3.0, false),
            y: axis(-1.0, 1.0, ny, 5.0, true),
            z: axis(0.0, 0.5, nz, 1.0, false),
            ..BlockSpec::default()
        }
    }

    /// The same block with `zMin` split by a window that is not flush with any
    /// edge of the patch, so every branch of `OutPatch::slot_index` is live.
    fn split_spec(nx: usize, ny: usize, nz: usize) -> BlockSpec {
        let mut b = spec(nx, ny, nz);
        b.patch_name[4] = "floor".to_string();
        b.patch_type[4] = "wall".to_string();
        b.window = Some(PatchWindow {
            slot: 4,
            lo: [1, 1],
            hi: [3, 3],
            name: "burner".to_string(),
            type_name: "patch".to_string(),
        });
        b
    }

    fn starts(block: &Block) -> Vec<usize> {
        block.patches.iter().map(|p| p.start).collect()
    }

    fn sizes(block: &Block) -> Vec<usize> {
        block.patches.iter().map(|p| p.size).collect()
    }

    /// Every patch is one unbroken `[startFace, startFace + nFaces)` run, they
    /// begin where the internal faces end, and together they reach exactly the
    /// last face. OpenFOAM records nothing else about where a patch lives, so
    /// a gap or an overlap here is a mesh that reads back as a different one.
    fn assert_patches_are_contiguous(block: &Block) {
        assert_eq!(block.patches[0].start, block.n_internal);

        for i in 1..block.patches.len() {
            assert_eq!(
                block.patches[i].start,
                block.patches[i - 1].start + block.patches[i - 1].size,
                "patch {i} does not begin where patch {} ends",
                i - 1
            );
        }

        let last = &block.patches[block.patches.len() - 1];
        assert_eq!(last.start + last.size, block.n_faces());
    }

    // ---- headers ---------------------------------------------------------

    /// The banner is matched character for character against the reference
    /// cases; a miscounted dash or asterisk makes every generated file differ
    /// from the C++ output for no visible reason.
    #[test]
    fn the_banner_is_a_well_formed_box_that_claims_nothing_it_should_not() {
        let lines: Vec<&str> = BANNER.lines().collect();

        // Eight lines of box, then the fixed head of the FoamFile entry. The
        // box is 79 columns because that is what the format's readers and
        // every diff tool expect, not because anything parses it.
        for (i, l) in lines.iter().take(8).enumerate() {
            assert_eq!(l.chars().count(), 79, "banner line {} is not 79 wide", i + 1);
        }
        assert!(lines[0].starts_with("/*-") && lines[0].ends_with("-*") == false);
        assert!(lines[7].starts_with(r"\*-"), "the box must close");
        for l in &lines[1..7] {
            assert!(l.starts_with('|') && l.ends_with('|'), "box line: {l}");
        }

        assert_eq!(lines[8], "FoamFile");
        assert_eq!(lines[9], "{");
        assert!(BANNER.ends_with("    class       "));

        // The banner ofgpu writes is ofgpu's own. It names the format, because
        // that is what the file is in, and it says plainly who wrote the file.
        // What it must never do is carry the upstream project's own banner
        // text, which is both copied and false about authorship.
        assert!(BANNER.contains("ofgpu"));
        assert!(!BANNER.contains("The Open Source CFD Toolbox"));
        assert!(!BANNER.contains("openfoam.org"));
        assert!(!BANNER.contains("O peration"));

        assert_eq!(SEPARATOR.trim_end().chars().count(), 79);
        assert_eq!(SEPARATOR.matches('*').count(), 37);
        assert_eq!(FOOTER_RULE.trim_end().chars().count(), 79);
        assert_eq!(FOOTER_RULE.matches('*').count(), 73);
    }

    /// OpenFOAM writes `nu [0 2 -1 0 0 0 0] 1e-05`; Rust's own float Display
    /// would write `0.00001` and make every case differ from the reference.
    #[test]
    fn dimensioned_constants_format_like_printf_g() {
        assert_eq!(fmt_g(1e-5), "1e-05");
        assert_eq!(fmt_g(0.0), "0");
        assert_eq!(fmt_g(1.0), "1");
        assert_eq!(fmt_g(10.0), "10");
        assert_eq!(fmt_g(0.00375), "0.00375");
        assert_eq!(fmt_g(0.05), "0.05");
        assert_eq!(fmt_g(1.0e7), "1e+07");
        assert_eq!(fmt_g(-1e-5), "-1e-05");
    }

    /// The point coordinates are written at 15 significant digits, and these
    /// are literally the values the reference `cases/` files contain. Under the
    /// `single` feature the literals below are not representable, so the check
    /// only means something in double precision.
    #[test]
    #[cfg(not(feature = "single"))]
    fn point_coordinates_match_the_reference_precision() {
        assert_eq!(fmt_g_prec(0.0023437500000000003, 15), "0.00234375");
        assert_eq!(fmt_g_prec(-0.9974011081597444, 15), "-0.997401108159744");
        assert_eq!(fmt_g_prec(-0.9946668497192848, 15), "-0.994666849719285");
        assert_eq!(fmt_g_prec(0.0, 15), "0");
        assert_eq!(fmt_g_prec(4.0, 15), "4");
        assert_eq!(fmt_g_prec(0.02, 15), "0.02");
    }

    // ---- grading ---------------------------------------------------------

    #[test]
    fn graded_nodes_pin_the_endpoints() {
        for a in [
            axis(0.0, 4.0, 7, 1.0, false),
            axis(-1.0, 1.0, 120, 20.0, true),
            axis(0.0, 30.0, 300, 4.0, false),
            axis(3.0, 3.5, 1, 9.0, false),
            axis(0.0, 1.0, 2, 8.0, true),
        ] {
            let v = graded_nodes(&a);
            assert_eq!(v.len(), a.n + 1);
            assert_eq!(v[0], a.lo, "lo is not exact");
            assert_eq!(v[a.n], a.hi, "hi is not exact");
            for i in 1..v.len() {
                assert!(v[i] > v[i - 1], "nodes are not increasing at {i}");
            }
        }
    }

    #[test]
    fn one_sided_expansion_is_last_cell_over_first() {
        let a = axis(0.0, 30.0, 300, 4.0, false);
        let v = graded_nodes(&a);
        let ratio = (v[a.n] - v[a.n - 1]) / (v[1] - v[0]);
        assert!((ratio - 4.0).abs() < 1e4 * EPS, "ratio {ratio}");
    }

    #[test]
    fn two_sided_grading_is_symmetric_and_hits_the_ratio() {
        let a = axis(-1.0, 1.0, 120, 20.0, true);
        let v = graded_nodes(&a);

        for i in 0..=a.n {
            let l = v[i] - a.lo;
            let r = a.hi - v[a.n - i];
            assert!((l - r).abs() < 1e4 * EPS, "asymmetric at {i}: {l} vs {r}");
        }

        let mut smallest = Scalar::MAX;
        let mut largest = 0.0 as Scalar;
        for i in 0..a.n {
            let d = v[i + 1] - v[i];
            smallest = smallest.min(d);
            largest = largest.max(d);
        }
        let ratio = largest / smallest;
        assert!((ratio - 20.0).abs() < 1e5 * EPS, "ratio {ratio}");
    }

    /// Two-sided with fewer than three cells has nowhere to grade; taking the
    /// geometric branch there would divide by `r^0 - 1 == 0`.
    #[test]
    fn two_sided_degenerates_to_uniform_below_three_cells() {
        let v = graded_nodes(&axis(0.0, 1.0, 2, 8.0, true));
        assert!(v.iter().all(|x| x.is_finite()), "{v:?}");
        assert!((v[1] - 0.5).abs() < 1e3 * EPS, "{v:?}");
    }

    #[test]
    fn a_unit_expansion_is_exactly_uniform() {
        let a = axis(0.0, 1.0, 8, 1.0, false);
        let v = graded_nodes(&a);
        for i in 0..=a.n {
            assert!((v[i] - i as Scalar / 8.0).abs() < 1e3 * EPS);
        }
    }

    // ---- topology --------------------------------------------------------

    #[test]
    fn cell_index_runs_with_i_fastest() {
        let block = Block::new(&spec(3, 4, 2)).expect("block");
        let g = &block.g;

        assert_eq!(g.cell(0, 0, 0), 0);
        assert_eq!(g.cell(1, 0, 0), 1);
        assert_eq!(g.cell(0, 1, 0), 3);
        assert_eq!(g.cell(0, 0, 1), 12);
        assert_eq!(g.point(1, 0, 0), 1);
        assert_eq!(g.point(0, 1, 0), 4);

        for c in 0..block.n_cells {
            let (i, j, k) = g.decompose_cell(c);
            assert_eq!(g.cell(i, j, k), c);
        }
    }

    #[test]
    fn internal_faces_are_upper_triangular() {
        let block = Block::new(&spec(4, 3, 3)).expect("block");
        assert_eq!(block.faces.len(), block.n_internal);

        let mut prev = (0usize, 0usize);
        for (f, face) in block.faces.iter().enumerate() {
            assert!(face.own < face.nei, "face {f} has owner >= neighbour");
            if f > 0 {
                assert!(prev < (face.own, face.nei), "face {f} is out of order");
            }
            prev = (face.own, face.nei);
        }
    }

    #[test]
    fn internal_face_area_points_from_owner_to_neighbour() {
        let block = Block::new(&spec(4, 5, 3)).expect("block");
        let g = &block.g;

        for (f, face) in block.faces.iter().enumerate() {
            let sf = quad_area(g, &internal_quad(g, *face));
            let d = g.cell_centre(face.nei) - g.cell_centre(face.own);
            assert!(sf.dot(d) > 0.0, "internal face {f} is wound the wrong way");
        }
    }

    #[test]
    fn boundary_face_area_points_out_of_the_domain() {
        let block = Block::new(&spec(4, 5, 3)).expect("block");
        let g = &block.g;

        for p in 0..6 {
            for idx in 0..block.patch_size[p] {
                let q = boundary_quad(g, p, idx);
                let sf = quad_area(g, &q);
                let d = quad_centre(g, &q) - g.cell_centre(q.own);
                assert!(sf.dot(d) > 0.0, "patch {p} face {idx} is wound inward");
            }
        }
    }

    /// Every cell must see each of its six faces exactly once, whether as an
    /// internal face or through a patch. A patch that overlapped another, or
    /// that skipped a row, would still produce a file OpenFOAM reads.
    #[test]
    fn patches_tile_the_boundary_exactly_once() {
        let block = Block::new(&spec(4, 5, 3)).expect("block");
        let g = &block.g;

        // slot = 2*dir + 0 for the low side, +1 for the high side, which is
        // also the patch numbering.
        let mut seen = vec![0u8; 6 * block.n_cells];

        for face in &block.faces {
            let d = face.dir as usize;
            seen[6 * face.own + 2 * d + 1] += 1;
            seen[6 * face.nei + 2 * d] += 1;
        }

        for p in 0..6 {
            for idx in 0..block.patch_size[p] {
                seen[6 * boundary_quad(g, p, idx).own + p] += 1;
            }
        }

        for (s, n) in seen.iter().enumerate() {
            assert_eq!(*n, 1, "cell {} slot {} is covered {} times", s / 6, s % 6, n);
        }

        assert_patches_are_contiguous(&block);
        assert_eq!(
            block.n_boundary,
            2 * (g.ny * g.nz + g.nx * g.nz + g.nx * g.ny)
        );
    }

    /// `sum_f s_f Sf` over a closed cell is zero to round-off. This is the one
    /// check that catches a sign error, a missing face and a duplicated face at
    /// the same time.
    #[test]
    fn every_cell_closes() {
        let block = Block::new(&spec(4, 5, 3)).expect("block");
        let g = &block.g;

        let mut sum = vec![Vec3::ZERO; block.n_cells];

        for face in &block.faces {
            let sf = quad_area(g, &internal_quad(g, *face));
            sum[face.own] += sf;
            sum[face.nei] -= sf;
        }

        for p in 0..6 {
            for idx in 0..block.patch_size[p] {
                let q = boundary_quad(g, p, idx);
                sum[q.own] += quad_area(g, &q);
            }
        }

        for (c, s) in sum.iter().enumerate() {
            assert!(s.mag() < 1e4 * EPS, "cell {c} does not close: {s}");
        }
    }

    // ---- written files ---------------------------------------------------

    fn temp_dir(tag: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!("ofgpu_blockgen_{}_{tag}", std::process::id()));
        let _ = fs::remove_dir_all(&p);
        p
    }

    /// Pull the `( ... )` list body out of a polyMesh file as integers. The
    /// header never contains a parenthesis, so the first one opens the list.
    fn read_list(path: &Path) -> Vec<usize> {
        let src = fs::read_to_string(path).expect("read");
        let open = src.find('(').expect("list open");
        let close = src.rfind(')').expect("list close");
        src[open + 1..close]
            .split_whitespace()
            .map(|t| t.parse::<usize>().expect("integer"))
            .collect()
    }

    /// The patch names of a `boundary` file, in file order. The header holds
    /// no parenthesis, so the first one opens the list.
    fn boundary_patch_names(src: &str) -> Vec<String> {
        let open = src.find('(').expect("list open");
        let close = src.rfind(')').expect("list close");
        src[open + 1..close]
            .lines()
            .map(str::trim)
            .filter(|l| !l.is_empty() && !l.contains(';') && *l != "{" && *l != "}")
            .map(String::from)
            .collect()
    }

    fn dict_values(src: &str, key: &str) -> Vec<usize> {
        src.lines()
            .filter_map(|l| l.trim().strip_prefix(key))
            .map(|v| v.trim().trim_end_matches(';').parse().expect("integer"))
            .collect()
    }

    #[test]
    fn written_polymesh_keeps_the_ordering_invariants() {
        let dir = temp_dir("polymesh");
        let b = spec(3, 4, 2);
        write_block_mesh(&dir, &b).expect("write");

        let poly = dir.join("constant").join("polyMesh");
        let owner = read_list(&poly.join("owner"));
        let neighbour = read_list(&poly.join("neighbour"));

        let block = Block::new(&b).expect("block");
        assert_eq!(owner.len(), block.n_faces());
        assert_eq!(neighbour.len(), block.n_internal);

        for f in 0..neighbour.len() {
            assert!(owner[f] < neighbour[f], "face {f}: owner >= neighbour");
            if f > 0 {
                assert!(
                    (owner[f - 1], neighbour[f - 1]) < (owner[f], neighbour[f]),
                    "face {f} breaks the upper-triangular order"
                );
            }
        }

        // Every boundary owner sits after the last internal face and names a
        // real cell.
        for own in owner.iter().skip(neighbour.len()) {
            assert!(*own < block.n_cells);
        }

        let boundary = fs::read_to_string(poly.join("boundary")).expect("read boundary");
        assert_eq!(dict_values(&boundary, "startFace"), starts(&block));
        assert_eq!(dict_values(&boundary, "nFaces"), sizes(&block));

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn faces_file_has_one_quad_per_face_in_the_written_order() {
        let dir = temp_dir("faces");
        let b = spec(2, 3, 2);
        write_block_mesh(&dir, &b).expect("write");

        let block = Block::new(&b).expect("block");
        let src = fs::read_to_string(dir.join("constant").join("polyMesh").join("faces"))
            .expect("read faces");
        let quads: Vec<&str> = src.lines().filter(|l| l.starts_with("4(")).collect();
        assert_eq!(quads.len(), block.n_faces());

        let first = internal_quad(&block.g, block.faces[0]);
        assert_eq!(
            quads[0],
            format!("4({} {} {} {})", first.p[0], first.p[1], first.p[2], first.p[3])
        );

        let last = boundary_quad(&block.g, 5, block.patch_size[5] - 1);
        assert_eq!(
            quads[block.n_faces() - 1],
            format!("4({} {} {} {})", last.p[0], last.p[1], last.p[2], last.p[3])
        );

        let _ = fs::remove_dir_all(&dir);
    }

    // ---- cases -----------------------------------------------------------

    #[test]
    fn empty_patches_become_walls_once_the_block_is_three_dimensional() {
        let two_d = case_block_spec(CaseKind::Cavity, 8, 8, 1);
        assert_eq!(two_d.patch_type[4], "empty");
        assert_eq!(two_d.patch_type[5], "empty");

        let three_d = case_block_spec(CaseKind::Cavity, 8, 8, 4);
        assert_eq!(three_d.patch_type[4], "wall");
        assert_eq!(three_d.patch_type[5], "wall");
        // The four side walls are untouched.
        assert_eq!(three_d.patch_type[0], "wall");
        assert_eq!(three_d.patch_name[3], "movingWall");
    }

    /// The stream function is only a legal initial condition if there is no
    /// flow through the four walls; a sign slip in either component breaks that
    /// silently, and the cavity leaks mass from the first iteration.
    #[test]
    fn the_cavity_stream_function_has_no_wall_normal_velocity() {
        let b = case_block_spec(CaseKind::Cavity, 16, 16, 1);
        let g = Grid::new(&b).expect("grid");
        let (lo, hi) = centre_bounds(&g);
        let lx = hi.x - lo.x;
        let ly = hi.y - lo.y;

        let at = |c: Vec3| initial_velocity(true, c, lo, lx, ly, 1.0, 0.05, 1);

        // x = lo.x and x = hi.x are the left and right walls: U.x must vanish.
        for j in 0..g.ny {
            let yc = 0.5 * (g.yn[j] + g.yn[j + 1]);
            assert!(at(Vec3::new(lo.x, yc, 0.0)).x.abs() < 1e3 * EPS);
            assert!(at(Vec3::new(hi.x, yc, 0.0)).x.abs() < 1e3 * EPS);
        }

        // y = lo.y and y = hi.y are the floor and the lid: U.y must vanish.
        for i in 0..g.nx {
            let xc = 0.5 * (g.xn[i] + g.xn[i + 1]);
            assert!(at(Vec3::new(xc, lo.y, 0.0)).y.abs() < 1e3 * EPS);
            assert!(at(Vec3::new(xc, hi.y, 0.0)).y.abs() < 1e3 * EPS);
        }
    }

    #[test]
    fn the_power_law_peaks_on_the_centreline_and_decays_to_the_wall() {
        let at = |y: Scalar| {
            initial_velocity(false, Vec3::new(0.0, y, 0.0), Vec3::ZERO, 1.0, 1.0, 2.0, 1.0, 1).x
        };

        assert!((at(0.0) - 2.0).abs() < 1e3 * EPS, "centreline {}", at(0.0));

        // The wall value is small but deliberately not zero - see the clamp in
        // `initial_velocity`.
        assert!(at(1.0) > 0.0 && at(1.0) < 0.1 * at(0.0), "wall {}", at(1.0));
        assert_eq!(at(-1.0), at(1.0), "profile is not mirrored");

        let mut prev = at(0.0);
        for n in 1..=20 {
            let u = at(n as Scalar / 20.0);
            assert!(u <= prev, "profile is not monotone at {n}");
            prev = u;
        }
    }


    // ---- split patches ---------------------------------------------------

    /// The two halves of a split slot must between them hit every face of that
    /// slot exactly once. If they did not, the boundary file would still be
    /// well formed and the mesh would still load - it would just be a
    /// different mesh from the one that was asked for.
    #[test]
    fn a_split_slot_is_partitioned_between_its_two_patches() {
        let block = Block::new(&split_spec(5, 6, 3)).expect("block");
        let g = &block.g;

        assert_eq!(block.patches.len(), 7);

        let (na, nb) = slot_dims(g, 4);
        let mut seen = vec![0u8; na * nb];

        for patch in block.patches.iter().filter(|p| p.slot == 4) {
            for idx in 0..patch.size {
                seen[patch.slot_index(na, idx)] += 1;
            }
        }

        for (i, n) in seen.iter().enumerate() {
            assert_eq!(*n, 1, "slot face {i} is covered {n} times");
        }
    }

    /// The window's faces are the ones inside it and nothing else, and they
    /// come out in the patch's own index order.
    #[test]
    fn the_window_patch_holds_exactly_the_windowed_faces() {
        let b = split_spec(5, 6, 3);
        let block = Block::new(&b).expect("block");
        let g = &block.g;
        let w = b.window.as_ref().expect("window");

        let inlet = &block.patches[4];
        assert_eq!(inlet.name, "burner");
        assert_eq!(inlet.part, SlotPart::Window);
        assert_eq!(inlet.size, 4);

        let (na, _) = slot_dims(g, 4);
        for idx in 0..inlet.size {
            let si = inlet.slot_index(na, idx);
            let (i, j) = (si % na, si / na);
            assert!(i >= w.lo[0] && i < w.hi[0], "face {idx} is outside in i");
            assert!(j >= w.lo[1] && j < w.hi[1], "face {idx} is outside in j");
        }

        let floor = &block.patches[5];
        assert_eq!(floor.name, "floor");
        assert_eq!(floor.part, SlotPart::Rest);
        assert_eq!(floor.size, block.patch_size[4] - inlet.size);

        for idx in 0..floor.size {
            let si = floor.slot_index(na, idx);
            let (i, j) = (si % na, si / na);
            assert!(
                !(i >= w.lo[0] && i < w.hi[0] && j >= w.lo[1] && j < w.hi[1]),
                "floor face {idx} is inside the window"
            );
        }
    }

    /// A window whose fast direction spans the whole slot leaves the middle
    /// run of `Rest` empty - the one arithmetic case that would divide by zero
    /// if it were reached.
    #[test]
    fn a_full_width_window_leaves_no_punched_rows() {
        let mut b = spec(4, 5, 3);
        b.window = Some(PatchWindow {
            slot: 4,
            lo: [0, 1],
            hi: [4, 3],
            name: "strip".to_string(),
            type_name: "patch".to_string(),
        });

        let block = Block::new(&b).expect("block");
        let (na, nb) = slot_dims(&block.g, 4);

        let mut seen = vec![0u8; na * nb];
        for patch in block.patches.iter().filter(|p| p.slot == 4) {
            for idx in 0..patch.size {
                seen[patch.slot_index(na, idx)] += 1;
            }
        }
        assert!(seen.iter().all(|n| *n == 1), "{seen:?}");
    }

    #[test]
    fn a_split_block_still_tiles_the_boundary_exactly_once() {
        let block = Block::new(&split_spec(5, 6, 3)).expect("block");
        let g = &block.g;

        let mut seen = vec![0u8; 6 * block.n_cells];

        for face in &block.faces {
            let d = face.dir as usize;
            seen[6 * face.own + 2 * d + 1] += 1;
            seen[6 * face.nei + 2 * d] += 1;
        }

        let mut sum = vec![Vec3::ZERO; block.n_cells];
        for face in &block.faces {
            let sf = quad_area(g, &internal_quad(g, *face));
            sum[face.own] += sf;
            sum[face.nei] -= sf;
        }

        for patch in &block.patches {
            let (na, _) = slot_dims(g, patch.slot);
            for idx in 0..patch.size {
                let q = boundary_quad(g, patch.slot, patch.slot_index(na, idx));
                seen[6 * q.own + patch.slot] += 1;
                sum[q.own] += quad_area(g, &q);
                assert!(winding_ok(g, &q), "{} face {idx} is wound inward", patch.name);
            }
        }

        for (i, n) in seen.iter().enumerate() {
            assert_eq!(*n, 1, "cell {} slot {} is covered {} times", i / 6, i % 6, n);
        }
        for (c, v) in sum.iter().enumerate() {
            assert!(v.mag() < 1e4 * EPS, "cell {c} does not close: {v}");
        }

        assert_patches_are_contiguous(&block);
    }

    #[test]
    fn a_window_that_swallows_its_slot_is_refused() {
        let mut b = spec(4, 5, 3);
        b.window = Some(PatchWindow {
            slot: 4,
            lo: [0, 0],
            hi: [4, 5],
            name: "all".to_string(),
            type_name: "patch".to_string(),
        });
        assert!(Block::new(&b).is_err());

        // Out of range, empty, and colliding with the host patch's name.
        for (lo, hi, name) in [
            ([0, 0], [5, 2], "w"),
            ([2, 2], [2, 3], "w"),
            ([0, 0], [2, 2], "zMin"),
        ] {
            let mut b = spec(4, 5, 3);
            b.window = Some(PatchWindow {
                slot: 4,
                lo,
                hi,
                name: name.to_string(),
                type_name: "patch".to_string(),
            });
            assert!(Block::new(&b).is_err(), "accepted {lo:?}..{hi:?} '{name}'");
        }
    }

    // ---- the plume case --------------------------------------------------

    #[test]
    fn only_the_plume_splits_a_patch() {
        for k in [CaseKind::Channel, CaseKind::Cavity, CaseKind::Step, CaseKind::Big] {
            let (nx, ny, nz) = k.default_resolution();
            assert!(
                case_block_spec(k, nx, ny, nz).window.is_none(),
                "{} grew a window",
                k.as_str()
            );
        }

        assert_eq!(CaseKind::from_name("plume"), Some(CaseKind::Plume));
        assert_eq!(CaseKind::Plume.as_str(), "plume");
        assert_eq!(CaseKind::Plume.default_resolution(), (98, 42, 20));
    }

    #[test]
    fn the_plume_box_is_the_published_benchmark_geometry() {
        let (nx, ny, nz) = CaseKind::Plume.default_resolution();
        let b = case_block_spec(CaseKind::Plume, nx, ny, nz);

        assert_eq!((b.x.lo, b.x.hi), (-8.32, 6.32));
        assert_eq!((b.y.lo, b.y.hi), (-2.62, 3.62));
        assert_eq!((b.z.lo, b.z.hi), (0.0, 3.0));
        for a in [&b.x, &b.y, &b.z] {
            assert_eq!(a.expansion, 1.0, "the plume grid is uniform");
            assert!(!a.two_sided);
        }

        let block = Block::new(&b).expect("block");
        assert_eq!(block.n_cells, 82_320);

        // Exactly one opening. Everything the burner injects has to leave
        // through `outlet`, which is what makes the flux balance a two-term
        // sum and the potential-flow solve well posed.
        let named: Vec<(&str, &str)> = block
            .patches
            .iter()
            .map(|p| (p.name.as_str(), p.type_name.as_str()))
            .collect();
        assert_eq!(
            named,
            vec![
                ("wallXMin", "wall"),
                ("outlet", "patch"),
                ("wallYMin", "wall"),
                ("wallYMax", "wall"),
                ("inlet", "patch"),
                ("floor", "wall"),
                ("ceiling", "wall"),
            ]
        );

        // The only two non-wall patches are the two the flux balance names.
        let open: Vec<&str> = block
            .patches
            .iter()
            .filter(|p| PatchKind::from_type(&p.type_name) != PatchKind::Wall)
            .map(|p| p.name.as_str())
            .collect();
        assert_eq!(open, vec!["outlet", "inlet"]);

        assert_patches_are_contiguous(&block);

        // The burner and what is left of the floor together are the zMin slot.
        assert_eq!(block.patches[4].size + block.patches[5].size, nx * ny);
    }

    /// The requested 1.2 m does not land on a node on either axis, so the
    /// opening has to be snapped. It must stay within one cell of the
    /// requested size and within half a cell of the origin, or the burner is
    /// not the burner the benchmark specifies.
    #[test]
    fn the_plume_burner_snaps_to_whole_cells_around_the_origin() {
        let (nx, ny, nz) = CaseKind::Plume.default_resolution();
        let b = case_block_spec(CaseKind::Plume, nx, ny, nz);
        let block = Block::new(&b).expect("block");
        let w = b.window.as_ref().expect("window");

        let (fx, fy) = window_extent(&block.g, w);
        let dx = (b.x.hi - b.x.lo) / nx as Scalar;
        let dy = (b.y.hi - b.y.lo) / ny as Scalar;

        assert_eq!(w.hi[0] - w.lo[0], 8, "x cells");
        assert_eq!(w.hi[1] - w.lo[1], 8, "y cells");
        assert_eq!(block.patches[4].size, 64);

        let wx = fx[1] - fx[0];
        let wy = fy[1] - fy[0];
        assert!((wx - PLUME_INLET_SIDE).abs() < dx, "x side {wx}");
        assert!((wy - PLUME_INLET_SIDE).abs() < dy, "y side {wy}");
        assert!((0.5 * (fx[0] + fx[1])).abs() < 0.5 * dx, "off centre in x");
        assert!((0.5 * (fy[0] + fy[1])).abs() < 0.5 * dy, "off centre in y");
    }

    /// Degenerate requests: an opening narrower than a cell still has to
    /// produce a patch, because a zero-face inlet is a case that cannot run.
    #[test]
    fn a_burner_narrower_than_one_cell_still_gets_a_face() {
        // Five cells, so one of them is centred on the origin outright.
        let nodes = graded_nodes(&axis(-1.0, 1.0, 5, 1.0, false));
        assert_eq!(centred_cell_range(&nodes, 0.0, 0.01), (2, 3));
        assert_eq!(centred_cell_range(&nodes, 0.0, 1.0), (1, 4));
        assert_eq!(centred_cell_range(&nodes, 0.9, 0.01), (4, 5));
        assert_eq!(centred_cell_range(&nodes, 0.0, 4.0), (0, 5));
        // A single node is not an axis at all; it must not index past the end.
        assert_eq!(centred_cell_range(&[0.0], 0.0, 1.0), (0, 0));
    }

    #[test]
    fn the_written_plume_inlet_is_one_contiguous_run_of_floor_faces() {
        let dir = temp_dir("plume");
        let (nx, ny, nz) = (20usize, 12usize, 6usize);
        write_case(&dir, CaseKind::Plume, nx, ny, nz).expect("write");

        let b = case_block_spec(CaseKind::Plume, nx, ny, nz);
        let block = Block::new(&b).expect("block");
        let w = b.window.as_ref().expect("window");

        let poly = dir.join("constant").join("polyMesh");
        let owner = read_list(&poly.join("owner"));
        let src = fs::read_to_string(poly.join("boundary")).expect("read boundary");

        assert_eq!(
            boundary_patch_names(&src),
            vec!["wallXMin", "outlet", "wallYMin", "wallYMax", "inlet", "floor", "ceiling"]
        );
        assert_eq!(dict_values(&src, "startFace"), starts(&block));
        assert_eq!(dict_values(&src, "nFaces"), sizes(&block));
        assert_eq!(
            dict_values(&src, "startFace").len(),
            7,
            "the split has to show up as seven patches"
        );

        // The cells the burner should sit under, straight from the window.
        let mut want = BTreeSet::new();
        for j in w.lo[1]..w.hi[1] {
            for i in w.lo[0]..w.hi[0] {
                want.insert(block.g.cell(i, j, 0));
            }
        }

        let inlet = &block.patches[4];
        let got: BTreeSet<usize> =
            owner[inlet.start..inlet.start + inlet.size].iter().copied().collect();
        assert_eq!(got, want, "the inlet run does not cover the burner cells");
        assert_eq!(got.len(), inlet.size, "the inlet run repeats a cell");

        // And the floor gets every other zMin cell, exactly once.
        let floor = &block.patches[5];
        let rest: BTreeSet<usize> =
            owner[floor.start..floor.start + floor.size].iter().copied().collect();
        assert_eq!(rest.len(), floor.size);
        assert!(rest.is_disjoint(&want));
        assert_eq!(rest.len() + want.len(), nx * ny);
        for c in rest.iter().chain(want.iter()) {
            assert_eq!(block.g.decompose_cell(*c).2, 0, "cell {c} is not on zMin");
        }

        let _ = fs::remove_dir_all(&dir);
    }

    /// The reader is what a non-contiguous patch would fool, so the split mesh
    /// is checked by loading it back through ofgpu's own polyMesh reader
    /// rather than by asking the writer's bookkeeping a second time.
    #[test]
    fn the_split_mesh_reads_back_with_the_burner_on_the_right_cells() {
        use crate::io::polymesh::{build_host_mesh, read_poly_mesh};
        use crate::Label;

        let dir = temp_dir("plume_roundtrip");
        let (nx, ny, nz) = (20usize, 12usize, 6usize);
        write_case(&dir, CaseKind::Plume, nx, ny, nz).expect("write");

        let m = build_host_mesh(&read_poly_mesh(&dir).expect("read")).expect("host mesh");
        let report = m.check();

        assert!(report.ldu_ordered, "read back out of lduAddressing order");
        assert!(
            report.max_closure_error < 1e-10,
            "closure error {}",
            report.max_closure_error
        );
        assert_eq!(m.n_cells, nx * ny * nz);

        let names: Vec<&str> = m.patches.iter().map(|p| p.name.as_str()).collect();
        assert_eq!(
            names,
            vec!["wallXMin", "outlet", "wallYMin", "wallYMax", "inlet", "floor", "ceiling"]
        );

        let b = case_block_spec(CaseKind::Plume, nx, ny, nz);
        let block = Block::new(&b).expect("block");
        let w = b.window.as_ref().expect("window");

        let inlet = m.patches.iter().find(|p| p.name == "inlet").expect("inlet");
        let floor = m.patches.iter().find(|p| p.name == "floor").expect("floor");
        assert_eq!(inlet.kind, PatchKind::Generic);
        assert_eq!(floor.kind, PatchKind::Wall);
        assert_eq!(inlet.size + floor.size, nx * ny);

        let mut want: BTreeSet<Label> = BTreeSet::new();
        for j in w.lo[1]..w.hi[1] {
            for i in w.lo[0]..w.hi[0] {
                want.insert(block.g.cell(i, j, 0) as Label);
            }
        }

        let got: BTreeSet<Label> = m.b_face_cells[inlet.start..inlet.start + inlet.size]
            .iter()
            .copied()
            .collect();
        assert_eq!(got, want, "the burner sits on the wrong cells");

        // And they really are floor faces: on z = 0, looking down.
        for f in inlet.start..inlet.start + inlet.size {
            assert!(m.b_sf[f].z < 0.0, "burner face {f} does not face down");
            assert!(m.b_cf[f].z.abs() < 1e-12, "burner face {f} is off the floor");
        }

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_plume_fields_carry_the_burner_and_the_single_opening() {
        let dir = temp_dir("plume_fields");
        write_case(&dir, CaseKind::Plume, 20, 12, 6).expect("write");

        let zero = dir.join("0");
        let u = fs::read_to_string(zero.join("U")).expect("read U");
        let t = fs::read_to_string(zero.join("T")).expect("read T");
        let props = fs::read_to_string(dir.join("constant").join("physicalProperties"))
            .expect("read physicalProperties");

        assert!(u.contains("uniform (0 0 2)"), "the inlet jet is missing:\n{u}");
        // The interior starts from rest now that the solver computes it; a
        // prescribed column would be an initial condition arguing with the
        // first pressure correction.
        assert!(
            u.contains("internalField   uniform (0 0 0)"),
            "the plume interior should start at rest:\n{u}"
        );
        assert_eq!(u.matches("inletOutlet").count(), 1, "one opening, at +x");
        assert_eq!(
            u.matches("noSlip").count(),
            5,
            "wallXMin, wallYMin, wallYMax, floor, ceiling"
        );

        assert!(t.contains("[0 0 0 1 0 0 0]"), "T has the wrong dimensions");
        assert!(t.contains("internalField   uniform 293.15"), "{t}");
        assert!(t.contains("uniform 1173.15"), "the burner is not hot:\n{t}");
        assert_eq!(t.matches("inletOutlet").count(), 1);
        // Adiabatic on all five walls.
        assert_eq!(t.matches("zeroGradient").count(), 5);

        // The wall functions have to reach the three new walls too, or the
        // sides would be frictionless surfaces wearing a wall's name.
        let eps = fs::read_to_string(zero.join("epsilon")).expect("read epsilon");
        assert_eq!(eps.matches("epsilonWallFunction").count(), 5);
        let nut = fs::read_to_string(zero.join("nut")).expect("read nut");
        assert_eq!(nut.matches("nutkWallFunction").count(), 5);

        assert!(props.contains("nu              [0 2 -1 0 0 0 0] 1.5e-05;"), "{props}");
        assert!(props.contains("Pr              0.71;"), "{props}");
        assert!(props.contains("Prt             0.85;"), "{props}");
        assert!(props.contains("TRef            293.15;"), "{props}");
        // Boussinesq would be wrong by a factor of three over 293 K to 1173 K,
        // so no expansion coefficient is written and none may appear. Matched
        // as an ENTRY, not as a substring: the file's own comment explains why
        // there is no such coefficient and says the word.
        assert!(
            !props.lines().any(|l| l.trim_start().starts_with("beta")),
            "{props}"
        );

        // ---- constant/g ---------------------------------------------------
        //
        // Parsed rather than string-matched: what matters is that
        // `BuoyancyCoeffs` reads -9.81 in z out of it, not how it is spelled.
        let gp = dir.join("constant").join("g");
        assert!(gp.exists(), "the plume has no constant/g");
        let bc = crate::momentum::BuoyancyCoeffs::from_case(&dir).expect("read g and TRef");
        assert_eq!(bc.g, Vec3::new(0.0, 0.0, -9.81));
        assert_eq!(bc.t_ref, 293.15);
        // The sign that decides whether the plume rises or sinks.
        assert!(bc.at(1173.15).z > 0.0, "hot air must be pushed up");
        assert_eq!(bc.at(293.15), Vec3::ZERO);

        // ---- 0/p ----------------------------------------------------------
        let pf = fs::read_to_string(zero.join("p")).expect("read p");
        assert!(pf.contains("[0 2 -2 0 0 0 0]"), "p is not kinematic:\n{pf}");
        assert!(pf.contains("internalField   uniform 0"), "{pf}");
        // One Dirichlet face is what keeps the pressure matrix non-singular.
        assert_eq!(pf.matches("fixedValue").count(), 1, "{pf}");
        assert_eq!(pf.matches("zeroGradient").count(), 6, "six of the seven patches");

        // Every field names all seven patches, or the solver would fall back
        // to a default on whichever one it could not find.
        for name in ["U", "p", "T", "k", "epsilon", "nut"] {
            let src = fs::read_to_string(zero.join(name)).expect("read field");
            for patch in ["wallXMin", "outlet", "wallYMin", "wallYMax", "inlet", "floor", "ceiling"]
            {
                assert!(src.contains(patch), "{name} has no {patch} entry");
            }
        }

        let _ = fs::remove_dir_all(&dir);
    }

    /// The other cases have to come out byte for byte as they did before the
    /// plume was added: no seventh patch, no `0/T`, no Prandtl numbers.
    #[test]
    fn the_existing_cases_are_untouched() {
        let dir = temp_dir("cavity_unchanged");
        write_case(&dir, CaseKind::Cavity, 8, 8, 1).expect("write");

        let src = fs::read_to_string(dir.join("constant").join("polyMesh").join("boundary"))
            .expect("read boundary");
        assert_eq!(
            boundary_patch_names(&src),
            vec!["leftWall", "rightWall", "fixedWall", "movingWall", "back", "front"]
        );

        assert!(!dir.join("0").join("T").exists(), "the cavity grew a T field");
        assert!(!dir.join("0").join("p").exists(), "the cavity grew a p field");
        assert!(!dir.join("constant").join("g").exists(), "the cavity grew gravity");

        let props = fs::read_to_string(dir.join("constant").join("physicalProperties"))
            .expect("read physicalProperties");
        assert!(props.contains("nu              [0 2 -1 0 0 0 0] 1e-05;"), "{props}");
        assert!(!props.contains("Pr "), "{props}");

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_degenerate_axis_is_refused_rather_than_written() {
        let mut b = spec(3, 3, 3);
        b.z.n = 0;
        assert!(Block::new(&b).is_err());
        assert!(Grid::new(&b).is_err());
        // Refused before anything touches the filesystem.
        assert!(write_case(Path::new("."), CaseKind::Cavity, 4, 4, 0).is_err());
    }
}
