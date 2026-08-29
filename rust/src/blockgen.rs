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
use crate::io::case::{Roughness, WallTreatment};
use crate::io::fields::{
    write_scalar_field, write_vector_field, PatchFieldSpec, RawScalarField, RawVectorField,
};
use crate::io::polymesh::{build_host_mesh, PolyMeshRaw};
use crate::mesh::{HostMesh, PatchInfo, PatchKind};
use crate::surface::classify::{classify, BlockAxes, SolidMask};
use crate::surface::cutcell::{
    classify_cutcells, merge_small_cells, CellState, CutCellField, MergeResult,
    DEFAULT_SUPERSAMPLE, DEFAULT_THETA_MIN,
};
use crate::surface::{Surface, TriIndex};
use crate::{Label, Scalar, Vec3};
use std::collections::HashMap;

/// One `0/` directory's worth of fields, entirely in memory - the same
/// [`RawScalarField`]/[`RawVectorField`] data a case's field writers would
/// otherwise serialise straight to disk. Order matches the field-name order
/// each case writes (`U` before `p` before `T` before the turbulence
/// quartet, or `alpha.water` before `U` before `p_rgh` for the dam break),
/// but nothing downstream depends on that - each field is its own file.
#[derive(Debug, Default, Clone)]
pub struct InMemoryFields {
    pub scalars: Vec<RawScalarField>,
    pub vectors: Vec<RawVectorField>,
}

/// Serialise every field in `fields` to `case_dir/0/<name>` - the shared tail
/// of every case's field writer, so the disk path and [`build_case`]'s
/// in-memory path are guaranteed to write exactly what they built.
fn write_fields(case_dir: &Path, fields: &InMemoryFields) -> Result<()> {
    for s in &fields.scalars {
        write_scalar_field(&case_dir.join("0").join(&s.name), s, "0")?;
    }
    for v in &fields.vectors {
        write_vector_field(&case_dir.join("0").join(&v.name), v, "0")?;
    }
    Ok(())
}

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
    /// Split slots in two; see [`PatchWindow`]. At most ONE window per slot -
    /// a slot with two would have to emit three patches whose faces interleave,
    /// and OpenFOAM records a patch as nothing but a `startFace`/`nFaces` pair,
    /// so it cannot be expressed. `boundary_patches` refuses it by name.
    ///
    /// SPEC-LIT §42.8 Gate 2 is what generalised this from the single slot it
    /// shipped with: a compartment fire needs a burner window in the floor AND
    /// a doorway window in a wall, and one window is not enough for the case
    /// the whole two-step scheme exists to answer.
    pub windows: Vec<PatchWindow>,
    /// SPEC-LIT §31.1/§34.2: the axes (0=x, 1=y, 2=z) whose two opposite
    /// slots are a cyclic pair - `constant/polyMesh/boundary` gets
    /// `neighbourPatch` on each, and the in-memory [`build_mesh`] path
    /// resolves the pairing directly, with no boundary file to read it back
    /// from. A plane channel needs two axes here, a fully periodic box three
    /// - §34.2 generalised this from the single `Option<usize>` slot §31.1
    /// shipped with, because a `Vec` of axes is already the whole
    /// generalisation a pair needs: each axis has exactly two opposite
    /// slots, so "one pair per axis" and "one pair per patch" are the same
    /// constraint once pairing is expressed this way. Set through
    /// [`BlockSpec::set_cyclic_axis`], which also fixes up `patch_type` -
    /// this field on its own is not enough to make a slot cyclic.
    pub cyclic: Vec<usize>,
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
            windows: Vec::new(),
            cyclic: Vec::new(),
        }
    }
}

impl BlockSpec {
    /// Declare the two opposite slots of `axis` (0=x, 1=y, 2=z) a cyclic
    /// pair (SPEC-LIT §31.1, generalised to more than one axis by §34.2):
    /// both slots' `patch_type` become `cyclic` and `axis` is appended to
    /// `cyclic`, which is what [`build_patches`] reads to wire
    /// `neighbourPatch` between them. There is no rotational form - a §13.4
    /// error naming `translate` is [`crate::io::case_json`]'s job, one layer
    /// up, before a `BlockSpec` is even built; by the time one gets here the
    /// only transform this method (or this whole module) knows how to do is
    /// the one implied by the block's own extent along `axis`.
    ///
    /// SPEC-LIT §34.2's "an axis may appear in at most one pair" is enforced
    /// HERE rather than only by the JSONC reader, so a caller that builds a
    /// `BlockSpec` directly (as `write_case_cyclic` and this module's own
    /// tests do) gets the same guarantee: calling this twice for the same
    /// axis is an error, not a silent no-op or a double-cyclic slot.
    pub fn set_cyclic_axis(&mut self, axis: usize) -> Result<()> {
        if axis > 2 {
            return Err(Error::Config(format!(
                "blockgen: cyclic axis {axis} is not x, y or z (0, 1 or 2)"
            )));
        }
        if self.cyclic.contains(&axis) {
            return Err(Error::Config(format!(
                "blockgen: axis {axis} ('{}'/'{}') is already a cyclic pair - an axis \
                 may appear in at most one cyclic pair",
                self.patch_name[2 * axis],
                self.patch_name[2 * axis + 1],
            )));
        }
        self.patch_type[2 * axis] = "cyclic".to_string();
        self.patch_type[2 * axis + 1] = "cyclic".to_string();
        self.cyclic.push(axis);
        Ok(())
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
    /// SPEC-LIT §31.1: set from [`BlockSpec::cyclic`] when this patch is one
    /// of the axis's two opposite slots - the OTHER patch's name, written as
    /// `neighbourPatch` in the boundary file and resolved to an index by
    /// [`poly_mesh_raw`] for the in-memory path.
    nbr_name: Option<String>,
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
    // One entry per slot, so the per-slot logic below is unchanged and a
    // second window on the same slot is refused rather than silently dropped.
    let mut win: [Option<(&PatchWindow, [usize; 4])>; 6] = [None; 6];
    for w in &b.windows {
        if w.slot > 5 {
            return Err(Error::Mesh(format!(
                "blockgen: patch window '{}' names slot {}, which is not one of                  the six (-x +x -y +y -z +z)",
                w.name, w.slot
            )));
        }
        if let Some((prev, _)) = win[w.slot] {
            return Err(Error::Mesh(format!(
                "blockgen: patch windows '{}' and '{}' both split slot {} - a                  slot can carry at most one window, because OpenFOAM records a                  patch as one contiguous startFace/nFaces run and three patches                  cut from one slot cannot be laid out without interleaving",
                prev.name, w.name, w.slot
            )));
        }
        win[w.slot] = Some((w, window_rect(g, w)?));
    }

    // SPEC-LIT §31.1/§34.2: `set_cyclic_axis` is the only supported way to
    // get here, but the field itself is public, so a caller that poked
    // `cyclic` directly without also setting both `patch_type` entries gets
    // an error naming the mismatch rather than a boundary file that claims
    // `cyclic` pairing for a patch whose own `type` says something else.
    // Checked per axis, since `cyclic` may now name more than one.
    for &axis in &b.cyclic {
        if axis > 2 {
            return Err(Error::Mesh(format!(
                "blockgen: BlockSpec.cyclic axis {axis} is not x, y or z (0, 1 or 2)"
            )));
        }
        let (lo, hi) = (2 * axis, 2 * axis + 1);
        if b.patch_type[lo] != "cyclic" || b.patch_type[hi] != "cyclic" {
            return Err(Error::Mesh(format!(
                "blockgen: BlockSpec.cyclic names axis {axis} ('{}'/'{}') but their \
                 patch_type is '{}'/'{}', not 'cyclic' on both - use \
                 BlockSpec::set_cyclic_axis, which sets both together",
                b.patch_name[lo], b.patch_name[hi], b.patch_type[lo], b.patch_type[hi]
            )));
        }
        for w in &b.windows {
            if w.slot == lo || w.slot == hi {
                return Err(Error::Mesh(format!(
                    "blockgen: patch window '{}' splits slot {}, which is half of the \
                     axis-{axis} cyclic pair - a cyclic patch cannot be windowed",
                    w.name, w.slot
                )));
            }
        }
    }
    let nbr_name = |p: usize| -> Option<String> {
        let axis = b.cyclic.iter().copied().find(|&a| p == 2 * a || p == 2 * a + 1)?;
        if p == 2 * axis {
            Some(b.patch_name[2 * axis + 1].clone())
        } else {
            Some(b.patch_name[2 * axis].clone())
        }
    };

    for p in 0..6 {
        match win[p] {
            Some((w, rect)) => {
                let n_win = (rect[1] - rect[0]) * (rect[3] - rect[2]);

                out.push(OutPatch {
                    name: w.name.clone(),
                    type_name: w.type_name.clone(),
                    slot: p,
                    part: SlotPart::Window,
                    win: rect,
                    start: start[p],
                    size: n_win,
                    nbr_name: None,
                });
                out.push(OutPatch {
                    name: b.patch_name[p].clone(),
                    type_name: b.patch_type[p].clone(),
                    slot: p,
                    part: SlotPart::Rest,
                    win: rect,
                    start: start[p] + n_win,
                    size: size[p] - n_win,
                    nbr_name: None,
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
                nbr_name: nbr_name(p),
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

    /// 17 significant digits, far more than OpenFOAM's default
    /// `writePrecision` of 6 (a graded mesh loses real geometric accuracy at
    /// 6) and enough to round-trip an `f64` EXACTLY (Steele & White 1990):
    /// 15 was not - `docs/05-io-redesign.md`'s B3 phase-1 gate (comparing an
    /// OpenFOAM-format `write_block_mesh` case against the in-memory
    /// `build_mesh` twin cell for cell) found points this mesh writes that a
    /// 15-digit round trip does not reproduce bit-for-bit (most nodes of a
    /// non-power-of-two-friendly axis like `plume`'s 98/42-cell x/y - e.g.
    /// `-7.87183673469388` reading back to a different `f64` than the
    /// `-7.871836734693877` that was written). 17 digits is the textbook
    /// fix, not a guess: verified against every node of that mesh's x and y
    /// axes.
    fn real(&mut self, v: Scalar) -> Result<()> {
        self.reserve(48)?;
        self.buf.push_str(&fmt_g_prec(v, 17));
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
            os.s(";\n")?;
            // SPEC-LIT §31.1: the only field the reader needs beyond `type`
            // to resolve a cyclic pair (`io/polymesh.rs`'s `neighbourPatch`
            // resolution).
            if let Some(nbr) = &patch.nbr_name {
                os.s("        neighbourPatch  ")?;
                os.s(nbr)?;
                os.s(";\n")?;
            }
            os.s("        nFaces          ")?;
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

/// The uncarved block's points, faces, owner/neighbour and patches, entirely
/// in memory - the same geometry [`write_poly_mesh`] serialises to
/// `constant/polyMesh`, built once here so [`build_mesh`] never has to touch
/// disk to get it. Kept in lock-step with `write_poly_mesh`'s two loops
/// deliberately: any change to face order or winding there must be made here
/// too, which is exactly what the byte-identical-output test guards.
fn poly_mesh_raw(block: &Block) -> Result<PolyMeshRaw> {
    let g = &block.g;

    let mut points = Vec::with_capacity(block.n_points);
    for k in 0..=g.nz {
        for j in 0..=g.ny {
            for i in 0..=g.nx {
                points.push(g.point_coord(g.point(i, j, k)));
            }
        }
    }

    let n_faces_total = block.n_faces();
    let mut faces: Vec<Vec<Label>> = Vec::with_capacity(n_faces_total);
    let mut owner: Vec<Label> = Vec::with_capacity(n_faces_total);
    let neighbour: Vec<Label> = block.faces.iter().map(|f| f.nei as Label).collect();

    for (f, iface) in block.faces.iter().enumerate() {
        let q = internal_quad(g, *iface);
        if !winding_ok(g, &q) {
            return Err(bad_winding("internal", f, &q));
        }
        faces.push(q.p.iter().map(|&p| p as Label).collect());
        owner.push(q.own as Label);
    }

    let mut patches: Vec<PatchInfo> = Vec::with_capacity(block.patches.len());
    for patch in &block.patches {
        let (na, _) = slot_dims(g, patch.slot);
        for idx in 0..patch.size {
            let q = boundary_quad(g, patch.slot, patch.slot_index(na, idx));
            if !winding_ok(g, &q) {
                return Err(bad_winding(&patch.name, idx, &q));
            }
            faces.push(q.p.iter().map(|&p| p as Label).collect());
            owner.push(q.own as Label);
        }
        // SPEC-LIT §31.1: `block.patches` and `patches` are built in the same
        // order, one push per entry, so the neighbour's index here is just
        // where its name sits in that same list - no different from what
        // `io/polymesh.rs`'s `neighbourPatch` resolution does for a mesh read
        // off disk, just without a boundary file to read it from.
        let nbr_patch = patch.nbr_name.as_ref().and_then(|nbr| {
            block.patches.iter().position(|q| &q.name == nbr)
        });
        patches.push(PatchInfo {
            name: patch.name.clone(),
            type_name: patch.type_name.clone(),
            kind: PatchKind::from_type(&patch.type_name),
            start: patch.start - block.n_internal,
            size: patch.size,
            nbr_patch,
        });
    }

    Ok(PolyMeshRaw { points, faces, owner, neighbour, patches })
}

/// Build a runnable [`HostMesh`] straight from a [`BlockSpec`], with no file
/// on disk at any point.
///
/// This is the in-memory twin of `write_block_mesh` + `read_poly_mesh` +
/// `build_host_mesh`: same geometry, same validation (`build_host_mesh` still
/// checks upper-triangular ordering and closure), just without round-tripping
/// through ASCII to get there. Callers that need the case ON disk (to hand to
/// another tool, or to re-run without regenerating) still want
/// [`write_block_mesh`]; this is for a solver or benchmark that only ever
/// wanted the mesh in memory.
pub fn build_mesh(b: &BlockSpec) -> Result<HostMesh> {
    let block = Block::new(b)?;
    let raw = poly_mesh_raw(&block)?;
    build_host_mesh(&raw)
}

// ==========================================================================
//  Castellated carving - SPEC-LIT §23.4
// ==========================================================================
//
// Written from SPEC-LIT §23.3-23.4 and Aftosmis, Berger & Melton, AIAA J.
// 36(6) (1998) 952 (the "castellate" stage only). No GPL-licensed source was
// consulted. Given the block and the §23.3 solid mask: fluid cells are
// renumbered i-fastest, internal faces stay upper-triangular, the domain
// patches keep their surviving faces, and every fluid face against a solid
// cell becomes a face of a NEW wall patch named for the surface patch of the
// nearest triangle. Castellation is first-order at the boundary (stair
// steps) - the documented trade until a cut-cell stage exists (§23.5).

/// What carving did, for the caller to print. Everything §23.4's CLI summary
/// needs: cell classification counts and the new boundary faces per patch.
#[derive(Debug, Clone)]
pub struct CarveSummary {
    /// Cells in the uncarved block.
    pub n_cells_block: usize,
    pub n_solid: usize,
    pub n_fluid: usize,
    /// Cells settled by the 3-axis majority vote (§23.3).
    pub voted: usize,
    /// Cells arbitrated by the winding number (§23.3).
    pub arbitrated: usize,
    pub n_internal_faces: usize,
    /// Surviving domain-boundary faces, all original patches together.
    pub n_domain_faces: usize,
    /// New wall patches in written order: `(name, nFaces)`.
    pub wall_faces: Vec<(String, usize)>,
}

/// The carved topology: everything the polyMesh and field writers need
/// beyond the [`Block`] itself.
pub(crate) struct Carved {
    /// Old cell id -> new fluid cell id, `-1` for solid. The renumbering is
    /// order-preserving (i fastest, as before), which is what keeps the
    /// internal faces upper-triangular without a re-sort.
    new_of_old: Vec<i64>,
    /// New fluid cell id -> old cell id.
    fluid_old: Vec<usize>,
    /// Internal faces between two FLUID cells; `own`/`nei` are OLD cell ids
    /// (the writers translate), already in (new owner, new neighbour) order.
    ifaces: Vec<IFace>,
    /// Per block patch (same indexing as `Block::patches`): the surviving
    /// SLOT-LOCAL face indices, in the patch's own face order.
    domain: Vec<Vec<usize>>,
    /// New wall patches, one per surface patch that received at least one
    /// face, in surface-patch order: `(name, [(old fluid cell, dir6)])`.
    /// `dir6` is 0..6 = `-x +x -y +y -z +z`.
    walls: Vec<(String, Vec<(usize, u8)>)>,
    // Classification statistics, carried for the summary.
    n_solid: usize,
    voted: usize,
    arbitrated: usize,
}

/// The quad of cell `c`'s face in direction `dir6` (0..6 = `-x +x -y +y -z
/// +z`), wound so the normal points OUT of the cell - the winding a carved
/// wall face needs, since its owner is the fluid cell it was cut from.
/// `winding_ok` re-verifies every one on write, same as every other face.
fn wall_quad(g: &Grid, c: usize, dir6: u8) -> Quad {
    // The three + directions are exactly the owner's internal-face windings.
    if dir6 % 2 == 1 {
        let mut q = internal_quad(g, IFace { own: c, nei: c, dir: dir6 / 2 });
        q.nei = None;
        return q;
    }

    let (i, j, k) = g.decompose_cell(c);
    let p = match dir6 {
        0 => [
            // -x, the `boundary_quad` xMin winding at plane i
            g.point(i, j, k),
            g.point(i, j, k + 1),
            g.point(i, j + 1, k + 1),
            g.point(i, j + 1, k),
        ],
        2 => [
            // -y
            g.point(i, j, k),
            g.point(i + 1, j, k),
            g.point(i + 1, j, k + 1),
            g.point(i, j, k + 1),
        ],
        _ => [
            // -z
            g.point(i, j, k),
            g.point(i, j + 1, k),
            g.point(i + 1, j + 1, k),
            g.point(i + 1, j, k),
        ],
    };

    Quad { p, own: c, nei: None }
}

/// Centre of cell `c`'s face in direction `dir6` - exact for a rectilinear
/// hex, and the query point for the nearest-triangle patch naming.
fn wall_face_centre(g: &Grid, c: usize, dir6: u8) -> Vec3 {
    let (i, j, k) = g.decompose_cell(c);
    let mx = 0.5 * (g.xn[i] + g.xn[i + 1]);
    let my = 0.5 * (g.yn[j] + g.yn[j + 1]);
    let mz = 0.5 * (g.zn[k] + g.zn[k + 1]);
    match dir6 {
        0 => Vec3::new(g.xn[i], my, mz),
        1 => Vec3::new(g.xn[i + 1], my, mz),
        2 => Vec3::new(mx, g.yn[j], mz),
        3 => Vec3::new(mx, g.yn[j + 1], mz),
        4 => Vec3::new(mx, my, g.zn[k]),
        _ => Vec3::new(mx, my, g.zn[k + 1]),
    }
}

/// Validate the surface against the block, classify, and build the carved
/// topology. The loud failures of §23.4's intake live here: an unclosed
/// surface (via the §13.4 contract), a surface entirely outside the domain,
/// and a surface that swallows the whole domain.
fn carve_block(block: &Block, surf: &Surface) -> Result<Carved> {
    let g = &block.g;

    // Closure check first (§23.2): strict mode refuses, -permissive warns
    // and leans on the parity voting below.
    surf.require_closed()?;

    // A surface whose bounding box misses the block cannot carve anything;
    // proceeding would silently write the uncarved mesh under a name that
    // claims otherwise.
    let (dlo, dhi) = (
        Vec3::new(g.xn[0], g.yn[0], g.zn[0]),
        Vec3::new(g.xn[g.nx], g.yn[g.ny], g.zn[g.nz]),
    );
    let (slo, shi) = surf.bbox;
    if shi.x < dlo.x || slo.x > dhi.x || shi.y < dlo.y || slo.y > dhi.y
        || shi.z < dlo.z || slo.z > dhi.z
    {
        return Err(Error::Mesh(format!(
            "carve: the surface (bounds [{} {} {}] to [{} {} {}]) lies entirely \
             outside the block ([{} {} {}] to [{} {} {}]) - nothing to carve",
            fmt_g(slo.x), fmt_g(slo.y), fmt_g(slo.z),
            fmt_g(shi.x), fmt_g(shi.y), fmt_g(shi.z),
            fmt_g(dlo.x), fmt_g(dlo.y), fmt_g(dlo.z),
            fmt_g(dhi.x), fmt_g(dhi.y), fmt_g(dhi.z),
        )));
    }

    let mask = classify(
        &BlockAxes { xn: &g.xn, yn: &g.yn, zn: &g.zn },
        surf,
    )?;
    if mask.n_fluid() == 0 {
        return Err(Error::Mesh(format!(
            "carve: every one of the {} cells is inside the surface - it \
             swallows the whole domain",
            mask.n_cells()
        )));
    }

    Carved::build(block, &mask, surf)
}

impl Carved {
    fn build(block: &Block, mask: &SolidMask, surf: &Surface) -> Result<Carved> {
        let g = &block.g;
        if mask.nx != g.nx || mask.ny != g.ny || mask.nz != g.nz {
            return Err(Error::Mesh(format!(
                "carve: mask is {} x {} x {} but the block is {} x {} x {}",
                mask.nx, mask.ny, mask.nz, g.nx, g.ny, g.nz
            )));
        }

        // ---- renumber the fluid cells, i fastest --------------------------
        let mut new_of_old = vec![-1i64; block.n_cells];
        let mut fluid_old = Vec::with_capacity(mask.n_fluid());
        for c in 0..block.n_cells {
            if !mask.solid[c] {
                new_of_old[c] = fluid_old.len() as i64;
                fluid_old.push(c);
            }
        }

        // ---- internal faces: fluid-fluid only ------------------------------
        // Old order was upper-triangular with neighbours c+1 < c+nx <
        // c+nx*ny; the renumbering is monotone in the old ids, so the same
        // emission order is already (new owner, new neighbour) sorted. The
        // check below makes that true rather than assumed, exactly as
        // `Block::new` does for the uncarved mesh.
        let mut ifaces: Vec<IFace> = Vec::new();
        for &c in &fluid_old {
            let (i, j, k) = g.decompose_cell(c);
            if i + 1 < g.nx && !mask.solid[c + 1] {
                ifaces.push(IFace { own: c, nei: c + 1, dir: 0 });
            }
            if j + 1 < g.ny && !mask.solid[c + g.nx] {
                ifaces.push(IFace { own: c, nei: c + g.nx, dir: 1 });
            }
            if k + 1 < g.nz && !mask.solid[c + g.nx * g.ny] {
                ifaces.push(IFace { own: c, nei: c + g.nx * g.ny, dir: 2 });
            }
        }
        for f in 0..ifaces.len() {
            let (o, n) = (new_of_old[ifaces[f].own], new_of_old[ifaces[f].nei]);
            if o < 0 || n < 0 || o >= n {
                return Err(Error::Mesh(format!(
                    "carve: internal face {f} is not upper-triangular after \
                     renumbering (owner {o}, neighbour {n})"
                )));
            }
            if f > 0 {
                let (po, pn) = (
                    new_of_old[ifaces[f - 1].own],
                    new_of_old[ifaces[f - 1].nei],
                );
                if (po, pn) >= (o, n) {
                    return Err(Error::Mesh(format!(
                        "carve: internal faces out of order at face {f}"
                    )));
                }
            }
        }

        // ---- domain boundary faces: keep the fluid-owned ones -------------
        let mut domain: Vec<Vec<usize>> = Vec::with_capacity(block.patches.len());
        for patch in &block.patches {
            let (na, _) = slot_dims(g, patch.slot);
            let mut keep = Vec::new();
            for idx in 0..patch.size {
                let sl = patch.slot_index(na, idx);
                if new_of_old[boundary_quad(g, patch.slot, sl).own] >= 0 {
                    keep.push(sl);
                }
            }
            domain.push(keep);
        }

        // ---- new wall faces: fluid against solid ---------------------------
        // Named for the surface patch of the nearest triangle to the face
        // centre (§23.4), via the uniform-grid bucket. Cell size ~ the mesh
        // spacing, same hint rule the classifier uses.
        let hint = ((g.xn[g.nx] - g.xn[0]).abs() / g.nx as Scalar
            + (g.yn[g.ny] - g.yn[0]).abs() / g.ny as Scalar
            + (g.zn[g.nz] - g.zn[0]).abs() / g.nz as Scalar)
            / 3.0;
        let tidx = TriIndex::new(surf, hint.max(Scalar::MIN_POSITIVE))?;

        let mut per_patch: Vec<Vec<(usize, u8)>> =
            vec![Vec::new(); surf.patch_names.len()];
        for &c in &fluid_old {
            let (i, j, k) = g.decompose_cell(c);
            let nbr: [Option<usize>; 6] = [
                (i > 0).then(|| c - 1),
                (i + 1 < g.nx).then(|| c + 1),
                (j > 0).then(|| c - g.nx),
                (j + 1 < g.ny).then(|| c + g.nx),
                (k > 0).then(|| c - g.nx * g.ny),
                (k + 1 < g.nz).then(|| c + g.nx * g.ny),
            ];
            for (dir6, n) in nbr.into_iter().enumerate() {
                let Some(n) = n else { continue };
                if !mask.solid[n] {
                    continue;
                }
                let (t, _) = tidx.nearest_triangle(wall_face_centre(g, c, dir6 as u8));
                per_patch[surf.tri_patch[t] as usize].push((c, dir6 as u8));
            }
        }

        let mut walls: Vec<(String, Vec<(usize, u8)>)> = Vec::new();
        for (p, faces) in per_patch.into_iter().enumerate() {
            if faces.is_empty() {
                continue;
            }
            let name = surf.patch_names[p].clone();
            if block.patches.iter().any(|q| q.name == name) {
                // A silent rename here would detach the user's boundary
                // conditions from their geometry; a duplicate patch name in
                // the boundary file is a mesh OpenFOAM resolves wrongly.
                return Err(Error::Mesh(format!(
                    "carve: surface patch '{name}' collides with a domain patch \
                     of the same name - rename the STL solid (or use \
                     -stl name=path)"
                )));
            }
            walls.push((name, faces));
        }

        Ok(Carved {
            new_of_old,
            fluid_old,
            ifaces,
            domain,
            walls,
            n_solid: mask.n_solid,
            voted: mask.voted,
            arbitrated: mask.arbitrated,
        })
    }

    fn n_domain_faces(&self) -> usize {
        self.domain.iter().map(Vec::len).sum()
    }

    fn n_wall_faces(&self) -> usize {
        self.walls.iter().map(|(_, f)| f.len()).sum()
    }

    fn summary(&self, block: &Block) -> CarveSummary {
        CarveSummary {
            n_cells_block: block.n_cells,
            n_solid: self.n_solid,
            n_fluid: self.fluid_old.len(),
            voted: self.voted,
            arbitrated: self.arbitrated,
            n_internal_faces: self.ifaces.len(),
            n_domain_faces: self.n_domain_faces(),
            wall_faces: self
                .walls
                .iter()
                .map(|(n, f)| (n.clone(), f.len()))
                .collect(),
        }
    }

    /// Every face of the carved mesh in written order - internal (already
    /// upper-triangular), then each domain patch's survivors, then each new
    /// wall patch - with the winding of every quad re-verified on the way
    /// through, exactly as the uncarved writer does.
    fn for_each_quad(
        &self,
        block: &Block,
        mut f: impl FnMut(&Quad, bool) -> Result<()>,
    ) -> Result<()> {
        let g = &block.g;

        for (fi, iface) in self.ifaces.iter().enumerate() {
            let q = internal_quad(g, *iface);
            if !winding_ok(g, &q) {
                return Err(bad_winding("carved internal", fi, &q));
            }
            f(&q, true)?;
        }
        for (bi, patch) in block.patches.iter().enumerate() {
            for (n, &sl) in self.domain[bi].iter().enumerate() {
                let q = boundary_quad(g, patch.slot, sl);
                if !winding_ok(g, &q) {
                    return Err(bad_winding(&patch.name, n, &q));
                }
                f(&q, false)?;
            }
        }
        for (name, faces) in &self.walls {
            for (n, &(cell, dir6)) in faces.iter().enumerate() {
                let q = wall_quad(g, cell, dir6);
                if !winding_ok(g, &q) {
                    return Err(bad_winding(name, n, &q));
                }
                f(&q, false)?;
            }
        }
        Ok(())
    }
}

/// Write the carved polyMesh through the same TextOut machinery as the
/// uncarved one: `points` (compacted to the points the surviving faces
/// use), `faces`, `owner`, `neighbour`, `boundary`.
fn write_carved_poly_mesh(case_dir: &Path, block: &Block, cv: &Carved) -> Result<()> {
    let g = &block.g;
    let dir = case_dir.join("constant").join("polyMesh");
    fs::create_dir_all(&dir).path(&dir)?;

    let n_fluid = cv.fluid_old.len();
    let n_internal = cv.ifaces.len();
    let n_faces = n_internal + cv.n_domain_faces() + cv.n_wall_faces();

    // ---- point compaction --------------------------------------------------
    // Solid-only corners must not survive into the file: a point no face
    // names is dead weight, and checkMesh reports it as an error.
    let mut used = vec![false; block.n_points];
    cv.for_each_quad(block, |q, _| {
        for &p in &q.p {
            used[p] = true;
        }
        Ok(())
    })?;
    let mut pmap = vec![-1i64; block.n_points];
    let mut n_points = 0usize;
    for (p, u) in used.iter().enumerate() {
        if *u {
            pmap[p] = n_points as i64;
            n_points += 1;
        }
    }

    let note = format!(
        "nPoints:{}  nCells:{}  nFaces:{}  nInternalFaces:{}",
        n_points, n_fluid, n_faces, n_internal
    );

    // ---- points ------------------------------------------------------------
    {
        let mut os = TextOut::create(&dir.join("points"))?;
        write_foam_header(&mut os, "vectorField", "points", "")?;
        os.num(n_points)?;
        os.s("\n(\n")?;
        for p in 0..block.n_points {
            if !used[p] {
                continue;
            }
            let v = g.point_coord(p);
            os.c('(')?;
            os.real(v.x)?;
            os.c(' ')?;
            os.real(v.y)?;
            os.c(' ')?;
            os.real(v.z)?;
            os.s(")\n")?;
        }
        os.s(")")?;
        write_foam_footer(&mut os)?;
        os.finish()?;
    }

    // ---- faces ---------------------------------------------------------------
    {
        let mut os = TextOut::create(&dir.join("faces"))?;
        write_foam_header(&mut os, "faceList", "faces", "")?;
        os.num(n_faces)?;
        os.s("\n(\n")?;
        cv.for_each_quad(block, |q, _| {
            let t = Quad {
                p: [
                    pmap[q.p[0]] as usize,
                    pmap[q.p[1]] as usize,
                    pmap[q.p[2]] as usize,
                    pmap[q.p[3]] as usize,
                ],
                own: q.own,
                nei: q.nei,
            };
            write_quad(&mut os, &t)
        })?;
        os.s(")")?;
        write_foam_footer(&mut os)?;
        os.finish()?;
    }

    // ---- owner ----------------------------------------------------------------
    {
        let mut os = TextOut::create(&dir.join("owner"))?;
        write_foam_header(&mut os, "labelList", "owner", &note)?;
        os.num(n_faces)?;
        os.s("\n(\n")?;
        cv.for_each_quad(block, |q, _| {
            os.num(cv.new_of_old[q.own] as usize)?;
            os.c('\n')
        })?;
        os.s(")")?;
        write_foam_footer(&mut os)?;
        os.finish()?;
    }

    // ---- neighbour --------------------------------------------------------------
    {
        let mut os = TextOut::create(&dir.join("neighbour"))?;
        write_foam_header(&mut os, "labelList", "neighbour", &note)?;
        os.num(n_internal)?;
        os.s("\n(\n")?;
        for f in &cv.ifaces {
            os.num(cv.new_of_old[f.nei] as usize)?;
            os.c('\n')?;
        }
        os.s(")")?;
        write_foam_footer(&mut os)?;
        os.finish()?;
    }

    // ---- boundary -------------------------------------------------------------
    // Domain patches first (all of them, even ones carving emptied - the
    // field files still name them), then the new wall patches. Every patch
    // is one contiguous run because the faces were written patch by patch.
    {
        let mut os = TextOut::create(&dir.join("boundary"))?;
        write_foam_header(&mut os, "polyBoundaryMesh", "boundary", "")?;

        os.num(block.patches.len() + cv.walls.len())?;
        os.s("\n(\n")?;

        let mut start = n_internal;
        let mut entry = |os: &mut TextOut, name: &str, tname: &str, size: usize| -> Result<()> {
            os.s("    ")?;
            os.s(name)?;
            os.s("\n    {\n        type            ")?;
            os.s(tname)?;
            os.s(";\n        nFaces          ")?;
            os.num(size)?;
            os.s(";\n        startFace       ")?;
            os.num(start)?;
            os.s(";\n    }\n")?;
            start += size;
            Ok(())
        };

        for (bi, patch) in block.patches.iter().enumerate() {
            entry(&mut os, &patch.name, &patch.type_name, cv.domain[bi].len())?;
        }
        for (name, faces) in &cv.walls {
            entry(&mut os, name, "wall", faces.len())?;
        }

        os.s(")")?;
        write_foam_footer(&mut os)?;
        os.finish()?;
    }

    println!(
        "[blockgen] {}: carved {} of {} cells solid -> {} fluid cells, {} points, \
         {} faces ({} internal, {} new wall)",
        dir.display(),
        cv.n_solid,
        block.n_cells,
        n_fluid,
        n_points,
        n_faces,
        n_internal,
        cv.n_wall_faces()
    );

    Ok(())
}

/// The carved mesh's points, faces, owner/neighbour and patches, entirely in
/// memory - the point-compaction and face-order twin of
/// [`write_carved_poly_mesh`], built from the same [`Carved::for_each_quad`]
/// so the two cannot disagree about face order without disagreeing about
/// winding too, which the geometry check downstream would catch.
fn poly_mesh_raw_carved(block: &Block, cv: &Carved) -> Result<PolyMeshRaw> {
    let g = &block.g;
    let n_internal = cv.ifaces.len();
    let n_faces = n_internal + cv.n_domain_faces() + cv.n_wall_faces();

    let mut used = vec![false; block.n_points];
    cv.for_each_quad(block, |q, _| {
        for &p in &q.p {
            used[p] = true;
        }
        Ok(())
    })?;
    let mut pmap = vec![-1i64; block.n_points];
    let mut points = Vec::with_capacity(block.n_points);
    for (p, &u) in used.iter().enumerate() {
        if u {
            pmap[p] = points.len() as i64;
            points.push(g.point_coord(p));
        }
    }

    let mut faces: Vec<Vec<Label>> = Vec::with_capacity(n_faces);
    let mut owner: Vec<Label> = Vec::with_capacity(n_faces);
    cv.for_each_quad(block, |q, _| {
        faces.push(q.p.iter().map(|&p| pmap[p] as Label).collect());
        owner.push(cv.new_of_old[q.own] as Label);
        Ok(())
    })?;
    let neighbour: Vec<Label> = cv
        .ifaces
        .iter()
        .map(|f| cv.new_of_old[f.nei] as Label)
        .collect();

    let mut patches: Vec<PatchInfo> = Vec::with_capacity(block.patches.len() + cv.walls.len());
    // FLATTENED offset (`startFace - nInternalFaces`, per `PatchInfo::start`'s
    // contract), not the global face index `write_carved_poly_mesh`'s `start`
    // tracks - the domain patches begin at 0 here, not at `n_internal`.
    let mut start = 0usize;
    for (bi, patch) in block.patches.iter().enumerate() {
        let size = cv.domain[bi].len();
        patches.push(PatchInfo {
            name: patch.name.clone(),
            type_name: patch.type_name.clone(),
            kind: PatchKind::from_type(&patch.type_name),
            start,
            size,
            nbr_patch: None,
        });
        start += size;
    }
    for (name, wfaces) in &cv.walls {
        let size = wfaces.len();
        patches.push(PatchInfo {
            name: name.clone(),
            type_name: "wall".to_string(),
            kind: PatchKind::Wall,
            start,
            size,
            nbr_patch: None,
        });
        start += size;
    }

    Ok(PolyMeshRaw { points, faces, owner, neighbour, patches })
}

/// Build a carved [`HostMesh`] straight from a [`BlockSpec`] and a surface -
/// the in-memory twin of [`write_carved_case`]'s mesh half, with no file on
/// disk anywhere.
pub fn build_carved_mesh(b: &BlockSpec, surface: &Surface) -> Result<(HostMesh, CarveSummary)> {
    let block = Block::new(b)?;
    let cv = carve_block(&block, surface)?;
    let summary = cv.summary(&block);
    let raw = poly_mesh_raw_carved(&block, &cv)?;
    Ok((build_host_mesh(&raw)?, summary))
}

// ==========================================================================
//  Cut-cell meshing - SPEC-LIT §24
// ==========================================================================
//
// Written from SPEC-LIT §24 (via `surface::cutcell`, which owns the
// fractions/closure/merging maths) and §23.4 (nearest-triangle patch naming,
// carried over unchanged). No GPL-licensed source was consulted.
//
// `surface::cutcell::classify_cutcells` and `merge_small_cells` compute every
// per-cell fraction and the closure-defined cut face; everything here is
// mesh ASSEMBLY - turning those fractions into an actual `PolyMeshRaw` (and,
// through it, a `HostMesh`) the rest of ofgpu can read and solve on.
//
// The one construction worth explaining is `synthetic_quad`: a face's
// area vector and centroid (`Sf`, `Cf`) are what SPEC-LIT §24 actually
// specifies - not any particular polygon. `mesh::geometry::face_geometry`
// (private, and out of this agent's files) recovers `(Sf, Cf)` from a face's
// point list by triangulating about the vertex average; `synthetic_quad`
// picks the one planar SQUARE, centred at the target `Cf` with the target
// `Sf` as its own normal-times-area, that this triangulation reproduces
// EXACTLY (verified in the `cutcell_geometry` tests below by round-tripping
// every face through `build_host_mesh` and comparing `m.sf`/`m.cf` back to
// what was asked for). That is what carries §24.3's "exact by construction"
// claim through an actual point-based polyMesh file: closure holds to
// round-off no matter how approximate the fractions are, because it is
// asking the same triangulation identity that already makes an ordinary
// hex mesh close, just fed a square instead of the real face shape.
//
// Cell VOLUME is a different story: `compute_geometry`'s pyramid formula,
// `V = (1/3) sum_f Sf . Cf` once closure has cancelled the apex term, is only
// the TRUE volume when `Sf`/`Cf` come from the true physical boundary. For a
// cut cell they do not - `synthetic_quad` reproduces the right (Sf, Cf) pair
// for the FLUX and CLOSURE identities, not the right enclosed shape - so
// `V = theta_c * V_full` (§24.4, a definition, not something to re-derive)
// is written into `m.v`/`m.c` directly after `build_host_mesh` runs, and the
// two per-face coefficients that depend on cell centroids (the interpolation
// weight and the non-orthogonal split, SPEC-LIT §2.3/§2.4) are recomputed
// from the corrected centroids - `mesh::geometry`'s own versions of those
// two small, cited formulas are private, so `cutcell_face_coeffs` below
// re-derives them from the same citation rather than editing a file this
// agent does not own.

/// What cut-cell meshing did, for the CLI summary and the §24.6 gates.
#[derive(Debug, Clone)]
pub struct CutCellSummary {
    pub n_cells_block: usize,
    pub n_solid: usize,
    /// Fluid cells with `theta = 1` (never supersampled, or found fully
    /// fluid by the fine lattice).
    pub n_fluid_full: usize,
    /// Cells the supersample lattice found mixed, before merging.
    pub n_cut: usize,
    /// Slivers merged away (§24.5).
    pub n_merged: usize,
    /// Cells in the final mesh (`n_fluid_full + n_cut - n_merged`).
    pub n_cells_out: usize,
    pub n_internal_faces: usize,
    pub n_domain_faces: usize,
    /// New wall patches in written order: `(name, nFaces)`.
    pub wall_faces: Vec<(String, usize)>,
    pub supersample: usize,
    pub theta_min: Scalar,
}

/// A planar square centred at `cf`, normal `sf.normalised()`, sized so that
/// `mesh::geometry::face_geometry`'s triangulate-about-vertex-average
/// reproduces `(sf, cf)` EXACTLY.
///
/// Proof sketch (also pinned down by `cutcell_geometry::round_trips_sf_and_cf`
/// below): for four points arranged symmetrically about their own vertex
/// average `cf` as `cf +- h*u +- h*v` (`u, v, n` a right-handed orthonormal
/// frame), each of the four triangle-about-average sub-triangles contributes
/// the SAME cross product `2h^2 n` (the four are 90-degree rotations of one
/// another about `n`), so `Sf = 0.5 * sum = 0.5 * 4 * 2h^2 n = 4h^2 n`; taking
/// `h = sqrt(area)/2` makes `|Sf| = area` with direction `n`. The centroid is
/// `cf` by the same four-fold symmetry (the four sub-triangle centroids are
/// evenly spaced around it).
fn synthetic_quad(sf: Vec3, cf: Vec3) -> [Vec3; 4] {
    let area = sf.mag();
    let n = if area > 0.0 { sf / area } else { Vec3::new(0.0, 0.0, 1.0) };
    // Any unit vector not parallel to `n` gives a valid in-plane frame; the
    // axis threshold just avoids the near-parallel case where the cross
    // product below would lose precision.
    let helper = if n.x.abs() < 0.9 { Vec3::new(1.0, 0.0, 0.0) } else { Vec3::new(0.0, 1.0, 0.0) };
    let u = helper.cross(n).normalised();
    let v = n.cross(u);
    let h = 0.5 * area.max(0.0).sqrt();
    [cf + u * h + v * h, cf - u * h + v * h, cf - u * h - v * h, cf + u * h - v * h]
}

/// Push a fresh, unshared quad's four points and return their labels.
///
/// Faces are not welded to the block's own point grid (unlike castellation):
/// a reduced or cut face's points are synthetic in the first place, so there
/// is nothing real to share, and giving every face its own four points keeps
/// this assembly a single uniform code path instead of two.
fn push_cc_quad(points: &mut Vec<Vec3>, q: [Vec3; 4]) -> Vec<Label> {
    let base = points.len() as Label;
    points.extend_from_slice(&q);
    vec![base, base + 1, base + 2, base + 3]
}

// ---- SPEC-LIT §2.3/§2.4, re-derived (mesh::geometry's are private) --------

#[cfg(feature = "single")]
const CC_SMALL: Scalar = 1.0e-19;
#[cfg(not(feature = "single"))]
const CC_SMALL: Scalar = 1.0e-150;

/// SPEC-LIT §2.4's floor: at least 5% of `|d|`, so a face nearly tangent to
/// the line joining the two centres cannot blow the coefficient up.
const CC_NON_ORTH_FLOOR: Scalar = 0.05;

#[inline]
fn cc_weight_from_offsets(d_p: Scalar, d_n: Scalar) -> Scalar {
    let sum = d_p + d_n;
    if sum > CC_SMALL {
        d_n / sum
    } else {
        0.5
    }
}

/// SPEC-LIT §2.3: the interpolation weight that places the interpolated
/// value where the face plane cuts the line `P-N` (Jasak 1996 §3.3.1).
#[inline]
fn cc_interp_weight(sf: Vec3, cf: Vec3, c_p: Vec3, c_n: Vec3) -> Scalar {
    cc_weight_from_offsets(sf.dot(cf - c_p).abs(), sf.dot(c_n - cf).abs())
}

#[inline]
fn cc_floor_along(proj: Scalar, d: Vec3) -> Scalar {
    proj.max(CC_NON_ORTH_FLOOR * d.mag())
}

/// SPEC-LIT §2.4's over-relaxed non-orthogonal split, `(Delta, k)`.
#[inline]
fn cc_non_orth_split(sf: Vec3, d: Vec3) -> (Scalar, Vec3) {
    let nf = sf.normalised();
    let denom = cc_floor_along(nf.dot(d), d);
    if denom <= CC_SMALL {
        return (0.0, Vec3::ZERO);
    }
    let delta = 1.0 / denom;
    (delta, nf - d * delta)
}

/// Overwrite `m.v`/`m.c` with the §24.4 volumes/centroids `build_host_mesh`
/// cannot have derived correctly (see the module doc), then redo every
/// per-face coefficient that depends on a cell centroid. `m.sf`/`m.cf`/
/// `m.mag_sf` (and their boundary twins) are untouched - `synthetic_quad`
/// already made those exactly right.
///
/// The cut-cell path never emits a cyclic patch, so every boundary face is
/// uncoupled - the same branch `mesh::geometry::compute` takes for one,
/// leaving `b_non_orth_corr` at zero and `b_weights` at its uncoupled `1.0`.
fn override_cutcell_geometry(m: &mut HostMesh, v: Vec<Scalar>, c: Vec<Vec3>) {
    m.v = v;
    m.c = c;

    for f in 0..m.n_internal_faces {
        let (p, nb) = (m.owner[f] as usize, m.neighbour[f] as usize);
        let (sf, cf) = (m.sf[f], m.cf[f]);
        m.weights[f] = cc_interp_weight(sf, cf, m.c[p], m.c[nb]);
        let (delta, k) = cc_non_orth_split(sf, m.c[nb] - m.c[p]);
        m.delta_coeffs[f] = delta;
        m.non_orth_corr[f] = k;
    }

    for bf in 0..m.n_boundary_faces {
        let p = m.b_face_cells[bf] as usize;
        let (sf, cf) = (m.b_sf[bf], m.b_cf[bf]);
        let d_own = cf - m.c[p];
        let nf = sf.normalised();
        m.b_y[bf] = cc_floor_along(nf.dot(d_own), d_own);
        m.b_delta_coeffs[bf] = cc_non_orth_split(sf, d_own).0;
    }
}

/// The direction-`dir` (`-x +x -y +y -z +z`) full-face `Sf`/`Cf` of block
/// cell `c`'s own face - the real grid geometry `alpha_f` scales (§24.2).
/// `dir` must name a face `c` actually owns internally (0,2,4 mean "my own
/// -x/-y/-z face", 1,3,5 mean "my +x/+y/+z face towards `nbr`"); callers
/// always know which because they are iterating a specific neighbour
/// direction.
fn cc_full_face(g: &Grid, c: usize, dir: usize) -> (Vec3, Vec3) {
    let q = wall_quad(g, c, dir as u8);
    (quad_area(g, &q), quad_centre(g, &q))
}

/// Assemble a `PolyMeshRaw` from cut-cell fractions and a merge result -
/// SPEC-LIT §24's ANSWER: internal faces (dropped where merging made them
/// interior, alpha-scaled and combined where merging left more than one
/// contact between the same two surviving cells), the domain's own boundary
/// faces (alpha-scaled the same way), and one new wall face per `Cut` cell
/// (owned by whatever survivor it now belongs to), grouped into patches by
/// nearest-triangle surface patch exactly as castellation's §23.4 does.
fn cutcell_mesh_raw(
    block: &Block,
    field: &CutCellField,
    merge: &MergeResult,
    surf: &Surface,
) -> Result<(PolyMeshRaw, CutCellSummary, Vec<Scalar>, Vec<Vec3>)> {
    let g = &block.g;
    let n_cells = block.n_cells;

    // ---- live roots -> new sequential cell ids, in original i-fastest order
    let mut new_id = vec![-1i64; n_cells];
    let mut orig_of_new: Vec<usize> = Vec::new();
    for c in 0..n_cells {
        if field.cells[c].is_none() {
            continue;
        }
        let r = merge.root[c] as usize;
        if r == c && new_id[r] < 0 {
            new_id[r] = orig_of_new.len() as i64;
            orig_of_new.push(r);
        }
    }
    let n_cells_out = orig_of_new.len();

    // Per-new-cell volume/centroid (§24.4/§24.5's authoritative values,
    // independent of anything below) and a running signed closure sum, used
    // at the end of this function to close every cell EXACTLY regardless of
    // any cross-cell disagreement in the face-fraction assembly below - see
    // the module doc's note on why a per-cell independent classification
    // cannot guarantee two neighbours agree on their shared face, and why
    // that is fixed here rather than by forcing agreement upstream.
    let mut v_out = vec![0.0 as Scalar; n_cells_out];
    let mut c_out = vec![Vec3::ZERO; n_cells_out];
    for (new_c, &orig_root) in orig_of_new.iter().enumerate() {
        v_out[new_c] = merge.volume[orig_root];
        c_out[new_c] = merge.centroid[orig_root];
    }
    let mut closure_sum = vec![Vec3::ZERO; n_cells_out];

    let mut points: Vec<Vec3> = Vec::new();
    let mut faces: Vec<Vec<Label>> = Vec::new();
    let mut owner: Vec<Label> = Vec::new();
    let mut neighbour: Vec<Label> = Vec::new();

    // ---- internal faces: +x/+y/+z neighbours only, so each grid face is
    // visited once; duplicates that land on the same (owner,neighbour) pair
    // after merging are summed together (§24.5's "a merged cell can absorb
    // several slivers" - more than one contact with the same neighbour is
    // exactly that), which keeps the mesh's addressing free of duplicate
    // pairs (`mesh::geometry::check`'s `ldu_ordered` requires it).
    struct Contrib {
        sf: Vec3,
        cf_num: Vec3,
        w: Scalar,
    }
    let mut internal: HashMap<(i64, i64), Contrib> = HashMap::new();

    for c in 0..n_cells {
        if field.cells[c].is_none() {
            continue;
        }
        let (i, j, k) = g.decompose_cell(c);
        let plus_dirs: [(usize, Option<usize>); 3] = [
            (1, (i + 1 < g.nx).then(|| c + 1)),
            (3, (j + 1 < g.ny).then(|| c + g.nx)),
            (5, (k + 1 < g.nz).then(|| c + g.nx * g.ny)),
        ];
        for (dir, nbr) in plus_dirs {
            let Some(nbr) = nbr else { continue };
            if field.cells[nbr].is_none() {
                continue; // solid neighbour: no face at all
            }
            let (ra, rb) = (merge.root[c] as usize, merge.root[nbr] as usize);
            let (ida, idb) = (new_id[ra], new_id[rb]);
            if ida == idb {
                continue; // merged into the same cell: now interior, drop
            }
            let alpha = field.cells[c].as_ref().unwrap().alpha[dir];
            if alpha <= 0.0 {
                continue;
            }

            let (full_sf, full_cf) = cc_full_face(g, c, dir);
            let (lo, hi, sign): (i64, i64, Scalar) =
                if ida < idb { (ida, idb, 1.0) } else { (idb, ida, -1.0) };
            let sf = full_sf * (alpha * sign);
            let w = (alpha * full_sf.mag()).max(CC_SMALL);

            let entry = internal.entry((lo, hi)).or_insert(Contrib {
                sf: Vec3::ZERO,
                cf_num: Vec3::ZERO,
                w: 0.0,
            });
            entry.sf += sf;
            entry.cf_num += full_cf * w;
            entry.w += w;
        }
    }

    let mut int_faces: Vec<(i64, i64, Vec3, Vec3)> = internal
        .into_iter()
        .map(|((lo, hi), c)| {
            let cf = if c.w > 0.0 { c.cf_num / c.w } else { Vec3::ZERO };
            (lo, hi, c.sf, cf)
        })
        .collect();
    // Keys are unique pairs, so this sort alone gives strictly ascending
    // (owner, neighbour) - the upper-triangular order every gather kernel
    // assumes.
    int_faces.sort_by(|a, b| (a.0, a.1).cmp(&(b.0, b.1)));

    for (lo, hi, sf, cf) in &int_faces {
        faces.push(push_cc_quad(&mut points, synthetic_quad(*sf, *cf)));
        owner.push(*lo as Label);
        neighbour.push(*hi as Label);
        closure_sum[*lo as usize] += *sf;
        closure_sum[*hi as usize] -= *sf;
    }
    let n_internal = int_faces.len();

    // ---- domain boundary faces, alpha-scaled the same way, per block patch
    let mut domain_per_patch: Vec<Vec<(i64, Vec3, Vec3)>> = vec![Vec::new(); block.patches.len()];
    for (bi, patch) in block.patches.iter().enumerate() {
        let (na, _) = slot_dims(g, patch.slot);
        for idx in 0..patch.size {
            let sl = patch.slot_index(na, idx);
            let q = boundary_quad(g, patch.slot, sl);
            let Some(cf_frac) = &field.cells[q.own] else { continue };
            let alpha = cf_frac.alpha[patch.slot];
            if alpha <= 0.0 {
                continue;
            }
            let owner_id = new_id[merge.root[q.own] as usize];
            let sf = quad_area(g, &q) * alpha;
            closure_sum[owner_id as usize] += sf;
            domain_per_patch[bi].push((owner_id, sf, quad_centre(g, &q)));
        }
    }

    // ---- wall faces, grouped by surface patch -----------------------------
    //
    // One source of DECIDED wall area, nearest-triangle patched exactly as
    // castellation's §23.4 carve does: any live cell's own axis face whose
    // neighbour is SOLID but whose `alpha_f > 0` (this cell's own side is
    // open there). This is the degenerate case a grid-aligned surface
    // produces exactly (the interface coincides with the shared plane, so
    // `alpha_f` on one side is a clean 0 or 1 rather than a fraction) and the
    // general case a genuinely `Cut` cell's own axis face touching a wholly
    // solid neighbour also needs.
    //
    // Every `Cut` cell's own §24.3 cut face is deliberately NOT emitted here:
    // `classify_cutcells` computes it from each cell's OWN independent
    // supersample array, so two neighbouring `Cut` cells (or a `Cut` cell and
    // a neighbour that independently classified as fully `Fluid`) can
    // disagree, by a small amount, on the alpha_f of the face they share -
    // the "one sample lattice, no seams" guarantee (§24.4) is only stated for
    // the DEGENERATE all-fluid/all-solid case, not for two genuinely mixed
    // neighbours. Left uncorrected, that disagreement is exactly the leftover
    // the reconciliation pass below closes: after every internal, domain and
    // solid-adjacent face above is decided, whatever a cell's own faces fail
    // to sum to zero becomes ITS cut face, by the same closure identity
    // §24.3 states - just measured against what was ACTUALLY assigned rather
    // than assumed in advance.
    let hint = ((g.xn[g.nx] - g.xn[0]).abs() / g.nx as Scalar
        + (g.yn[g.ny] - g.yn[0]).abs() / g.ny as Scalar
        + (g.zn[g.nz] - g.zn[0]).abs() / g.nz as Scalar)
        / 3.0;
    let tidx = TriIndex::new(surf, hint.max(Scalar::MIN_POSITIVE))?;

    let mut per_surface_patch: Vec<Vec<(i64, Vec3, Vec3)>> = vec![Vec::new(); surf.patch_names.len()];
    // The first `Cut` constituent's own interface centroid found for each
    // surviving cell, so the reconciliation pass has a physically meaningful
    // place to put a correction rather than the cell centroid.
    let mut cut_position: Vec<Option<Vec3>> = vec![None; n_cells_out];

    for c in 0..n_cells {
        let Some(cf) = &field.cells[c] else { continue };
        let owner_id = new_id[merge.root[c] as usize];

        if cf.state == CellState::Cut && cut_position[owner_id as usize].is_none() {
            cut_position[owner_id as usize] = Some(cf.cut_cf);
        }

        let (i, j, k) = g.decompose_cell(c);
        let all_dirs: [(usize, Option<usize>); 6] = [
            (0, (i > 0).then(|| c - 1)),
            (1, (i + 1 < g.nx).then(|| c + 1)),
            (2, (j > 0).then(|| c - g.nx)),
            (3, (j + 1 < g.ny).then(|| c + g.nx)),
            (4, (k > 0).then(|| c - g.nx * g.ny)),
            (5, (k + 1 < g.nz).then(|| c + g.nx * g.ny)),
        ];
        for (dir, nbr) in all_dirs {
            let Some(nbr) = nbr else { continue }; // domain edge: `domain_per_patch`'s job
            if field.cells[nbr].is_some() {
                continue; // fluid/cut neighbour: the internal-face loop's job
            }
            let alpha = cf.alpha[dir];
            if alpha <= 0.0 {
                continue;
            }
            let (full_sf, full_cf) = cc_full_face(g, c, dir);
            let (t, _) = tidx.nearest_triangle(full_cf);
            let patch_id = surf.tri_patch[t] as usize;
            let sf = full_sf * alpha;
            closure_sum[owner_id as usize] += sf;
            per_surface_patch[patch_id].push((owner_id, sf, full_cf));
        }
    }

    // ---- reconciliation: close every surviving cell EXACTLY ---------------
    // A cell whose decided faces already sum to (round-off) zero gets no
    // correction at all - true for every plain `Fluid` cell and for a `Cut`
    // cell whose neighbours all happened to agree with it.
    for nc in 0..n_cells_out {
        let leftover = -closure_sum[nc];
        let scale = v_out[nc].max(CC_SMALL).powf(2.0 / 3.0);
        if leftover.mag() < 1.0e-9 * scale {
            continue;
        }
        let position = cut_position[nc].unwrap_or(c_out[nc]);
        let (t, _) = tidx.nearest_triangle(position);
        let patch_id = surf.tri_patch[t] as usize;
        per_surface_patch[patch_id].push((nc as i64, leftover, position));
    }

    let mut patches: Vec<PatchInfo> = Vec::with_capacity(block.patches.len() + surf.patch_names.len());
    let mut start = 0usize;
    for (bi, patch) in block.patches.iter().enumerate() {
        let list = &domain_per_patch[bi];
        for &(owner_id, sf, cf) in list {
            faces.push(push_cc_quad(&mut points, synthetic_quad(sf, cf)));
            owner.push(owner_id as Label);
        }
        patches.push(PatchInfo {
            name: patch.name.clone(),
            type_name: patch.type_name.clone(),
            kind: PatchKind::from_type(&patch.type_name),
            start,
            size: list.len(),
            nbr_patch: None,
        });
        start += list.len();
    }

    let mut wall_faces: Vec<(String, usize)> = Vec::new();
    for (p, list) in per_surface_patch.into_iter().enumerate() {
        if list.is_empty() {
            continue;
        }
        let name = surf.patch_names[p].clone();
        if block.patches.iter().any(|q| q.name == name) {
            return Err(Error::Mesh(format!(
                "cutcell: surface patch '{name}' collides with a domain patch of the \
                 same name - rename the STL solid (or use -stl name=path)"
            )));
        }
        for &(owner_id, sf, cf) in &list {
            faces.push(push_cc_quad(&mut points, synthetic_quad(sf, cf)));
            owner.push(owner_id as Label);
        }
        wall_faces.push((name.clone(), list.len()));
        patches.push(PatchInfo {
            name,
            type_name: "wall".to_string(),
            kind: PatchKind::Wall,
            start,
            size: list.len(),
            nbr_patch: None,
        });
        start += list.len();
    }

    let n_fluid_full = field
        .cells
        .iter()
        .flatten()
        .filter(|c| c.state == CellState::Fluid)
        .count();

    let summary = CutCellSummary {
        n_cells_block: n_cells,
        n_solid: field.n_solid,
        n_fluid_full,
        n_cut: field.n_cut,
        n_merged: merge.n_merged,
        n_cells_out,
        n_internal_faces: n_internal,
        n_domain_faces: domain_per_patch.iter().map(Vec::len).sum(),
        wall_faces,
        supersample: field.s,
        theta_min: DEFAULT_THETA_MIN,
    };

    Ok((PolyMeshRaw { points, faces, owner, neighbour, patches }, summary, v_out, c_out))
}

/// Build a runnable cut-cell [`HostMesh`] straight from a [`BlockSpec`] and a
/// surface (SPEC-LIT §24), with no file on disk anywhere. Uses
/// [`DEFAULT_SUPERSAMPLE`] and [`DEFAULT_THETA_MIN`]; see
/// [`build_cutcell_mesh_with`] to override either.
pub fn build_cutcell_mesh(b: &BlockSpec, surface: &Surface) -> Result<(HostMesh, CutCellSummary)> {
    build_cutcell_mesh_with(b, surface, DEFAULT_SUPERSAMPLE, DEFAULT_THETA_MIN)
}

/// Validate a surface against a block, classify and merge it (SPEC-LIT §24),
/// and assemble the raw mesh - the common core of [`build_cutcell_mesh_with`]
/// (which turns the raw mesh into a runnable [`HostMesh`]) and
/// [`write_cutcell_case`] (which writes it to disk directly, since the
/// override the in-memory path applies has no hook into the ordinary
/// polyMesh-reading load path - see the module doc).
fn cutcell_case_raw(
    b: &BlockSpec,
    surface: &Surface,
    s: usize,
    theta_min: Scalar,
) -> Result<(Block, PolyMeshRaw, CutCellSummary, Vec<Scalar>, Vec<Vec3>)> {
    let block = Block::new(b)?;
    let g = &block.g;

    surface.require_closed()?;

    let (dlo, dhi) = (
        Vec3::new(g.xn[0], g.yn[0], g.zn[0]),
        Vec3::new(g.xn[g.nx], g.yn[g.ny], g.zn[g.nz]),
    );
    let (slo, shi) = surface.bbox;
    if shi.x < dlo.x || slo.x > dhi.x || shi.y < dlo.y || slo.y > dhi.y
        || shi.z < dlo.z || slo.z > dhi.z
    {
        return Err(Error::Mesh(format!(
            "cutcell: the surface (bounds [{} {} {}] to [{} {} {}]) lies entirely \
             outside the block ([{} {} {}] to [{} {} {}]) - nothing to cut",
            fmt_g(slo.x), fmt_g(slo.y), fmt_g(slo.z),
            fmt_g(shi.x), fmt_g(shi.y), fmt_g(shi.z),
            fmt_g(dlo.x), fmt_g(dlo.y), fmt_g(dlo.z),
            fmt_g(dhi.x), fmt_g(dhi.y), fmt_g(dhi.z),
        )));
    }

    let axes = BlockAxes { xn: &g.xn, yn: &g.yn, zn: &g.zn };
    let field = classify_cutcells(&axes, surface, s)?;
    if field.n_fluid + field.n_cut == 0 {
        return Err(Error::Mesh(format!(
            "cutcell: every one of the {} cells is solid - the surface swallows \
             the whole domain",
            field.n_solid + field.n_fluid + field.n_cut
        )));
    }

    let merge = merge_small_cells(&field, theta_min)?;
    let (raw, mut summary, v_out, c_out) = cutcell_mesh_raw(&block, &field, &merge, surface)?;
    summary.theta_min = theta_min;

    Ok((block, raw, summary, v_out, c_out))
}

/// [`build_cutcell_mesh`] with an explicit supersample size and merge
/// threshold.
pub fn build_cutcell_mesh_with(
    b: &BlockSpec,
    surface: &Surface,
    s: usize,
    theta_min: Scalar,
) -> Result<(HostMesh, CutCellSummary)> {
    let (_block, raw, summary, v_out, c_out) = cutcell_case_raw(b, surface, s, theta_min)?;
    let mut mesh = build_host_mesh(&raw)?;
    override_cutcell_geometry(&mut mesh, v_out, c_out);
    Ok((mesh, summary))
}

/// [`write_carved_case`]'s cut-cell twin: a complete runnable case (polyMesh,
/// `constant/`, `system/`, `0/`) built by cutting the block against `surface`
/// with fractional volumes and areas (SPEC-LIT §24) instead of castellating
/// it away. New wall patches carry exactly the wall boundary conditions the
/// uncarved writer gives its walls, the same way castellation's do.
///
/// *DESIGN*: the disk copy's cell VOLUMES are whatever
/// `mesh::geometry::compute` derives from the synthetic per-face points on
/// its own ordinary read-back path (see the module doc for why that is not
/// `theta_c * V_full` exactly) - there is no hook to run
/// `override_cutcell_geometry` on a case `ofgpu-k-epsilon <dir>` loads from
/// files, only on a mesh built in-process by [`build_cutcell_mesh`]. The
/// fractions, the closure-exact cut face and every AREA (`Sf`, hence every
/// flux) are exact either way; only the volume used in `ofgpu-k-epsilon`'s
/// own run is the pyramid-decomposition approximation. `cutcell_geometry`'s
/// tests below measure how far that approximation is from `theta_c*V_full`.
#[allow(clippy::too_many_arguments)]
pub fn write_cutcell_case(
    case_dir: &Path,
    kind: CaseKind,
    nx: usize,
    ny: usize,
    nz: usize,
    surface: &Surface,
    s: usize,
    theta_min: Scalar,
) -> Result<CutCellSummary> {
    write_cutcell_case_impl(
        case_dir, kind, nx, ny, nz, surface, s, theta_min, WallTreatment::Standard, None, false,
    )
}

/// [`write_cutcell_case`], with SPEC-LIT §29.1's `wallTreatment` preset (route
/// c) expanded into the cut-cell case's `0/` fields instead of the hardcoded
/// `standard` row - the new wall faces the cut follows the surface with
/// (`summary.wall_faces`) get exactly the same row as the block's own walls,
/// so "carved STL wall patches follow the same preset" holds for cut cells
/// too, not only castellation.
#[allow(clippy::too_many_arguments)]
pub fn write_cutcell_case_with_wall_model(
    case_dir: &Path,
    kind: CaseKind,
    nx: usize,
    ny: usize,
    nz: usize,
    surface: &Surface,
    s: usize,
    theta_min: Scalar,
    wall: WallTreatment,
    roughness: Option<Roughness>,
) -> Result<CutCellSummary> {
    write_cutcell_case_impl(case_dir, kind, nx, ny, nz, surface, s, theta_min, wall, roughness, true)
}

#[allow(clippy::too_many_arguments)]
fn write_cutcell_case_impl(
    case_dir: &Path,
    kind: CaseKind,
    nx: usize,
    ny: usize,
    nz: usize,
    surface: &Surface,
    s: usize,
    theta_min: Scalar,
    wall: WallTreatment,
    roughness: Option<Roughness>,
    thermal_wall: bool,
) -> Result<CutCellSummary> {
    if nx < 1 || ny < 1 || nz < 1 {
        return Err(Error::Config(format!("generate_cases: bad resolution {nx} x {ny} x {nz}")));
    }
    if kind == CaseKind::DamBreak {
        // Not a substitutable setting - VOF/free-surface interaction with a
        // cut cell simply is not implemented yet, so there is no fallback to
        // hand `-permissive`, unlike the ordinary §13.4 contract cases.
        return Err(Error::Config(
            "cutcell: damBreak has no cut-cell path yet (no VOF/free-surface \
             interaction with a cut cell) - castellate the surface instead \
             (drop -cutcell), or use channel|cavity|step|big|plume"
                .to_string(),
        ));
    }

    let b = case_block_spec(kind, nx, ny, nz);
    let (block, raw, summary, _v_out, c_out) = cutcell_case_raw(&b, surface, s, theta_min)?;

    write_raw_poly_mesh_ascii(case_dir, &raw, summary.n_cells_out)?;
    write_system(case_dir)?;

    let (nu, u_ref, half_height) = case_run_params(kind, &b, &block);
    let extra = if buoyant_case(kind) {
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
    if buoyant_case(kind) {
        write_gravity(case_dir)?;
    }

    let wall_names: Vec<String> = summary.wall_faces.iter().map(|(n, _)| n.clone()).collect();
    let fields = build_cutcell_fields(
        kind, &c_out, &block, &wall_names, nu, u_ref, half_height, 1, wall, roughness, thermal_wall,
    )?;
    write_fields(case_dir, &fields)?;

    println!(
        "[cutcell] {} x {} x {} block, s = {}, theta_min = {}: {} solid, {} fluid, \
         {} cut ({} merged) -> {} cells",
        nx, ny, nz, summary.supersample, summary.theta_min,
        summary.n_solid, summary.n_fluid_full, summary.n_cut, summary.n_merged,
        summary.n_cells_out
    );
    if summary.wall_faces.is_empty() {
        println!("[cutcell] no new wall faces - the surface encloses no cell centres");
    } else {
        for (patch, n) in &summary.wall_faces {
            println!("[cutcell] new wall patch {patch}: {n} face(s)");
        }
    }

    Ok(summary)
}

/// Write `case_dir/constant/polyMesh` straight from a [`PolyMeshRaw`) - the
/// generic twin of [`write_poly_mesh`]/[`write_carved_poly_mesh`] for a mesh
/// whose faces are not all quads sharing one block's point grid. Uses the
/// same `TextOut`/`FoamFile` machinery; unlike the block writers it does not
/// re-verify winding face by face (there is no `Quad`/owner-cell-centre pair
/// to check it against here) - `mesh::geometry::compute`, run the moment
/// anything reads this mesh back, is what enforces `owner < neighbour`, and
/// closure is `check()`'s job, not a per-face structural one.
fn write_raw_poly_mesh_ascii(case_dir: &Path, raw: &PolyMeshRaw, n_cells: usize) -> Result<()> {
    let dir = case_dir.join("constant").join("polyMesh");
    fs::create_dir_all(&dir).path(&dir)?;

    let n_faces = raw.faces.len();
    let n_internal = raw.neighbour.len();
    let note = format!(
        "nPoints:{}  nCells:{n_cells}  nFaces:{n_faces}  nInternalFaces:{n_internal}",
        raw.points.len()
    );

    {
        let mut os = TextOut::create(&dir.join("points"))?;
        write_foam_header(&mut os, "vectorField", "points", "")?;
        os.num(raw.points.len())?;
        os.s("\n(\n")?;
        for p in &raw.points {
            os.c('(')?;
            os.real(p.x)?;
            os.c(' ')?;
            os.real(p.y)?;
            os.c(' ')?;
            os.real(p.z)?;
            os.s(")\n")?;
        }
        os.s(")")?;
        write_foam_footer(&mut os)?;
        os.finish()?;
    }
    {
        let mut os = TextOut::create(&dir.join("faces"))?;
        write_foam_header(&mut os, "faceList", "faces", "")?;
        os.num(n_faces)?;
        os.s("\n(\n")?;
        for f in &raw.faces {
            os.num(f.len())?;
            os.c('(')?;
            for (i, &p) in f.iter().enumerate() {
                if i > 0 {
                    os.c(' ')?;
                }
                os.num(p as usize)?;
            }
            os.s(")\n")?;
        }
        os.s(")")?;
        write_foam_footer(&mut os)?;
        os.finish()?;
    }
    {
        let mut os = TextOut::create(&dir.join("owner"))?;
        write_foam_header(&mut os, "labelList", "owner", &note)?;
        os.num(n_faces)?;
        os.s("\n(\n")?;
        for &o in &raw.owner {
            os.num(o as usize)?;
            os.c('\n')?;
        }
        os.s(")")?;
        write_foam_footer(&mut os)?;
        os.finish()?;
    }
    {
        let mut os = TextOut::create(&dir.join("neighbour"))?;
        write_foam_header(&mut os, "labelList", "neighbour", &note)?;
        os.num(n_internal)?;
        os.s("\n(\n")?;
        for &n in &raw.neighbour {
            os.num(n as usize)?;
            os.c('\n')?;
        }
        os.s(")")?;
        write_foam_footer(&mut os)?;
        os.finish()?;
    }
    {
        let mut os = TextOut::create(&dir.join("boundary"))?;
        write_foam_header(&mut os, "polyBoundaryMesh", "boundary", "")?;
        os.num(raw.patches.len())?;
        os.s("\n(\n")?;
        for p in &raw.patches {
            os.s("    ")?;
            os.s(&p.name)?;
            os.s("\n    {\n        type            ")?;
            os.s(&p.type_name)?;
            os.s(";\n        nFaces          ")?;
            os.num(p.size)?;
            os.s(";\n        startFace       ")?;
            os.num(n_internal + p.start)?;
            os.s(";\n    }\n")?;
        }
        os.s(")")?;
        write_foam_footer(&mut os)?;
        os.finish()?;
    }

    Ok(())
}

/// A cut-cell twin of `build_initial_fields`: the same field names,
/// dimensions and wall boundary conditions (so a cut-cell case runs through
/// exactly the same solver code path a castellated one does), seeded at each
/// merged cell's OWN centroid (`centroids`, from [`cutcell_mesh_raw`]) rather
/// than through `Carved`'s per-block-cell indexing, which does not apply
/// once cells have been merged. Boundary velocity is a single uniform value
/// per patch rather than the uncarved writer's per-face profile, since a
/// cut-cell boundary face has no regular `(j,k)` index to build that profile
/// from.
#[allow(clippy::too_many_arguments)]
fn build_cutcell_fields(
    kind: CaseKind,
    centroids: &[Vec3],
    block: &Block,
    wall_patch_names: &[String],
    nu: Scalar,
    u_ref: Scalar,
    half_height: Scalar,
    wall_normal: usize,
    wall: WallTreatment,
    roughness: Option<Roughness>,
    thermal_wall: bool,
) -> Result<InMemoryFields> {
    let _ = nu; // carried for signature parity with `build_initial_fields`; unused here
    let n_cells = centroids.len();
    let plume = buoyant_case(kind);
    let t_inlet = if kind == CaseKind::Room { ROOM_T_INLET } else { PLUME_T_INLET };
    let cavity = kind == CaseKind::Cavity;

    struct FP<'a> {
        name: &'a str,
        type_name: &'a str,
    }
    let mut fps: Vec<FP> = block
        .patches
        .iter()
        .map(|p| FP { name: &p.name, type_name: &p.type_name })
        .collect();
    for name in wall_patch_names {
        fps.push(FP { name, type_name: "wall" });
    }

    let i_turb = 0.05 as Scalar;
    let cmu = 0.09 as Scalar;
    let k0 = 1.5 * (i_turb * u_ref) * (i_turb * u_ref);
    let l = 0.07 * half_height * 2.0;
    let eps0 = cmu.powf(0.75) * k0.powf(1.5) / l;
    let omega0 = k0.sqrt() / (cmu.powf(0.25) * l);

    let g = &block.g;
    let (lo, hi) = centre_bounds(g);
    let lx = (hi.x - lo.x).max(1e-30);
    let ly = (hi.y - lo.y).max(1e-30);

    let mut internal = Vec::with_capacity(n_cells);
    for &centre in centroids {
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
    for fp in &fps {
        let pk = PatchKind::from_type(fp.type_name);
        let s = if pk == PatchKind::Empty {
            patch_spec("empty")
        } else if pk == PatchKind::Cyclic {
            // SPEC-LIT §31.1: every field carries `cyclic` on a cyclic
            // patch - `kinds_from_patches` (field.rs) gives the mesh the
            // last word at runtime, but the written file has to agree
            // with `constant/polyMesh/boundary` for a round trip.
            patch_spec("cyclic")
        } else if cavity && fp.name == "movingWall" {
            let mut s = patch_spec("fixedValue");
            s.value_v = vec![Vec3::new(u_ref, 0.0, 0.0)];
            s
        } else if pk == PatchKind::Wall {
            let mut s = patch_spec("noSlip");
            s.value_v = vec![Vec3::ZERO];
            s
        } else if fp.name == "inlet" {
            let mut s = patch_spec("fixedValue");
            s.value_v = vec![Vec3::new(u_ref, 0.0, 0.0)];
            s
        } else if plume {
            let mut s = patch_spec("inletOutlet");
            s.inlet_value_v = vec![Vec3::ZERO];
            s.value_v = vec![Vec3::ZERO];
            s
        } else {
            patch_spec("zeroGradient")
        };
        u.boundary.insert(fp.name.to_string(), s);
    }

    let mut scalars: Vec<RawScalarField> = Vec::new();

    if plume {
        let mut pf = RawScalarField {
            name: "p".to_string(),
            dimensions: "[0 2 -2 0 0 0 0]".to_string(),
            internal: vec![0.0 as Scalar; n_cells],
            boundary: BTreeMap::new(),
            boundary_patterns: Vec::new(),
        };
        for fp in &fps {
            let pk = PatchKind::from_type(fp.type_name);
            let s = if pk == PatchKind::Empty {
                patch_spec("empty")
            } else if pk == PatchKind::Cyclic {
                patch_spec("cyclic")
            } else if fp.name == "outlet" {
                let mut s = patch_spec("fixedValue");
                s.value = vec![0.0];
                s
            } else {
                patch_spec("zeroGradient")
            };
            pf.boundary.insert(fp.name.to_string(), s);
        }
        scalars.push(pf);

        let mut t = RawScalarField {
            name: "T".to_string(),
            dimensions: "[0 0 0 1 0 0 0]".to_string(),
            internal: vec![PLUME_T_AMBIENT; n_cells],
            boundary: BTreeMap::new(),
            boundary_patterns: Vec::new(),
        };
        for fp in &fps {
            let pk = PatchKind::from_type(fp.type_name);
            let s = if pk == PatchKind::Empty {
                patch_spec("empty")
            } else if pk == PatchKind::Cyclic {
                patch_spec("cyclic")
            } else if fp.name == "inlet" {
                let mut s = patch_spec("fixedValue");
                s.value = vec![t_inlet];
                s
            } else if pk == PatchKind::Wall {
                // SPEC-LIT §29.3: every wallTreatment row except `lowRe`
                // applies the thermal wall function on a wall, when the
                // energy equation is solved AND a wall model was actually
                // asked for (`thermal_wall`) - the legacy adiabatic default
                // (`write_case`/`build_case`, no `-wallModel`) stays exactly
                // `zeroGradient`.
                match thermal_wall.then(|| wall.thermal_type()).flatten() {
                    Some(t) => patch_spec(t),
                    None => patch_spec("zeroGradient"),
                }
            } else {
                let mut s = patch_spec("inletOutlet");
                s.inlet_value = vec![PLUME_T_AMBIENT];
                s.value = vec![PLUME_T_AMBIENT];
                s
            };
            t.boundary.insert(fp.name.to_string(), s);
        }
        scalars.push(t);
    }

    let specs: [(&str, &str, Scalar); 4] = [
        ("k", "[0 2 -2 0 0 0 0]", k0),
        ("epsilon", "[0 2 -3 0 0 0 0]", eps0),
        ("omega", "[0 0 -1 0 0 0 0]", omega0),
        ("nut", "[0 2 -1 0 0 0 0]", 0.0),
    ];
    for (name, dims, value) in specs {
        let mut f = RawScalarField {
            name: name.to_string(),
            dimensions: dims.to_string(),
            internal: vec![value; n_cells],
            boundary: BTreeMap::new(),
            boundary_patterns: Vec::new(),
        };
        for fp in &fps {
            let pk = PatchKind::from_type(fp.type_name);
            let s = if pk == PatchKind::Empty {
                patch_spec("empty")
            } else if pk == PatchKind::Cyclic {
                patch_spec("cyclic")
            } else if pk == PatchKind::Wall {
                let type_name = wall_row_type(name, wall);
                let mut s = patch_spec(type_name);
                // SPEC-LIT §33.2: `epsilon`'s `lowRe` completion is
                // `fixedValue`, and the value it fixes is the homogeneous
                // Dirichlet the model needs (0), not the domain's
                // equilibrium epsilon - every other row's value here is a
                // wall-function seed a model overwrites, so the domain
                // value is the right seed for THOSE, and wrong for this one.
                s.value = vec![if type_name == "fixedValue" { 0.0 } else { value }];
                apply_roughness(&mut s, name, wall, roughness);
                s
            } else if name == "nut" {
                let mut s = patch_spec("calculated");
                s.value = vec![0.0];
                s
            } else if fp.name == "inlet" {
                let mut s = patch_spec("fixedValue");
                s.value = vec![value];
                s
            } else {
                patch_spec("zeroGradient")
            };
            f.boundary.insert(fp.name.to_string(), s);
        }
        scalars.push(f);
    }

    Ok(InMemoryFields { scalars, vectors: vec![u] })
}

// ==========================================================================
//  Cases
// ==========================================================================

/// The ready-to-run cases `generate_cases` knows how to build.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CaseKind {
    /// 10 x 10 x 3 m test room: hot air in through the whole -x wall, a
    /// 2 x 2 m door in the +x wall as the single pressure opening; interior
    /// baffles come from an `-stl` carve.
    Room,
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
            Self::Room => "room",
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
            "room" => Some(Self::Room),
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
            // 10 x 10 x 3 m at 0.1 m: 300k cells; centres at z = ..., 2.75,
            // 2.85, 2.95, so a near-ceiling slice sits in the hot layer.
            Self::Room => (100, 100, 30),
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
                    "unknown case '{s}' (channel|cavity|step|big|plume|room|damBreak)"
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

/// Room-case inlet: 300 degC air at 2 m/s through the whole -x wall.
const ROOM_T_INLET: Scalar = 573.15;
const ROOM_INLET_U: Scalar = 2.0;

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
/// The kinds that carry gravity, a temperature equation and the buoyant
/// `constant/` dictionaries. Field assembly treats them identically; only
/// the numbers (inlet temperature, reference velocity) differ per kind.
fn buoyant_case(kind: CaseKind) -> bool {
    matches!(kind, CaseKind::Plume | CaseKind::Room)
}

/// The room's door: 2 m wide x 2 m tall, centred on the +x wall at y = 5,
/// floor-mounted (z 0..2). The window mechanism turns `wallXMax` into
/// `outlet` (the door) plus the remaining wall.
fn room_door_window(b: &BlockSpec) -> PatchWindow {
    let (j0, j1) = centred_cell_range(&graded_nodes(&b.y), 5.0, 2.0);
    let (k0, k1) = centred_cell_range(&graded_nodes(&b.z), 1.0, 2.0);

    PatchWindow {
        slot: 1,
        lo: [j0, k0],
        hi: [j1, k1],
        name: "outlet".to_string(),
        type_name: "patch".to_string(),
    }
}

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
            // is the physical choice for a genuinely developed channel
            // (SPEC-LIT §31.1: pass `-cyclic x` to get it, or see
            // `write_case_cyclic`) - this DEFAULT preset stays inlet/outlet,
            // matching every earlier `channel` case this reader has built.
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
        CaseKind::Room => {
            // Hot-air serpentine test room. The whole -x wall blows 2 m/s of
            // 300 degC air; the ONLY pressure opening is the door window cut
            // from `wallXMax` below - the same single-opening reasoning as
            // the plume. Baffles are carved from an STL, not geometry here.
            b.x.lo = 0.0;
            b.x.hi = 10.0;
            b.y.lo = 0.0;
            b.y.hi = 10.0;
            b.z.lo = 0.0;
            b.z.hi = 3.0;
            (
                ["inlet", "wallXMax", "wallYMin", "wallYMax", "floor", "ceiling"],
                ["patch", "wall", "wall", "wall", "wall", "wall"],
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
        b.windows = vec![plume_inlet_window(&b)];
    }
    if kind == CaseKind::Room {
        b.windows = vec![room_door_window(&b)];
    }

    b
}

/// `(nu, u_ref, half_height)` for a case's initial fields and
/// `constant/transportProperties` - shared by `write_case_impl` and
/// [`build_case`] so the file path and the in-memory path can never pick
/// different numbers for the same case.
fn case_run_params(kind: CaseKind, b: &BlockSpec, block: &Block) -> (Scalar, Scalar, Scalar) {
    // Air for the plume; every other case keeps the 1e-5 it has always had.
    let nu: Scalar = if buoyant_case(kind) { 1.5e-5 } else { 1e-5 };

    let inlet = b.windows.first().map(|w| window_extent(&block.g, w));
    let (u_ref, half_height): (Scalar, Scalar) = match kind {
        CaseKind::Channel => (1.0, 1.0),
        CaseKind::Step => (10.0, 1.0),
        CaseKind::Cavity => (1.0, 0.05),
        CaseKind::Big => (1.0, 0.5),
        // Unreachable for a runnable case: the two-phase case is handled
        // before this is ever called. Present because the compiler counts
        // arms, not reachability.
        CaseKind::DamBreak => (0.0, DAM_BREAK_A),
        // The plume has no wall-normal profile; `half_height` only feeds the
        // inlet mixing length, which scales with the burner, not the room. The
        // two snapped sides differ by well under a percent, so their mean is
        // the honest single number to hand it.
        // The room's inlet spans the whole 10 x 3 m wall; the mixing
        // length scales with that opening's half-height.
        CaseKind::Room => (ROOM_INLET_U, 1.5),
        CaseKind::Plume => {
            let side = match inlet {
                Some((fx, fy)) => 0.25 * ((fx[1] - fx[0]) + (fy[1] - fy[0])),
                None => 0.5 * PLUME_INLET_SIDE,
            };
            (PLUME_INLET_U, side)
        }
    };

    (nu, u_ref, half_height)
}

/// Write a complete runnable case: polyMesh, `constant/`, `system/` and a `0/`
/// directory whose fields are real per-cell profiles rather than a uniform
/// guess.
pub fn write_case(case_dir: &Path, kind: CaseKind, nx: usize, ny: usize, nz: usize) -> Result<()> {
    write_case_impl(case_dir, kind, nx, ny, nz, None, WallTreatment::Standard, None, false, &[])
        .map(|_| ())
}

/// [`write_case`], with the two opposite patches of each of `axes` (0=x,
/// 1=y, 2=z) declared a cyclic pair instead of whatever `kind`'s own preset
/// would put there (SPEC-LIT §31.1, more than one axis per §34.2) -
/// `ofgpu-generate-mesh <case> <dir> -cyclic <axis>` repeatable.
/// `channel`'s own comment in [`case_block_spec`] names exactly this gap: "a
/// streamwise cyclic pair would be the physical choice but cyclics need a
/// coupled patch pair" - `-cyclic x` on `channel` is that pair, turning
/// `inlet`/`outlet` into a periodic streamwise direction instead; `-cyclic x
/// -cyclic z` on `channel` (already `empty` front/back per §34.1) makes it a
/// plane channel periodic in both wall-parallel directions.
pub fn write_case_cyclic(
    case_dir: &Path,
    kind: CaseKind,
    nx: usize,
    ny: usize,
    nz: usize,
    axes: &[usize],
) -> Result<()> {
    write_case_impl(
        case_dir,
        kind,
        nx,
        ny,
        nz,
        None,
        WallTreatment::Standard,
        None,
        false,
        axes,
    )
    .map(|_| ())
}

/// [`write_case_cyclic`] plus [`write_case_with_wall_model`]'s wall-treatment
/// preset, for a cyclic case whose remaining (non-cyclic) patches are walls -
/// `ofgpu-generate-mesh <case> <dir> -cyclic <axis> -wallModel <preset>`.
#[allow(clippy::too_many_arguments)]
pub fn write_case_cyclic_with_wall_model(
    case_dir: &Path,
    kind: CaseKind,
    nx: usize,
    ny: usize,
    nz: usize,
    axes: &[usize],
    wall: WallTreatment,
    roughness: Option<Roughness>,
) -> Result<()> {
    write_case_impl(case_dir, kind, nx, ny, nz, None, wall, roughness, true, axes)
        .map(|_| ())
}

/// [`write_case`], with SPEC-LIT §29.1's `wallTreatment` preset (route c)
/// expanded into the generated `0/` fields instead of the hardcoded
/// `standard` row - `-wallModel <preset> [-Ks x [-Cs y]]` on
/// `ofgpu-generate-mesh`. `standard` (passed explicitly or via [`write_case`])
/// reproduces today's hardcoded row exactly, `k`/`epsilon`/`omega`/`nut`
/// string for string.
pub fn write_case_with_wall_model(
    case_dir: &Path,
    kind: CaseKind,
    nx: usize,
    ny: usize,
    nz: usize,
    wall: WallTreatment,
    roughness: Option<Roughness>,
) -> Result<()> {
    write_case_impl(case_dir, kind, nx, ny, nz, None, wall, roughness, true, &[]).map(|_| ())
}

/// Build a complete runnable case's mesh and `0/` fields entirely in memory -
/// no `constant/polyMesh`, no `constant/transportProperties`, no `system/`,
/// just the two things a solver actually needs to start from: the geometry
/// and the initial fields. [`write_case`] is this plus every dictionary
/// serialised to disk; both go through [`case_block_spec`],
/// [`case_run_params`] and the same field builders, so the two cannot drift
/// apart on what a case's initial state is.
///
/// The dam break case is two-phase and carries `alpha.water`, `U` and
/// `p_rgh` rather than the single-phase fields below; every other case kind
/// carries `U` and, for the plume only, `p`, `T` and the four turbulence
/// fields.
pub fn build_case(kind: CaseKind, nx: usize, ny: usize, nz: usize) -> Result<(HostMesh, InMemoryFields)> {
    if nx < 1 || ny < 1 || nz < 1 {
        return Err(Error::Config(format!(
            "generate_cases: bad resolution {nx} x {ny} x {nz}"
        )));
    }

    let b = case_block_spec(kind, nx, ny, nz);
    let block = Block::new(&b)?;
    let mesh = build_host_mesh(&poly_mesh_raw(&block)?)?;

    if kind == CaseKind::DamBreak {
        let fields = build_dam_break_fields(&block, None)?;
        return Ok((mesh, fields));
    }

    let (nu, u_ref, half_height) = case_run_params(kind, &b, &block);
    let fields =
        build_initial_fields(kind, &block, None, nu, u_ref, half_height, 1, WallTreatment::Standard, None, false)?;
    Ok((mesh, fields))
}

/// [`write_case`], with the block castellated against `surface` first
/// (SPEC-LIT §23.4): solid cells removed, new wall patches for the
/// fluid-solid faces, and the new patches carrying exactly the wall boundary
/// conditions the uncarved case writer gives its walls - same code path, so
/// a carved case runs unmodified.
pub fn write_carved_case(
    case_dir: &Path,
    kind: CaseKind,
    nx: usize,
    ny: usize,
    nz: usize,
    surface: &Surface,
) -> Result<CarveSummary> {
    match write_case_impl(
        case_dir, kind, nx, ny, nz, Some(surface), WallTreatment::Standard, None, false, &[],
    )? {
        Some(s) => Ok(s),
        // Unreachable: the impl returns a summary whenever a surface went in.
        None => Err(Error::Mesh("carve produced no summary".to_string())),
    }
}

/// [`write_carved_case`], with the `wallTreatment` preset (route c) expanded
/// into the carved case's `0/` fields - the newly carved wall patches follow
/// the same preset as the block's own walls, since both go through
/// [`build_initial_fields`]'s one `PatchKind::Wall` branch.
pub fn write_carved_case_with_wall_model(
    case_dir: &Path,
    kind: CaseKind,
    nx: usize,
    ny: usize,
    nz: usize,
    surface: &Surface,
    wall: WallTreatment,
    roughness: Option<Roughness>,
) -> Result<CarveSummary> {
    match write_case_impl(case_dir, kind, nx, ny, nz, Some(surface), wall, roughness, true, &[])? {
        Some(s) => Ok(s),
        None => Err(Error::Mesh("carve produced no summary".to_string())),
    }
}

#[allow(clippy::too_many_arguments)]
fn write_case_impl(
    case_dir: &Path,
    kind: CaseKind,
    nx: usize,
    ny: usize,
    nz: usize,
    surface: Option<&Surface>,
    wall: WallTreatment,
    roughness: Option<Roughness>,
    thermal_wall: bool,
    cyclic: &[usize],
) -> Result<Option<CarveSummary>> {
    if nx < 1 || ny < 1 || nz < 1 {
        return Err(Error::Config(format!(
            "generate_cases: bad resolution {nx} x {ny} x {nz}"
        )));
    }

    let mut b = case_block_spec(kind, nx, ny, nz);
    // SPEC-LIT §31.1/§34.2: only reachable through `write_case_cyclic`, which
    // never carries a `surface` - carving and cyclic pairing together is a
    // combination nothing has asked for yet, so it stays unreachable through
    // the public API rather than half-supported here. `set_cyclic_axis`
    // itself refuses a repeated axis, so passing the same axis twice here is
    // an error rather than a silent no-op.
    for &axis in cyclic {
        b.set_cyclic_axis(axis)?;
    }
    let block = Block::new(&b)?;

    let carved = match surface {
        Some(s) => Some(carve_block(&block, s)?),
        None => None,
    };
    let summary = carved.as_ref().map(|cv| cv.summary(&block));
    let carve = carved.as_ref();
    let n_cells_out = carve.map_or(block.n_cells, |cv| cv.fluid_old.len());

    match carve {
        Some(cv) => write_carved_poly_mesh(case_dir, &block, cv)?,
        None => write_poly_mesh(case_dir, &block)?,
    }

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
        write_dam_break_fields(case_dir, &block, carve)?;

        println!(
            "damBreak: {} x {} x {} = {} cells -> {}",
            nx,
            ny,
            nz,
            n_cells_out,
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
        return Ok(summary);
    }

    // Air for the plume; every other case keeps the 1e-5 it has always had.
    let (nu, u_ref, half_height) = case_run_params(kind, &b, &block);
    let inlet = b.windows.first().map(|w| window_extent(&block.g, w));

    // Only the plume carries a temperature equation, so only its dictionary
    // gets the two Prandtl numbers that equation reads. Writing them into the
    // other cases would change files nothing there looks at.
    let extra = if buoyant_case(kind) {
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
    if buoyant_case(kind) {
        write_gravity(case_dir)?;
    }

    // Wall-normal direction and half height per case, so the initial profile is
    // oriented the way the geometry expects. `u_ref`/`half_height` came from
    // `case_run_params` above, together with `nu`.
    write_initial_fields(
        case_dir, kind, &block, carve, nu, u_ref, half_height, 1, wall, roughness, thermal_wall,
    )?;

    if let (Some(w), Some((fx, fy))) = (b.windows.first(), inlet) {
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
        n_cells_out,
        case_dir.display()
    );

    Ok(summary)
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

/// One boundary-field entry to write: a block patch, or - when the mesh was
/// carved - one of the new wall patches of §23.4.
///
/// The point of routing BOTH through the same list is the §23.4 *DESIGN*
/// note: the new patches are `wall` type and take exactly the wall boundary
/// conditions the writers below already choose for walls, by falling into
/// the same `PatchKind::Wall` branches - one code path, not a second one.
struct FieldPatch<'a> {
    name: &'a str,
    type_name: &'a str,
    /// Index into `Block::patches`; `None` for a carved wall patch.
    bidx: Option<usize>,
}

/// The boundary-field entries of this case, in boundary-file order: the
/// block's patches, then any carved wall patches.
fn field_patches<'a>(block: &'a Block, carve: Option<&'a Carved>) -> Vec<FieldPatch<'a>> {
    let mut v: Vec<FieldPatch<'a>> = block
        .patches
        .iter()
        .enumerate()
        .map(|(i, p)| FieldPatch { name: &p.name, type_name: &p.type_name, bidx: Some(i) })
        .collect();
    if let Some(cv) = carve {
        for (name, _) in &cv.walls {
            v.push(FieldPatch { name, type_name: "wall", bidx: None });
        }
    }
    v
}

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

/// SPEC-LIT §29.1's table: the `wallTreatment` row's type name for one
/// turbulence-closure field. `field` is one of `"k"`/`"epsilon"`/`"omega"`/
/// `"nut"`; anything else falls back to `zeroGradient`, which nothing here
/// calls this with.
fn wall_row_type(field: &str, wall: WallTreatment) -> &'static str {
    match field {
        "nut" => wall.nut_type(),
        "k" => wall.k_type(),
        "epsilon" => wall.epsilon_type(),
        "omega" => wall.omega_type(),
        _ => "zeroGradient",
    }
}

/// `Ks`/`Cs` onto a `nut` wall-function entry under the `rough` preset -
/// SPEC-LIT §29.1/§29.2. A no-op for every other field or treatment.
fn apply_roughness(s: &mut PatchFieldSpec, field: &str, wall: WallTreatment, roughness: Option<Roughness>) {
    if field == "nut" && wall == WallTreatment::Rough {
        if let Some(r) = roughness {
            s.extra.insert("Ks".to_string(), r.ks.to_string());
            s.extra.insert("Cs".to_string(), r.cs.to_string());
        }
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
fn write_dam_break_fields(case_dir: &Path, block: &Block, carve: Option<&Carved>) -> Result<()> {
    write_fields(case_dir, &build_dam_break_fields(block, carve)?)
}

fn build_dam_break_fields(block: &Block, carve: Option<&Carved>) -> Result<InMemoryFields> {
    let g = &block.g;
    let n_cells = carve.map_or(g.n_cells(), |cv| cv.fluid_old.len());
    // Internal values are per FLUID cell on a carved mesh; `old_of` maps the
    // written cell id back to the block cell whose centre the profile reads.
    let old_of = |c: usize| carve.map_or(c, |cv| cv.fluid_old[c]);
    let fps = field_patches(block, carve);

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
            let p = g.cell_centre(old_of(c));
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

    for fp in &fps {
        let pk = PatchKind::from_type(fp.type_name);
        let s = if pk == PatchKind::Empty {
            patch_spec("empty")
        } else if pk == PatchKind::Cyclic {
            // SPEC-LIT §31.1: every field carries `cyclic` on a cyclic
            // patch - `kinds_from_patches` (field.rs) gives the mesh the
            // last word at runtime, but the written file has to agree
            // with `constant/polyMesh/boundary` for a round trip.
            patch_spec("cyclic")
        } else if fp.name == "atmosphere" {
            let mut s = patch_spec("inletOutlet");
            s.inlet_value = vec![0.0];
            s.value = vec![0.0];
            s
        } else {
            patch_spec("zeroGradient")
        };
        alpha.boundary.insert(fp.name.to_string(), s);
    }

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

    for fp in &fps {
        let pk = PatchKind::from_type(fp.type_name);
        let s = if pk == PatchKind::Empty {
            patch_spec("empty")
        } else if pk == PatchKind::Cyclic {
            // SPEC-LIT §31.1: every field carries `cyclic` on a cyclic
            // patch - `kinds_from_patches` (field.rs) gives the mesh the
            // last word at runtime, but the written file has to agree
            // with `constant/polyMesh/boundary` for a round trip.
            patch_spec("cyclic")
        } else if fp.name == "atmosphere" {
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
        u.boundary.insert(fp.name.to_string(), s);
    }

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

    for fp in &fps {
        let pk = PatchKind::from_type(fp.type_name);
        let s = if pk == PatchKind::Empty {
            patch_spec("empty")
        } else if pk == PatchKind::Cyclic {
            // SPEC-LIT §31.1: every field carries `cyclic` on a cyclic
            // patch - `kinds_from_patches` (field.rs) gives the mesh the
            // last word at runtime, but the written file has to agree
            // with `constant/polyMesh/boundary` for a round trip.
            patch_spec("cyclic")
        } else if fp.name == "atmosphere" {
            let mut s = patch_spec("fixedValue");
            s.value = vec![0.0];
            s
        } else {
            patch_spec("zeroGradient")
        };
        p.boundary.insert(fp.name.to_string(), s);
    }

    Ok(InMemoryFields { scalars: vec![alpha, p], vectors: vec![u] })
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
    carve: Option<&Carved>,
    nu: Scalar,
    u_ref: Scalar,
    half_height: Scalar,
    wall_normal: usize,
    wall: WallTreatment,
    roughness: Option<Roughness>,
    thermal_wall: bool,
) -> Result<()> {
    let fields = build_initial_fields(
        kind, block, carve, nu, u_ref, half_height, wall_normal, wall, roughness, thermal_wall,
    )?;
    write_fields(case_dir, &fields)
}

#[allow(clippy::too_many_arguments)]
fn build_initial_fields(
    kind: CaseKind,
    block: &Block,
    carve: Option<&Carved>,
    nu: Scalar,
    u_ref: Scalar,
    half_height: Scalar,
    wall_normal: usize,
    wall: WallTreatment,
    roughness: Option<Roughness>,
    thermal_wall: bool,
) -> Result<InMemoryFields> {
    let g = &block.g;
    let n_cells = carve.map_or(g.n_cells(), |cv| cv.fluid_old.len());
    // On a carved mesh the internal list runs over the FLUID cells only;
    // `old_of` maps back to the block cell whose centre the profiles read.
    let old_of = |c: usize| carve.map_or(c, |cv| cv.fluid_old[c]);
    let fps = field_patches(block, carve);

    // Turbulence intensity 5 %, mixing length 7 % of the half height: the
    // standard OpenFOAM inlet estimate.
    let i_turb = 0.05 as Scalar;
    let cmu = 0.09 as Scalar;
    let k0 = 1.5 * (i_turb * u_ref) * (i_turb * u_ref);
    let l = 0.07 * half_height * 2.0;
    let eps0 = cmu.powf(0.75) * k0.powf(1.5) / l;
    let omega0 = k0.sqrt() / (cmu.powf(0.25) * l);

    let cavity = kind == CaseKind::Cavity;
    let plume = buoyant_case(kind);
    let t_inlet = if kind == CaseKind::Room { ROOM_T_INLET } else { PLUME_T_INLET };

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
        let centre = g.cell_centre(old_of(c));

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

    for fp in &fps {
        let name = fp.name;
        let pk = PatchKind::from_type(fp.type_name);

        let s = if pk == PatchKind::Empty {
            patch_spec("empty")
        } else if pk == PatchKind::Cyclic {
            // SPEC-LIT §31.1: every field carries `cyclic` on a cyclic
            // patch - `kinds_from_patches` (field.rs) gives the mesh the
            // last word at runtime, but the written file has to agree
            // with `constant/polyMesh/boundary` for a round trip.
            patch_spec("cyclic")
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
            } else if let Some(bi) = fp.bidx {
                // The inlet carries the same profile as the interior, so the
                // solution does not have to develop one from a top hat. On a
                // carved mesh only the surviving (fluid-owned) faces exist,
                // in the same relative order, and the owner index must go
                // through the fluid renumbering.
                let patch = &block.patches[bi];
                let (na, _) = slot_dims(g, patch.slot);
                match carve {
                    None => (0..patch.size)
                        .map(|idx| {
                            let q =
                                boundary_quad(g, patch.slot, patch.slot_index(na, idx));
                            u.internal[q.own]
                        })
                        .collect(),
                    Some(cv) => cv.domain[bi]
                        .iter()
                        .map(|&sl| {
                            let q = boundary_quad(g, patch.slot, sl);
                            u.internal[cv.new_of_old[q.own] as usize]
                        })
                        .collect(),
                }
            } else {
                // Unreachable: a carved wall patch named "inlet" would have
                // collided with the block's inlet in `Carved::build`.
                vec![Vec3::ZERO]
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

    let vectors = vec![u];
    let mut scalars: Vec<RawScalarField> = Vec::new();

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

        for fp in &fps {
            let pk = PatchKind::from_type(fp.type_name);

            let s = if pk == PatchKind::Empty {
                patch_spec("empty")
            } else if pk == PatchKind::Cyclic {
                patch_spec("cyclic")
            } else if fp.name == "outlet" {
                let mut s = patch_spec("fixedValue");
                s.value = vec![0.0];
                s
            } else {
                // Walls and the burner alike. A wall imposes no pressure, and
                // neither does an inlet whose velocity is prescribed: fixing
                // both `U` and `p` on the same face over-specifies the face.
                patch_spec("zeroGradient")
            };

            pf.boundary.insert(fp.name.to_string(), s);
        }

        scalars.push(pf);
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

        for fp in &fps {
            let pk = PatchKind::from_type(fp.type_name);

            let s = if pk == PatchKind::Empty {
                patch_spec("empty")
            } else if pk == PatchKind::Cyclic {
                patch_spec("cyclic")
            } else if fp.name == "inlet" {
                let mut s = patch_spec("fixedValue");
                s.value = vec![t_inlet];
                s
            } else if pk == PatchKind::Wall {
                // Adiabatic floor and ceiling by default: the case is about
                // the plume, not about how much heat the room absorbs.
                // SPEC-LIT §29.3 overrides that only when a wall model was
                // actually asked for (`thermal_wall`, `-wallModel`), except
                // `lowRe`, which pins the same molecular resistance this
                // default already gives.
                match thermal_wall.then(|| wall.thermal_type()).flatten() {
                    Some(t) => patch_spec(t),
                    None => patch_spec("zeroGradient"),
                }
            } else {
                let mut s = patch_spec("inletOutlet");
                s.inlet_value = vec![PLUME_T_AMBIENT];
                s.value = vec![PLUME_T_AMBIENT];
                s
            };

            t.boundary.insert(fp.name.to_string(), s);
        }

        scalars.push(t);
    }

    // ---- scalars ---------------------------------------------------------
    let specs: [(&str, &str, Scalar); 4] = [
        ("k", "[0 2 -2 0 0 0 0]", k0),
        ("epsilon", "[0 2 -3 0 0 0 0]", eps0),
        ("omega", "[0 0 -1 0 0 0 0]", omega0),
        ("nut", "[0 2 -1 0 0 0 0]", 0.0),
    ];

    for (name, dims, value) in specs {
        let mut f = RawScalarField {
            name: name.to_string(),
            dimensions: dims.to_string(),
            internal: vec![value; n_cells],
            boundary: BTreeMap::new(),
            boundary_patterns: Vec::new(),
        };

        for fp in &fps {
            let pname = fp.name;
            let pk = PatchKind::from_type(fp.type_name);

            let s = if pk == PatchKind::Empty {
                patch_spec("empty")
            } else if pk == PatchKind::Cyclic {
                patch_spec("cyclic")
            } else if pk == PatchKind::Wall {
                // SPEC-LIT §29.1: the wallTreatment row's entry for this
                // field, `standard` by default - `wall_row_type` reproduces
                // today's hardcoded strings exactly when `wall` is
                // `WallTreatment::Standard`.
                let type_name = wall_row_type(name, wall);
                let mut s = patch_spec(type_name);
                // SPEC-LIT §33.2: see the twin branch above (`build_plume...`)
                // for why `lowRe`'s `epsilon` (the only "fixedValue" this
                // table produces) gets a literal 0 rather than the domain's
                // equilibrium value.
                s.value = vec![if type_name == "fixedValue" { 0.0 } else { value }];
                apply_roughness(&mut s, name, wall, roughness);
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

        scalars.push(f);
    }

    println!(
        "  0/ fields: Uref {}  k {}  epsilon {}  omega {}  (nu {})",
        fmt_g(u_ref),
        fmt_g(k0),
        fmt_g(eps0),
        fmt_g(omega0),
        fmt_g(nu)
    );

    Ok(InMemoryFields { scalars, vectors })
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
        b.windows = vec![PatchWindow {
            slot: 4,
            lo: [1, 1],
            hi: [3, 3],
            name: "burner".to_string(),
            type_name: "patch".to_string(),
        }];
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
        let w = b.windows.first().expect("window");

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

    // ----------------------------------------------------------------------
    //  SPEC-LIT §42.8 Gate 2: more than one window
    // ----------------------------------------------------------------------

    /// Two windows on DIFFERENT slots both get carved, and each slot's
    /// remainder keeps its own name. This is what a compartment fire needs -
    /// a burner in the floor and a doorway in a wall - and what `BlockSpec`
    /// could not express until §42.8's gate needed it.
    #[test]
    fn two_windows_on_different_slots_are_both_carved() {
        let mut b = split_spec(6, 6, 4);
        // `split_spec` already put `burner` in the floor (slot 4); add a
        // doorway in the -y wall (slot 2).
        b.patch_name[2] = "wallFront".to_string();
        b.patch_type[2] = "wall".to_string();
        b.windows.push(PatchWindow {
            slot: 2,
            lo: [1, 0],
            hi: [4, 2],
            name: "door".to_string(),
            type_name: "patch".to_string(),
        });
        let block = Block::new(&b).expect("two windows on two slots");

        let by_name = |n: &str| block.patches.iter().find(|p| p.name == n).expect(n);
        assert_eq!(by_name("burner").size, 4, "burner is 2x2 floor faces");
        assert_eq!(by_name("door").size, 6, "door is 3x2 wall faces");
        assert_eq!(by_name("burner").part, SlotPart::Window);
        assert_eq!(by_name("door").part, SlotPart::Window);
        assert_eq!(by_name("floor").part, SlotPart::Rest);
        assert_eq!(by_name("wallFront").part, SlotPart::Rest);

        // Eight patches now - six slots, two of them split - and every one of
        // them is still an unbroken run, which is the invariant a boundary
        // file cannot express if it is broken.
        assert_eq!(block.patches.len(), 8);
        let mut runs: Vec<(usize, usize)> =
            block.patches.iter().map(|p| (p.start, p.size)).collect();
        runs.sort_unstable();
        // `start` is a GLOBAL face index, so the run begins at the first
        // boundary face, not at zero.
        let mut next = runs[0].0;
        let first = next;
        for (start, size) in runs {
            assert_eq!(start, next, "patches must tile the boundary face list");
            next += size;
        }
        assert_eq!(next - first, block.patch_size.iter().sum::<usize>());
    }

    /// TWO windows on the SAME slot is refused by name. A slot split three
    /// ways cannot be laid out as contiguous `startFace`/`nFaces` runs, and a
    /// boundary file that claimed otherwise would be read back as a different
    /// mesh with no error anywhere.
    #[test]
    fn two_windows_on_the_same_slot_are_refused_by_name() {
        let mut b = split_spec(6, 6, 4);
        b.windows.push(PatchWindow {
            slot: 4,
            lo: [4, 4],
            hi: [5, 5],
            name: "secondBurner".to_string(),
            type_name: "patch".to_string(),
        });
        let Err(e) = Block::new(&b) else {
            panic!("two windows on one slot must be refused");
        };
        let msg = format!("{e}");
        assert!(msg.contains("burner"), "{msg}");
        assert!(msg.contains("secondBurner"), "{msg}");
        assert!(msg.contains("at most one window"), "{msg}");
    }

    /// A window whose fast direction spans the whole slot leaves the middle
    /// run of `Rest` empty - the one arithmetic case that would divide by zero
    /// if it were reached.
    #[test]
    fn a_full_width_window_leaves_no_punched_rows() {
        let mut b = spec(4, 5, 3);
        b.windows = vec![PatchWindow {
            slot: 4,
            lo: [0, 1],
            hi: [4, 3],
            name: "strip".to_string(),
            type_name: "patch".to_string(),
        }];

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
        b.windows = vec![PatchWindow {
            slot: 4,
            lo: [0, 0],
            hi: [4, 5],
            name: "all".to_string(),
            type_name: "patch".to_string(),
        }];
        assert!(Block::new(&b).is_err());

        // Out of range, empty, and colliding with the host patch's name.
        for (lo, hi, name) in [
            ([0, 0], [5, 2], "w"),
            ([2, 2], [2, 3], "w"),
            ([0, 0], [2, 2], "zMin"),
        ] {
            let mut b = spec(4, 5, 3);
            b.windows = vec![PatchWindow {
                slot: 4,
                lo,
                hi,
                name: name.to_string(),
                type_name: "patch".to_string(),
            }];
            assert!(Block::new(&b).is_err(), "accepted {lo:?}..{hi:?} '{name}'");
        }
    }

    // ---- the plume case --------------------------------------------------

    #[test]
    fn only_the_plume_splits_a_patch() {
        for k in [CaseKind::Channel, CaseKind::Cavity, CaseKind::Step, CaseKind::Big] {
            let (nx, ny, nz) = k.default_resolution();
            assert!(
                case_block_spec(k, nx, ny, nz).windows.is_empty(),
                "{} grew a window",
                k.as_str()
            );
        }

        assert_eq!(CaseKind::from_name("plume"), Some(CaseKind::Plume));
        assert_eq!(CaseKind::from_name("room"), Some(CaseKind::Room));
        assert_eq!(CaseKind::Room.as_str(), "room");
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
        let w = b.windows.first().expect("window");

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
        let w = b.windows.first().expect("window");

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
        let w = b.windows.first().expect("window");

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

    // ------------------------------------------------------------------
    //  SPEC-LIT §29.1: wallTreatment presets (route c, generate-mesh)
    // ------------------------------------------------------------------

    /// `write_case_with_wall_model(.., Standard, ..)` reproduces
    /// `write_case`'s hardcoded row exactly - "standard stays the default".
    #[test]
    fn standard_wall_model_matches_the_legacy_default_row() {
        let dir = temp_dir("wall_model_standard");
        write_case_with_wall_model(&dir, CaseKind::Plume, 20, 12, 6, WallTreatment::Standard, None)
            .expect("write");

        let zero = dir.join("0");
        assert_eq!(fs::read_to_string(zero.join("nut")).unwrap().matches("nutkWallFunction").count(), 5);
        assert_eq!(fs::read_to_string(zero.join("k")).unwrap().matches("kqRWallFunction").count(), 5);
        assert_eq!(
            fs::read_to_string(zero.join("epsilon")).unwrap().matches("epsilonWallFunction").count(),
            5
        );

        let _ = fs::remove_dir_all(&dir);
    }

    /// Each preset expands to exactly its row - SPEC-LIT §29.1's table,
    /// string-level, on the generated `0/` fields.
    #[test]
    fn each_wall_model_expands_to_exactly_its_row() {
        for (wt, nut_want, k_want, eps_want) in [
            (WallTreatment::Standard, "nutkWallFunction", "kqRWallFunction", "epsilonWallFunction"),
            (WallTreatment::Spalding, "nutUWallFunction", "kqRWallFunction", "epsilonWallFunction"),
            (WallTreatment::LowRe, "nutLowReWallFunction", "kLowReWallFunction", "fixedValue"),
        ] {
            let dir = temp_dir(&format!("wall_model_row_{}", wt.name()));
            write_case_with_wall_model(&dir, CaseKind::Plume, 20, 12, 6, wt, None).expect("write");

            let zero = dir.join("0");
            let nut = fs::read_to_string(zero.join("nut")).unwrap();
            let k = fs::read_to_string(zero.join("k")).unwrap();
            let eps = fs::read_to_string(zero.join("epsilon")).unwrap();
            let omega = fs::read_to_string(zero.join("omega")).unwrap();
            let omega_want = if wt == WallTreatment::LowRe { "zeroGradient" } else { "omegaWallFunction" };

            // Every one of the five wall patches, exactly - not a substring
            // count, which `zeroGradient` (the `lowRe` completion AND the
            // unrelated non-wall default `outlet` already carries) would
            // over-count.
            for wall_patch in ["wallXMin", "wallYMin", "wallYMax", "floor", "ceiling"] {
                assert!(
                    patch_entry(&nut, wall_patch).contains(nut_want),
                    "{wt:?} {wall_patch} nut: {}",
                    patch_entry(&nut, wall_patch)
                );
                assert!(
                    patch_entry(&k, wall_patch).contains(k_want),
                    "{wt:?} {wall_patch} k: {}",
                    patch_entry(&k, wall_patch)
                );
                assert!(
                    patch_entry(&eps, wall_patch).contains(eps_want),
                    "{wt:?} {wall_patch} epsilon: {}",
                    patch_entry(&eps, wall_patch)
                );
                // SPEC-LIT §33.2: `lowRe`'s `fixedValue` on `epsilon` must
                // hold the homogeneous Dirichlet value, not the domain's
                // equilibrium epsilon - the whole reason it is `fixedValue`
                // rather than `zeroGradient` in the first place.
                if wt == WallTreatment::LowRe {
                    assert!(
                        patch_entry(&eps, wall_patch).contains("uniform 0;"),
                        "{wt:?} {wall_patch} epsilon should hold 0, not the domain value: {}",
                        patch_entry(&eps, wall_patch)
                    );
                }
                assert!(
                    patch_entry(&omega, wall_patch).contains(omega_want),
                    "{wt:?} {wall_patch} omega: {}",
                    patch_entry(&omega, wall_patch)
                );
            }

            let _ = fs::remove_dir_all(&dir);
        }
    }

    /// `rough` writes `Ks`/`Cs` onto every wall face's `nut` entry.
    #[test]
    fn rough_wall_model_writes_ks_and_cs() {
        let dir = temp_dir("wall_model_rough");
        let rough = Roughness { ks: 0.002, cs: 0.6 };
        write_case_with_wall_model(&dir, CaseKind::Plume, 20, 12, 6, WallTreatment::Rough, Some(rough))
            .expect("write");

        let nut = fs::read_to_string(dir.join("0").join("nut")).expect("read nut");
        assert_eq!(nut.matches("nutkRoughWallFunction").count(), 5, "{nut}");
        assert_eq!(nut.matches("0.002").count(), 5, "Ks missing:\n{nut}");
        assert_eq!(nut.matches("0.6").count(), 5, "Cs missing:\n{nut}");

        let _ = fs::remove_dir_all(&dir);
    }

    /// §29.3, through `-wallModel`: every row except `lowRe` puts a
    /// `thermalWallFunction` on the walls of a case that solves `T`; `lowRe`
    /// keeps the adiabatic default.
    #[test]
    fn wall_model_applies_the_thermal_wall_function_except_low_re() {
        for (wt, want) in [
            (WallTreatment::Standard, "thermalWallFunction"),
            (WallTreatment::LowRe, "zeroGradient"),
        ] {
            let dir = temp_dir(&format!("wall_model_thermal_{}", wt.name()));
            write_case_with_wall_model(&dir, CaseKind::Plume, 20, 12, 6, wt, None).expect("write");

            let t = fs::read_to_string(dir.join("0").join("T")).expect("read T");
            assert_eq!(t.matches(want).count(), 5, "{wt:?}:\n{t}");

            let _ = fs::remove_dir_all(&dir);
        }
    }

    /// Without `-wallModel` (plain [`write_case`]), `T` stays exactly the
    /// legacy adiabatic default - route (c) changes nothing unless asked.
    #[test]
    fn no_wall_model_leaves_t_adiabatic() {
        let dir = temp_dir("no_wall_model_t");
        write_case(&dir, CaseKind::Plume, 20, 12, 6).expect("write");
        let t = fs::read_to_string(dir.join("0").join("T")).expect("read T");
        assert_eq!(t.matches("zeroGradient").count(), 5);
        assert!(!t.contains("thermalWallFunction"));
        let _ = fs::remove_dir_all(&dir);
    }

    /// Carved STL wall patches follow the same preset as the block's own
    /// walls - route (c)'s explicit requirement. A small cuboid obstruction
    /// well inside the plume domain, away from the floor's burner window and
    /// the +x outlet, castellates into new `wall`-kind patches; those must
    /// carry the SAME `spalding` row as `wallXMin`/`wallYMin`/etc., not the
    /// `standard` default.
    #[test]
    fn carved_walls_follow_the_same_wall_model() {
        let dir = temp_dir("wall_model_carved");
        let plug = cuboid_surface(Vec3::new(2.0, 0.0, 1.0), Vec3::new(3.0, 1.0, 2.0), "plug");

        let summary =
            write_carved_case_with_wall_model(&dir, CaseKind::Plume, 20, 12, 6, &plug, WallTreatment::Spalding, None)
                .expect("carve");
        assert!(!summary.wall_faces.is_empty(), "the plug should carve out new wall faces");

        let nut = fs::read_to_string(dir.join("0").join("nut")).expect("read nut");
        for (patch, _) in &summary.wall_faces {
            let entry = patch_entry(&nut, patch);
            assert!(entry.contains("nutUWallFunction"), "{patch}: {entry}");
        }
        // The block's own original walls carry the same row.
        assert!(patch_entry(&nut, "wallXMin").contains("nutUWallFunction"));

        let _ = fs::remove_dir_all(&dir);
    }

    /// The text of one `boundaryField { <patch> { ... } }` entry, for
    /// per-patch assertions on a field written with one entry per patch
    /// (never a `".*"` pattern - `write_fields`/`build_initial_fields` always
    /// expand explicitly).
    fn patch_entry<'a>(src: &'a str, patch: &str) -> &'a str {
        let at = src.find(&format!("    {patch}\n")).unwrap_or_else(|| {
            panic!("no `{patch}` entry:\n{src}")
        });
        let open = src[at..].find('{').map(|i| at + i).unwrap();
        let close = src[open..].find('}').map(|i| open + i).unwrap();
        &src[open..close]
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

    /// §B1 gate: `write_case`'s `0/` fields must be byte-identical before and
    /// after the field writers were split into `build_*` + `write_fields`.
    /// Regenerated straight from `build_initial_fields`/`build_dam_break_fields`
    /// and compared against the same case written whole through `write_case`.
    #[test]
    fn write_case_and_build_case_agree_on_every_0_field_byte_for_byte() {
        for kind in [
            CaseKind::Channel,
            CaseKind::Step,
            CaseKind::Cavity,
            CaseKind::Big,
            CaseKind::Plume,
            CaseKind::DamBreak,
        ] {
            let dir = temp_dir(&format!("split_{}", kind.as_str()));
            write_case(&dir, kind, 6, 5, if kind.as_str() == "cavity" { 1 } else { 4 })
                .unwrap_or_else(|e| panic!("write_case({kind:?}) failed: {e}"));

            let (_mesh, fields) = build_case(kind, 6, 5, if kind.as_str() == "cavity" { 1 } else { 4 })
                .unwrap_or_else(|e| panic!("build_case({kind:?}) failed: {e}"));

            for s in &fields.scalars {
                let path = dir.join("0").join(format!("{}.rebuilt", s.name));
                write_scalar_field(&path, s, "0").expect("write rebuilt scalar");
                assert_eq!(
                    fs::read(dir.join("0").join(&s.name)).expect("read on-disk bytes"),
                    fs::read(&path).expect("read rebuilt"),
                    "{kind:?}: field {} differs between write_case and build_case",
                    s.name
                );
            }
            for v in &fields.vectors {
                let path = dir.join("0").join(format!("{}.rebuilt", v.name));
                write_vector_field(&path, v, "0").expect("write rebuilt vector");
                assert_eq!(
                    fs::read(dir.join("0").join(&v.name)).expect("read on-disk bytes"),
                    fs::read(&path).expect("read rebuilt"),
                    "{kind:?}: field {} differs between write_case and build_case",
                    v.name
                );
            }

            let _ = fs::remove_dir_all(&dir);
        }
    }

    /// §B1 gate: `build_mesh` must produce the same `HostMesh` as the file
    /// round trip (`write_block_mesh` -> `read_poly_mesh` -> `build_host_mesh`)
    /// it replaces in `validate.rs` and `bench.rs`.
    #[test]
    fn build_mesh_matches_the_file_round_trip() {
        let b = BlockSpec {
            x: axis(0.0, 1.0, 6, 1.0, false),
            y: axis(0.0, 0.7, 5, 1.3, true),
            z: axis(0.0, 0.4, 4, 1.0, false),
            ..BlockSpec::default()
        };

        let direct = build_mesh(&b).expect("build_mesh");

        let dir = temp_dir("direct_vs_disk");
        write_block_mesh(&dir, &b).expect("write_block_mesh");
        let via_disk = crate::io::polymesh::build_host_mesh(
            &crate::io::polymesh::read_poly_mesh(&dir).expect("read_poly_mesh"),
        )
        .expect("build_host_mesh");
        let _ = fs::remove_dir_all(&dir);

        assert_eq!(direct.n_cells, via_disk.n_cells);
        assert_eq!(direct.n_internal_faces, via_disk.n_internal_faces);
        assert_eq!(direct.n_boundary_faces, via_disk.n_boundary_faces);
        assert_eq!(direct.owner, via_disk.owner);
        assert_eq!(direct.neighbour, via_disk.neighbour);
        for (a, bb) in direct.v.iter().zip(via_disk.v.iter()) {
            assert!((a - bb).abs() < 1e3 * EPS, "{a} vs {bb}");
        }
        for (a, bb) in direct.sf.iter().zip(via_disk.sf.iter()) {
            assert!((*a - *bb).mag() < 1e3 * EPS, "{a} vs {bb}");
        }
        assert_eq!(direct.patches.len(), via_disk.patches.len());
        for (a, bb) in direct.patches.iter().zip(via_disk.patches.iter()) {
            assert_eq!(a.name, bb.name);
            assert_eq!(a.start, bb.start);
            assert_eq!(a.size, bb.size);
        }
    }

    /// SPEC-LIT §31.1, point 1: the in-memory `build_mesh` path (which never
    /// touches a boundary file) has to produce the SAME `HostMesh` a periodic
    /// case's file round trip does - hashed, not spot-checked, so nothing
    /// cyclic-specific (`b_nbr_cell`, `b_kind`, `nbr_patch`) can drift between
    /// the two without failing.
    fn hash_mesh(m: &HostMesh) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        m.n_cells.hash(&mut h);
        m.n_internal_faces.hash(&mut h);
        m.n_boundary_faces.hash(&mut h);
        m.owner.hash(&mut h);
        m.neighbour.hash(&mut h);
        for x in m.v.iter().chain(m.mag_sf.iter()).chain(m.b_mag_sf.iter()) {
            x.to_bits().hash(&mut h);
        }
        for x in m.c.iter().chain(m.sf.iter()).chain(m.b_sf.iter()).chain(m.b_cf.iter()) {
            x.x.to_bits().hash(&mut h);
            x.y.to_bits().hash(&mut h);
            x.z.to_bits().hash(&mut h);
        }
        m.b_nbr_cell.hash(&mut h);
        m.b_kind.hash(&mut h);
        m.b_patch.hash(&mut h);
        for p in &m.patches {
            p.name.hash(&mut h);
            p.type_name.hash(&mut h);
            p.start.hash(&mut h);
            p.size.hash(&mut h);
            p.nbr_patch.hash(&mut h);
        }
        h.finish()
    }

    #[test]
    fn build_mesh_matches_the_file_round_trip_for_a_cyclic_pair() {
        let mut b = spec(6, 5, 4);
        b.set_cyclic_axis(0).expect("axis 0 is x");

        let direct = build_mesh(&b).expect("build_mesh");

        let dir = temp_dir("cyclic_direct_vs_disk");
        write_block_mesh(&dir, &b).expect("write_block_mesh");
        let via_disk = crate::io::polymesh::build_host_mesh(
            &crate::io::polymesh::read_poly_mesh(&dir).expect("read_poly_mesh"),
        )
        .expect("build_host_mesh");
        let _ = fs::remove_dir_all(&dir);

        assert_eq!(hash_mesh(&direct), hash_mesh(&via_disk));

        // The pair itself: some boundary face is actually coupled, and the
        // geometry closes to round-off same as any other mesh (SPEC-LIT §10).
        assert!(direct.b_nbr_cell.iter().any(|&c| c >= 0));
        let rep = crate::mesh::geometry::check(&direct);
        assert!(rep.max_closure_error < 1e-10, "closure {}", rep.max_closure_error);
    }

    /// SPEC-LIT §31.1's flux test: for a uniform convecting velocity, the
    /// two patches of a cyclic pair carry equal and opposite total flux -
    /// `phi_b = U . Sf_b` for a uniform `U` regardless of boundary kind, so
    /// this is really `sum(Sf_a) == -sum(Sf_b)` (already true face-by-face,
    /// per [`crate::io::polymesh::check_cyclic_invariants`]) read back as the
    /// physical quantity a solver actually cares about.
    #[test]
    fn a_cyclic_pair_carries_equal_and_opposite_total_flux() {
        let mut b = spec(6, 5, 3);
        b.set_cyclic_axis(0).expect("axis 0 is x");
        let hm = build_mesh(&b).expect("build_mesh");

        let u = Vec3::new(1.3, -0.4, 0.2);
        let cyclic = hm
            .patches
            .iter()
            .enumerate()
            .filter(|(_, p)| p.kind == PatchKind::Cyclic)
            .map(|(i, _)| i)
            .collect::<Vec<_>>();
        assert_eq!(cyclic.len(), 2, "exactly one cyclic pair on this block");

        let flux_of = |p: usize| -> Scalar {
            let patch = &hm.patches[p];
            (0..patch.size)
                .map(|k| u.dot(hm.b_sf[patch.start + k]))
                .sum()
        };
        let (flux_a, flux_b) = (flux_of(cyclic[0]), flux_of(cyclic[1]));

        assert!(flux_a.abs() > 1e-6, "flux_a is suspiciously near zero: {flux_a}");
        assert!(
            (flux_a + flux_b).abs() < 1e3 * EPS * flux_a.abs().max(1.0),
            "flux_a {flux_a} and flux_b {flux_b} are not equal and opposite"
        );
    }

    /// SPEC-LIT §34.2, the 34.3 table's "two cyclic pairs close": a plane
    /// channel periodic in x AND y (z stays the block's default `wall`).
    /// Every one of the two pairs' four patches must come back
    /// `PatchKind::Cyclic`, and each pair must independently carry equal and
    /// opposite total flux for a uniform field - exactly
    /// `a_cyclic_pair_carries_equal_and_opposite_total_flux` above, run
    /// twice, on the SAME mesh, to show the two pairs do not interfere with
    /// each other.
    #[test]
    fn two_cyclic_pairs_each_carry_equal_and_opposite_total_flux() {
        let mut b = spec(6, 5, 3);
        b.set_cyclic_axis(0).expect("axis 0 is x");
        b.set_cyclic_axis(1).expect("axis 1 is y");
        let hm = build_mesh(&b).expect("build_mesh");

        assert_eq!(
            hm.patches.iter().filter(|p| p.kind == PatchKind::Cyclic).count(),
            4,
            "both pairs' four patches must all be PatchKind::Cyclic"
        );
        // z was never touched - `spec`'s default patch_type keeps it `empty`.
        assert!(hm.patches.iter().any(|p| p.kind == PatchKind::Empty));

        let u = Vec3::new(1.3, -0.4, 0.2);
        let flux_of = |p: &crate::mesh::PatchInfo| -> Scalar {
            (0..p.size).map(|k| u.dot(hm.b_sf[p.start + k])).sum()
        };

        for axis_names in [["xMin", "xMax"], ["yMin", "yMax"]] {
            let get = |name: &str| hm.patches.iter().find(|p| p.name == name).expect(name);
            let (fa, fb) = (flux_of(get(axis_names[0])), flux_of(get(axis_names[1])));
            assert!(fa.abs() > 1e-6, "{axis_names:?}: flux_a suspiciously near zero: {fa}");
            assert!(
                (fa + fb).abs() < 1e3 * EPS * fa.abs().max(1.0),
                "{axis_names:?}: flux_a {fa} and flux_b {fb} are not equal and opposite"
            );
        }
    }

    /// SPEC-LIT §34.2, the 34.3 table's "three pairs (a periodic box) close
    /// with zero net flux through every pair": every one of the six patches
    /// is cyclic, and each of the three axis pairs independently balances a
    /// uniform field's flux to round-off.
    #[test]
    fn three_cyclic_pairs_close_a_periodic_box_with_zero_net_flux_per_pair() {
        let mut b = spec(6, 5, 4);
        b.set_cyclic_axis(0).expect("axis 0 is x");
        b.set_cyclic_axis(1).expect("axis 1 is y");
        b.set_cyclic_axis(2).expect("axis 2 is z");
        let hm = build_mesh(&b).expect("build_mesh");

        assert_eq!(hm.patches.len(), 6);
        assert!(hm.patches.iter().all(|p| p.kind == PatchKind::Cyclic));

        let u = Vec3::new(0.7, 1.1, -0.9);
        let flux_of = |p: &crate::mesh::PatchInfo| -> Scalar {
            (0..p.size).map(|k| u.dot(hm.b_sf[p.start + k])).sum()
        };

        for axis_names in [["xMin", "xMax"], ["yMin", "yMax"], ["zMin", "zMax"]] {
            let get = |name: &str| hm.patches.iter().find(|p| p.name == name).expect(name);
            let (fa, fb) = (flux_of(get(axis_names[0])), flux_of(get(axis_names[1])));
            assert!(fa.abs() > 1e-6, "{axis_names:?}: flux_a suspiciously near zero: {fa}");
            assert!(
                (fa + fb).abs() < 1e3 * EPS * fa.abs().max(1.0),
                "{axis_names:?}: flux_a {fa} and flux_b {fb} are not equal and opposite"
            );
        }

        // The whole box, all three pairs at once: a uniform field's net
        // flux over the ENTIRE boundary must also vanish to round-off - the
        // mesh-closure claim (SPEC-LIT §10) specialised to a fully periodic
        // domain, where the boundary is nothing but cyclic pairs.
        let total: Scalar = hm.patches.iter().map(flux_of).sum();
        assert!(
            total.abs() < 1e3 * EPS * u.mag() * hm.n_boundary_faces as Scalar,
            "total boundary flux over a periodic box must vanish: {total}"
        );
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

    // ---- castellated carving (SPEC-LIT §23.4-23.5) -------------------------

    /// An axis-aligned closed cuboid as a 12-triangle outward-wound surface.
    fn cuboid_surface(lo: Vec3, hi: Vec3, name: &str) -> Surface {
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
            [0, 3, 2], [0, 2, 1], // z lo
            [4, 5, 6], [4, 6, 7], // z hi
            [0, 4, 7], [0, 7, 3], // x lo
            [1, 2, 6], [1, 6, 5], // x hi
            [0, 1, 5], [0, 5, 4], // y lo
            [3, 7, 6], [3, 6, 2], // y hi
        ];
        let soup: Vec<crate::surface::SoupTri> =
            T.iter().map(|&[a, b, c]| (0u32, [p[a], p[b], p[c]])).collect();
        Surface::from_soup(soup, vec![name.to_string()]).expect("cuboid surface")
    }

    /// A closed UV sphere: `n_theta` bands, `n_phi` slices, pole fans, every
    /// shared vertex computed once so the bit-exact weld closes it.
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
                                r * th.sin() * ph.cos(),
                                r * th.sin() * ph.sin(),
                                r * th.cos(),
                            )
                    })
                    .collect()
            })
            .collect();
        let north = centre + Vec3::new(0.0, 0.0, r);
        let south = centre - Vec3::new(0.0, 0.0, r);

        let mut soup: Vec<crate::surface::SoupTri> = Vec::new();
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
        Surface::from_soup(soup, vec!["sphere".to_string()]).expect("sphere surface")
    }

    /// §23.5 row 1: a grid-aligned cuboid carved from the 20^3 unit box must
    /// leave EXACTLY the analytic fluid cell count and the analytic new-wall
    /// face count - and the whole classification must come from parity
    /// alone, with zero vote and zero arbitration cost.
    #[test]
    fn carved_cuboid_matches_the_analytic_cell_and_face_counts() {
        use crate::io::polymesh::{build_host_mesh, read_poly_mesh};

        let dir = temp_dir("carve_cuboid");
        // [0.25, 0.75]^3 in the `big` unit box at h = 0.05: faces on grid
        // nodes, so exactly 10^3 cell centres are inside and each cuboid
        // side covers 10 x 10 carved faces.
        let s = cuboid_surface(
            Vec3::new(0.25, 0.25, 0.25),
            Vec3::new(0.75, 0.75, 0.75),
            "boxWall",
        );
        let sum = write_carved_case(&dir, CaseKind::Big, 20, 20, 20, &s).expect("carve");

        assert_eq!(sum.n_cells_block, 8000);
        assert_eq!(sum.n_solid, 1000, "analytic solid count");
        assert_eq!(sum.n_fluid, 7000, "analytic fluid count");
        assert_eq!(sum.voted, 0, "untouched cells classify by parity alone");
        assert_eq!(sum.arbitrated, 0, "zero arbitration cost");
        assert_eq!(sum.wall_faces, vec![("boxWall".to_string(), 600)]);

        // §23.5 rows 3 and 6: the carved mesh reads back through the real
        // polyMesh reader, closes to round-off, and keeps the
        // upper-triangular (lduAddressing) face order.
        let m = build_host_mesh(&read_poly_mesh(&dir).expect("read")).expect("host mesh");
        let rep = m.check();
        assert_eq!(m.n_cells, 7000);
        assert!(rep.ldu_ordered, "carving broke the upper-triangular order");
        assert!(rep.max_closure_error < 1e-10, "closure {}", rep.max_closure_error);
        assert!(
            (rep.total_volume - 0.875).abs() < 1e-10,
            "fluid volume {} is not the analytic 0.875",
            rep.total_volume
        );

        let wall = m.patches.iter().find(|p| p.name == "boxWall").expect("boxWall patch");
        assert_eq!(wall.size, 600);
        assert_eq!(wall.kind, PatchKind::Wall);
        // The new patch's faces are one contiguous run at the end.
        assert_eq!(wall.start + wall.size, m.n_boundary_faces);

        // The new patch took EXACTLY the wall BCs the uncarved writer gives
        // its walls - through the same code path, not a copy of it.
        for (field, bc) in [
            ("U", "noSlip"),
            ("k", "kqRWallFunction"),
            ("epsilon", "epsilonWallFunction"),
            ("omega", "omegaWallFunction"),
            ("nut", "nutkWallFunction"),
        ] {
            let text = fs::read_to_string(dir.join("0").join(field)).expect(field);
            let at = text.find("boxWall").unwrap_or_else(|| panic!("{field}: no boxWall"));
            let entry = &text[at..text[at..].find('}').map_or(text.len(), |e| at + e)];
            assert!(entry.contains(bc), "{field}: boxWall entry lacks {bc}:\n{entry}");
        }

        let _ = fs::remove_dir_all(&dir);
    }

    /// SPEC-LIT §30.2: SST's wall distance (§6.6) must work on a carved mesh
    /// exactly as it does on a block one - `crate::walldistance::wall_distance`
    /// keys off `HostMesh::b_kind == PatchKind::Wall`, and §23.4's *DESIGN*
    /// choice to give a carved face the ordinary `wall` type is what makes
    /// that true with no special case anywhere. Finite and positive at every
    /// fluid cell, and zero at the carved wall boundary itself.
    #[test]
    fn sst_wall_distance_on_a_carved_mesh_is_finite_positive_and_zero_at_the_wall() -> Result<()> {
        let Ok(gpu) = crate::device::Gpu::new(0) else {
            return Ok(());
        };

        let s = cuboid_surface(
            Vec3::new(0.25, 0.25, 0.25),
            Vec3::new(0.75, 0.75, 0.75),
            "boxWall",
        );
        let b = case_block_spec(CaseKind::Big, 20, 20, 20);
        let (hm, summary) = build_carved_mesh(&b, &s)?;
        assert_eq!(summary.n_fluid, 7000);
        let (_, box_wall_faces) = summary
            .wall_faces
            .iter()
            .find(|(n, _)| n == "boxWall")
            .expect("boxWall carved");
        assert_eq!(*box_wall_faces, 600);

        let mesh = crate::mesh::GpuMesh::upload(&gpu, &hm)?;
        let ctrl = crate::io::case::SolverControls {
            solver: crate::io::case::LinearSolverKind::PCG,
            precon: crate::io::case::Preconditioner::Diagonal,
            tolerance: 1e-10,
            rel_tol: 0.0,
            max_iter: 5000,
            report_residuals: true,
            ..Default::default()
        };

        let wd = crate::walldistance::wall_distance(&gpu, &hm, &mesh, &ctrl, 0)?;
        // Both the block's own walls AND the carved `boxWall` faces must have
        // been fed to the Poisson solve - not just the 600 carved ones.
        assert!(
            wd.n_wall_faces > 600,
            "n_wall_faces = {}, expected more than the 600 carved boxWall faces alone",
            wd.n_wall_faces
        );

        let y = gpu.download(&wd.y.f)?;
        assert!(
            y.iter().all(|v| v.is_finite() && *v > 0.0),
            "every fluid cell must have a finite, positive wall distance"
        );
        let y_max = wd.max(&gpu)?;
        assert!(y_max.is_finite() && y_max > 0.0);
        // `NO_WALL` is the wall-FREE sentinel; a mesh with real walls must
        // never fall back to it silently.
        assert!(y_max < crate::walldistance::NO_WALL, "y_max = {y_max}");

        // Zero exactly on the wall boundary faces - the Dirichlet condition
        // the Poisson solve was given, not an approximation of it.
        let y_bf = gpu.download(&wd.y.bf)?;
        let boxwall = hm.patches.iter().find(|p| p.name == "boxWall").expect("boxWall patch");
        for i in 0..boxwall.size {
            let v = y_bf[boxwall.start + i];
            assert!(v.abs() < 1e-9, "boxWall face {i}: y = {v}, expected 0");
        }
        for p in hm.patches.iter().filter(|p| p.kind == PatchKind::Wall && p.name != "boxWall") {
            for i in 0..p.size {
                let v = y_bf[p.start + i];
                assert!(v.abs() < 1e-9, "{}: face {i}: y = {v}, expected 0", p.name);
            }
        }

        Ok(())
    }

    /// §B1 gate: `build_carved_mesh` (no file on disk anywhere) must agree
    /// with `write_carved_case` + the polyMesh reader on the same surface.
    #[test]
    fn build_carved_mesh_matches_the_file_round_trip() {
        let s = cuboid_surface(
            Vec3::new(0.25, 0.25, 0.25),
            Vec3::new(0.75, 0.75, 0.75),
            "boxWall",
        );

        let b = case_block_spec(CaseKind::Big, 20, 20, 20);
        let (direct, summary) = build_carved_mesh(&b, &s).expect("build_carved_mesh");
        assert_eq!(summary.n_fluid, 7000);
        assert_eq!(summary.wall_faces, vec![("boxWall".to_string(), 600)]);

        let dir = temp_dir("carve_direct_vs_disk");
        write_carved_case(&dir, CaseKind::Big, 20, 20, 20, &s).expect("write_carved_case");
        let via_disk = crate::io::polymesh::build_host_mesh(
            &crate::io::polymesh::read_poly_mesh(&dir).expect("read"),
        )
        .expect("host mesh");
        let _ = fs::remove_dir_all(&dir);

        assert_eq!(direct.n_cells, via_disk.n_cells);
        assert_eq!(direct.n_internal_faces, via_disk.n_internal_faces);
        assert_eq!(direct.n_boundary_faces, via_disk.n_boundary_faces);
        assert_eq!(direct.owner, via_disk.owner);
        assert_eq!(direct.neighbour, via_disk.neighbour);
        assert!((direct.v.iter().sum::<Scalar>() - via_disk.v.iter().sum::<Scalar>()).abs()
            < 1e3 * EPS);
        assert_eq!(direct.patches.len(), via_disk.patches.len());
        for (a, bb) in direct.patches.iter().zip(via_disk.patches.iter()) {
            assert_eq!(a.name, bb.name);
            assert_eq!(a.start, bb.start);
            assert_eq!(a.size, bb.size);
        }
    }

    /// §23.5 row 2: a sphere of r = 0.3 in the unit box at h = 1/40. The
    /// carved fluid volume must be within O(h) of `1 - 4 pi r^3 / 3` -
    /// bounded here by `h * (sphere area)`, the honest first-order estimate
    /// of what stair-stepping can displace.
    #[test]
    fn carved_sphere_volume_error_is_first_order_in_h() {
        use crate::io::polymesh::{build_host_mesh, read_poly_mesh};
        let pi = std::f64::consts::PI as Scalar;

        let dir = temp_dir("carve_sphere");
        let r: Scalar = 0.3;
        let s = sphere_surface(Vec3::new(0.5, 0.5, 0.5), r, 48, 96);
        assert_eq!(s.edge_defects(), (0, 0), "the test sphere must be closed");

        let sum = write_carved_case(&dir, CaseKind::Big, 40, 40, 40, &s).expect("carve");
        assert_eq!(sum.arbitrated, 0, "a smooth closed sphere needs no arbitration");
        assert!(sum.n_solid > 0 && sum.n_fluid > 0);

        let m = build_host_mesh(&read_poly_mesh(&dir).expect("read")).expect("host mesh");
        let rep = m.check();
        assert!(rep.ldu_ordered);
        assert!(rep.max_closure_error < 1e-10, "closure {}", rep.max_closure_error);

        let expect = 1.0 - 4.0 / 3.0 * pi * r * r * r;
        let err = (rep.total_volume - expect).abs();
        let h: Scalar = 1.0 / 40.0;
        let bound = 4.0 * pi * r * r * h; // area * h
        assert!(
            err < bound,
            "volume error {err} exceeds the O(h) bound {bound} \
             (got {}, expected {expect})",
            rep.total_volume
        );
        println!(
            "carved sphere: fluid volume {} vs analytic {expect}, |err| = {err} \
             (bound {bound}, h = {h})",
            rep.total_volume
        );

        let _ = fs::remove_dir_all(&dir);
    }

    /// §23.4 intake failures are loud: a surface that misses the block, and
    /// one that swallows it whole.
    #[test]
    fn carve_refuses_surfaces_outside_or_swallowing_the_domain() {
        let _guard = crate::io::contract::permissive_test_guard();
        crate::io::contract::set_permissive(false);
        let outside = cuboid_surface(Vec3::new(5.0, 5.0, 5.0), Vec3::new(6.0, 6.0, 6.0), "s");
        let e = match write_carved_case(Path::new("."), CaseKind::Big, 4, 4, 4, &outside) {
            Err(e) => e.to_string(),
            Ok(_) => panic!("a surface outside the domain was accepted"),
        };
        assert!(e.contains("entirely outside"), "{e}");

        let swallow =
            cuboid_surface(Vec3::new(-1.0, -1.0, -1.0), Vec3::new(2.0, 2.0, 2.0), "s");
        let e = match write_carved_case(Path::new("."), CaseKind::Big, 4, 4, 4, &swallow) {
            Err(e) => e.to_string(),
            Ok(_) => panic!("a surface swallowing the domain was accepted"),
        };
        assert!(e.contains("swallows the whole domain"), "{e}");
    }

    // ---- cut-cell meshing (SPEC-LIT §24) ----------------------------------

    /// §24.6 row 2: a grid-aligned cuboid must reproduce castellation's cell
    /// counts exactly (no cell mixed) and close to round-off.
    #[test]
    fn cutcell_grid_aligned_cuboid_matches_castellation() {
        let s = cuboid_surface(Vec3::new(0.25, 0.25, 0.25), Vec3::new(0.75, 0.75, 0.75), "boxWall");
        let b = case_block_spec(CaseKind::Big, 20, 20, 20);
        let (mesh, summary) =
            build_cutcell_mesh_with(&b, &s, DEFAULT_SUPERSAMPLE, DEFAULT_THETA_MIN).expect("cutcell");

        assert_eq!(summary.n_solid, 1000, "analytic solid count");
        assert_eq!(summary.n_fluid_full, 7000, "analytic fluid count");
        assert_eq!(summary.n_cut, 0, "a grid-aligned cuboid must cut nothing");
        assert_eq!(mesh.n_cells, 7000);

        let rep = mesh.check();
        assert!(rep.ldu_ordered, "cut-cell assembly broke the upper-triangular order");
        assert!(rep.max_closure_error < 1e-10, "closure {}", rep.max_closure_error);
        assert!(
            (rep.total_volume - 0.875).abs() < 1e-10,
            "fluid volume {} is not the analytic 0.875",
            rep.total_volume
        );
        println!(
            "cutcell grid-aligned cuboid: {} cells, closure {:e}, volume {}",
            mesh.n_cells, rep.max_closure_error, rep.total_volume
        );
    }

    /// §24.6 row 4: the cut-cell sphere's IN-MEMORY volume (theta_c * V_full,
    /// written by `override_cutcell_geometry` - exact, not the disk path's
    /// pyramid approximation) must be much closer to the analytic volume than
    /// castellation's, and closure must still hold to round-off.
    #[test]
    fn cutcell_sphere_volume_much_closer_than_castellation() {
        let pi = std::f64::consts::PI as Scalar;
        let r: Scalar = 0.3;
        let surf = sphere_surface(Vec3::new(0.5, 0.5, 0.5), r, 48, 96);
        let b = case_block_spec(CaseKind::Big, 40, 40, 40);

        let (mesh, summary) =
            build_cutcell_mesh_with(&b, &surf, DEFAULT_SUPERSAMPLE, DEFAULT_THETA_MIN).expect("cutcell");
        assert!(summary.n_cut > 0);

        let rep = mesh.check();
        assert!(rep.ldu_ordered);
        assert!(rep.max_closure_error < 1e-10, "closure {}", rep.max_closure_error);

        let expect = 1.0 - 4.0 / 3.0 * pi * r * r * r;
        let err_cut = (rep.total_volume - expect).abs();
        let h: Scalar = 1.0 / 40.0;
        let err_castellated_bound = 4.0 * pi * r * r * h; // O(h), §23.5's own bound
        println!(
            "cutcell sphere: volume {} vs analytic {expect}, |err| = {err_cut:e} \
             (castellation's own O(h) bound is {err_castellated_bound:e})",
            rep.total_volume
        );
        assert!(
            err_cut < 0.2 * err_castellated_bound,
            "cut-cell error {err_cut} is not much smaller than castellation's O(h) bound \
             {err_castellated_bound}"
        );
    }

    /// §24.6 row 5, at the mesh-assembly level: after merging, `check()` sees
    /// no non-positive or absurdly small cell.
    #[test]
    fn cutcell_merging_leaves_a_sane_mesh() {
        let r: Scalar = 0.3;
        let surf = sphere_surface(Vec3::new(0.501_3, 0.499_7, 0.502_1), r, 32, 64);
        let b = case_block_spec(CaseKind::Big, 30, 30, 30);

        let (mesh, summary) =
            build_cutcell_mesh_with(&b, &surf, DEFAULT_SUPERSAMPLE, DEFAULT_THETA_MIN).expect("cutcell");
        assert!(summary.n_merged > 0, "this off-grid sphere must produce slivers to merge");
        assert_eq!(
            mesh.n_cells,
            summary.n_fluid_full + summary.n_cut - summary.n_merged,
            "n_cells_out bookkeeping"
        );

        let rep = mesh.check();
        assert!(rep.min_volume > 0.0, "a merged mesh must have no non-positive cell");
        let h3 = (1.0 / 30.0 as Scalar).powi(3);
        assert!(
            rep.min_volume >= DEFAULT_THETA_MIN * h3 * 0.999,
            "smallest surviving cell {} is below theta_min * V_full = {}",
            rep.min_volume,
            DEFAULT_THETA_MIN * h3
        );
        println!(
            "cutcell merge: {} slivers merged, {} cells remain, min volume {} (theta_min*V_full = {})",
            summary.n_merged, mesh.n_cells, rep.min_volume, DEFAULT_THETA_MIN * h3
        );
    }

    /// The disk path: `write_cutcell_case` + the real polyMesh reader must
    /// produce a mesh that closes to round-off (closure needs only `Sf`,
    /// which `synthetic_quad` reproduces exactly regardless of the volume
    /// caveat) and whose topology matches the in-memory summary.
    #[test]
    fn write_cutcell_case_round_trips_and_closes() {
        use crate::io::polymesh::{build_host_mesh as read_build_host_mesh, read_poly_mesh};

        let s = cuboid_surface(Vec3::new(0.25, 0.25, 0.25), Vec3::new(0.75, 0.75, 0.75), "boxWall");
        let dir = temp_dir("cutcell_disk");
        let summary = write_cutcell_case(
            &dir, CaseKind::Big, 20, 20, 20, &s, DEFAULT_SUPERSAMPLE, DEFAULT_THETA_MIN,
        )
        .expect("write_cutcell_case");
        assert_eq!(summary.n_cells_out, 7000);

        let m = read_build_host_mesh(&read_poly_mesh(&dir).expect("read")).expect("host mesh");
        assert_eq!(m.n_cells, 7000);
        let rep = m.check();
        assert!(rep.ldu_ordered);
        assert!(rep.max_closure_error < 1e-10, "closure {}", rep.max_closure_error);

        // §24 gate honesty check (this module's *DESIGN* note): on THIS
        // grid-aligned cuboid every face is alpha in {0,1}, so the pyramid
        // volume the disk path derives is not an approximation at all here -
        // it must match the analytic volume too.
        assert!(
            (rep.total_volume - 0.875).abs() < 1e-8,
            "disk-path volume {} vs analytic 0.875",
            rep.total_volume
        );

        for field in ["U", "k", "epsilon", "omega", "nut"] {
            assert!(dir.join("0").join(field).exists(), "missing 0/{field}");
        }
        for (field, bc) in [
            ("U", "noSlip"),
            ("k", "kqRWallFunction"),
            ("epsilon", "epsilonWallFunction"),
        ] {
            let text = fs::read_to_string(dir.join("0").join(field)).expect(field);
            assert!(text.contains("boxWall"), "{field}: no boxWall entry");
            assert!(text.contains(bc), "{field}: boxWall entry lacks {bc}");
        }

        let _ = fs::remove_dir_all(&dir);
    }

    /// The disk path on a genuinely CUT mesh: closure still holds exactly
    /// (it only needs `Sf`), and the pyramid-derived volume is reported next
    /// to the exact in-memory one so the module doc's caveat is a measured
    /// number, not a guess.
    #[test]
    fn write_cutcell_case_sphere_closes_and_reports_the_volume_gap() {
        use crate::io::polymesh::{build_host_mesh as read_build_host_mesh, read_poly_mesh};

        let r: Scalar = 0.3;
        let surf = sphere_surface(Vec3::new(0.5, 0.5, 0.5), r, 48, 96);
        let dir = temp_dir("cutcell_disk_sphere");
        let exact = build_cutcell_mesh_with(
            &case_block_spec(CaseKind::Big, 40, 40, 40),
            &surf,
            DEFAULT_SUPERSAMPLE,
            DEFAULT_THETA_MIN,
        )
        .expect("in-memory cutcell");

        let summary = write_cutcell_case(
            &dir, CaseKind::Big, 40, 40, 40, &surf, DEFAULT_SUPERSAMPLE, DEFAULT_THETA_MIN,
        )
        .expect("write_cutcell_case");
        assert_eq!(summary.n_cells_out, exact.0.n_cells);

        let m = read_build_host_mesh(&read_poly_mesh(&dir).expect("read")).expect("host mesh");
        let rep = m.check();
        assert!(rep.ldu_ordered);
        assert!(rep.max_closure_error < 1e-10, "disk-path closure {}", rep.max_closure_error);

        let exact_volume = exact.0.check().total_volume;
        println!(
            "cutcell disk-path volume gap: exact (in-memory) {exact_volume}, \
             pyramid-derived (disk) {} - relative gap {:.4}%",
            rep.total_volume,
            100.0 * (rep.total_volume - exact_volume).abs() / exact_volume
        );

        let _ = fs::remove_dir_all(&dir);
    }
    /// The room's door window: on the +x slot, named `outlet`, snapped to
    /// roughly 2 m x 2 m starting at the floor.
    #[test]
    fn room_spec_has_a_floor_mounted_door_on_plus_x() {
        let (nx, ny, nz) = CaseKind::Room.default_resolution();
        let b = case_block_spec(CaseKind::Room, nx, ny, nz);
        let w = b.windows.first().expect("room has a door window");
        assert_eq!(w.slot, 1);
        assert_eq!(w.name, "outlet");
        let ynodes = graded_nodes(&b.y);
        let znodes = graded_nodes(&b.z);
        let wy = ynodes[w.hi[0]] - ynodes[w.lo[0]];
        let wz = znodes[w.hi[1]] - znodes[w.lo[1]];
        assert!((wy - 2.0).abs() < 0.11, "door width {wy}");
        assert!((wz - 2.0).abs() < 0.11, "door height {wz}");
        assert_eq!(w.lo[1], 0, "door starts at the floor");
    }

}
