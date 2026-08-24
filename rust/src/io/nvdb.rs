// meteor-cfd - Copyright (c) 2026 주식회사 메테오시뮬레이션 (Meteo Simulation Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

//! NanoVDB (.nvdb) file writer, pure Rust, from the byte layout the
//! Apache-2.0 NanoVDB.h/PNanoVDB.h headers specify. Codec NONE.
//!
//! Written from `nanovdb/nanovdb/NanoVDB.h`, `nanovdb/nanovdb/io/IO.h` and
//! `nanovdb/nanovdb/tools/CreateNanoGrid.h`, all Apache-2.0, fetched from
//! `github.com/AcademySoftwareFoundation/openvdb` at the `v13.0.0` tag (NanoVDB
//! file-format major.minor.patch `32.9.0`, `NANOVDB_USE_NEW_MAGIC_NUMBERS` and
//! `NANOVDB_USE_SINGLE_ROOT_KEY` both on, which is the default in that header).
//! The struct-layout comments in `NanoVDB.h` (byte offsets for `GridData`,
//! sizes for `FileHeader`/`FileMetaData`/`TreeData`) are taken as authoritative
//! and cross-checked here against that header's own `padding()` accounting
//! formulas for `RootData`, `InternalData` and `LeafData`, and against
//! `io::writeUncompressedGrid` for the segment (`FileHeader` + `FileMetaData`
//! + name + grid blob) framing and `io::writeGrids` for the multi-grid-file
//! convention (one independent segment per grid, concatenated - not one
//! shared header). File FORMATS are not copyrightable and these are published
//! byte layouts, not code; the header text is quoted only in comments, for
//! traceability, never executed. No NanoVDB *implementation* source (the
//! templated node classes, accessors, tools) was translated or adapted - this
//! file is an independent writer built from the documented layout.
//!
//! # What this writer produces
//!
//! One independent NanoVDB *segment* per scalar grid: `FileHeader` (16 B,
//! `gridCount = 1`) + `FileMetaData` (176 B) + the grid's name + a grid blob
//! (`GridData` 672 B, `TreeData` 64 B, one `RootData`, its table of root
//! tiles, then the Upper (32³ fan-out), Lower (16³) and Leaf (8³) node
//! blocks, top-down and each block internally contiguous - `GridFlags::
//! IsBreadthFirst`). A caller's `&[OutputField]` becomes one segment per
//! scalar field; a vector field becomes four (`name.x`, `name.y`, `name.z`,
//! `name.mag`) rather than one `Vec3f` grid, because a `Vec3f` NanoVDB grid
//! is the documented failure mode in `docs/05-io-redesign.md` §8 Q4: it
//! breaks in Omniverse and only round-trips through Blender. Multiple
//! segments concatenate into one `.nvdb` file exactly as `nanovdb::io::
//! writeGrids` produces one, so any reader that loops "read a segment until
//! EOF" (which every conformant reader must, since that is the only
//! documented multi-grid convention) reads this file.
//!
//! Every tile/child/table byte layout below (`RootData`, `RootData::Tile`,
//! `InternalData` for the Lower and Upper levels, `LeafData`) is derived
//! generically at write time by [`Layout`], a tiny C-struct-layout replica
//! (pad each field to its own alignment, round the final size up to the
//! type's `alignas(32)`) rather than hard-coded, so the same code produces
//! the right byte offsets for both value widths below. The sizes it computes
//! (`RootData` 64 B, `RootData::Tile` 32 B, Lower `InternalData` 33 856 B,
//! Upper `InternalData` 270 400 B, `float` `LeafData` 2 144 B, `Half`
//! `LeafData` 1 120 B) are pinned by [`tests::layout_sizes_match_the_header`],
//! cross-checked against `NanoVDB.h`'s own `padding()` formulas by hand in
//! this file's design notes.
//!
//! # Precision: `fp32` is the real `GridType::Float`; `fp16` is `GridType::
//! Half`, hand-converted
//!
//! `fp32` writes `GridType::Float` (`= 1`) with the plain IEEE 754 binary32
//! values every NanoVDB reader supports. `fp16` writes `GridType::Half`
//! (`= 9`), which `NanoVDB.h` itself labels "placeholder for IEEE 754 Half"
//! and for which the header never instantiates a concrete `HalfTree`/
//! `HalfGrid` (contrast `Fp4Tree`, `Fp8Tree`, `Fp16Tree`, `FpNTree`, which
//! *are* instantiated - and which are NanoVDB's own min+scale quantisation
//! scheme, not IEEE 754 half floats, so they are not an option here: the
//! task is literally IEEE 754 binary16, round-to-nearest-even, done by
//! hand). The generic `LeafData<ValueT>`/`InternalData<ValueT>`/
//! `RootData<ValueT>` templates are defined for any POD `ValueT`, so a
//! 2-byte value type is a legal instance of the documented layout even
//! though the header ships no named alias for it; whether third-party tools
//! that switch on `GridType` happen to special-case `Half` is a different
//! question, which this file does not claim to answer. **`fp16` readability
//! by external tools (Blender, Houdini, ParaView) is externally unverified -
//! only this module's own reader round-trips it, bit-exact.** `fp32` is the
//! interoperable, spec-conformant path and is what should be reached for
//! when a case needs both correctness and a tool to actually open the file.
//! The binary16 conversion itself ([`f32_to_f16_bits`] / [`f16_bits_to_f32`])
//! is ordinary IEEE 754 arithmetic - not implementation-specific to any
//! codebase - with round-to-nearest-even and its own unit tests.
//!
//! # World/index transform - *DESIGN*
//!
//! `NanoVDB.h`'s `Map` is an affine `index -> world` transform; nothing in
//! the header prescribes what a Cartesian CFD block's `origin` should mean
//! under it. Here `origin` is the world position of cell `(0,0,0)`'s
//! *centre* (this crate's own cell-centred convention), so
//! `applyMap(i,j,k) = origin + spacing * (i,j,k)` places the sample point of
//! cell `(i,j,k)` correctly and `worldBBox` is widened by half a voxel on
//! every side of `indexBBox` to describe the domain's physical extent
//! rather than only the lattice of sample points. `gridClass` is set to
//! `FogVolume` (`= 2`, "density"-shaped scalar field) for every scalar/
//! per-component grid this writer produces, so a viewer's volume-rendering
//! path picks it up by default; this is a rendering hint, not a physical
//! claim, and is ours to choose.
//!
//! Fields are read in this crate's existing Cartesian flattening,
//! `idx(i,j,k) = i + nx*(j + ny*k)` (matching `pressure::cartesian`'s
//! `CartesianGrid::ijk`), and `nx*ny*nz` must equal every field's length.
//!
//! No GPL-licensed source was consulted.

use crate::error::{Error, IoContext, Result};
use crate::io::output_types::{FieldValues, OutputField};
use crate::{Scalar, Vec3};
use std::path::Path;

// ============================================================================
//  Public API
// ============================================================================

/// Per-voxel value precision. See the module doc for what each one really
/// is on disk.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Precision {
    /// `GridType::Float` - real, interoperable, IEEE 754 binary32.
    F32,
    /// `GridType::Half` - IEEE 754 binary16, hand-converted; see the module
    /// doc for why this is not NanoVDB's own `Fp16` quantisation, and for
    /// the external-readability caveat.
    F16,
}

impl Precision {
    fn width(self) -> usize {
        match self {
            Precision::F32 => 4,
            Precision::F16 => 2,
        }
    }
}

/// A uniform Cartesian block: `nx*ny*nz` cells, cell `(0,0,0)` centred at
/// `origin`, spacing `spacing` per axis (anisotropic spacing is legal - a
/// Cartesian pressure grid in this crate already carries `dx != dy != dz`,
/// see `pressure::cartesian::CartesianGrid`).
#[derive(Debug, Clone, Copy)]
pub struct UniformGrid {
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    pub origin: Vec3,
    pub spacing: Vec3,
}

impl UniformGrid {
    pub fn n(&self) -> usize {
        self.nx * self.ny * self.nz
    }

    /// This crate's standard Cartesian flattening, `i + nx*(j + ny*k)`
    /// (x fastest) - matching `pressure::cartesian::CartesianGrid::ijk` and
    /// shared with `vdb.rs`, which writes the same `UniformGrid`.
    #[inline]
    pub fn idx(&self, i: usize, j: usize, k: usize) -> usize {
        i + self.nx * (j + self.ny * k)
    }
}

/// Write `fields` on `grid` to `path` as a sequence of NanoVDB segments -
/// see the module doc. A `FieldValues::Vector` field becomes four segments
/// (`name.x`, `name.y`, `name.z`, `name.mag`); a `FieldValues::Scalar` field
/// becomes one.
pub fn write(
    path: impl AsRef<Path>,
    grid: &UniformGrid,
    fields: &[OutputField],
    precision: Precision,
) -> Result<()> {
    let path = path.as_ref();
    if grid.nx == 0 || grid.ny == 0 || grid.nz == 0 {
        return Err(Error::Config(format!(
            "nvdb: grid dims must all be positive, got {}x{}x{}",
            grid.nx, grid.ny, grid.nz
        )));
    }
    if fields.is_empty() {
        return Err(Error::Config("nvdb: no fields to write".into()));
    }

    let n = grid.n();
    let mut out = Vec::new();
    for field in fields {
        match &field.values {
            FieldValues::Scalar(v) => {
                check_len(field.name, v.len(), n)?;
                append_segment(&mut out, grid, field.name, v, precision)?;
            }
            FieldValues::Vector(v) => {
                check_len(field.name, v.len(), n)?;
                let xs: Vec<Scalar> = v.iter().map(|p| p.x).collect();
                let ys: Vec<Scalar> = v.iter().map(|p| p.y).collect();
                let zs: Vec<Scalar> = v.iter().map(|p| p.z).collect();
                let mags: Vec<Scalar> = v.iter().map(|p| p.mag()).collect();
                append_segment(&mut out, grid, &format!("{}.x", field.name), &xs, precision)?;
                append_segment(&mut out, grid, &format!("{}.y", field.name), &ys, precision)?;
                append_segment(&mut out, grid, &format!("{}.z", field.name), &zs, precision)?;
                append_segment(&mut out, grid, &format!("{}.mag", field.name), &mags, precision)?;
            }
        }
    }

    std::fs::write(path, &out).path(path)
}

fn check_len(field: &str, got: usize, want: usize) -> Result<()> {
    if got != want {
        return Err(Error::Field {
            field: field.to_string(),
            msg: format!("{got} values but the grid has {want} cells"),
        });
    }
    Ok(())
}

// ============================================================================
//  Format constants - NanoVDB.h, v13.0.0
// ============================================================================

/// `NANOVDB_MAGIC_FILE` - "NanoVDB2" - `FileHeader::magic`.
const MAGIC_FILE: u64 = 0x324244566f6e614e;
/// `NANOVDB_MAGIC_GRID` - "NanoVDB1" - `GridData::mMagic` under
/// `NANOVDB_USE_NEW_MAGIC_NUMBERS` (the header's default).
const MAGIC_GRID: u64 = 0x314244566f6e614e;
/// `Version(32, 9, 0)`: `major<<21 | minor<<10 | patch`, the version this
/// tag of `NanoVDB.h` stamps by default.
const VERSION: u32 = (32u32 << 21) | (9u32 << 10);
/// `Checksum::EMPTY64` - all 64 bits on means "checksum disabled". CRC32 is
/// not implemented here; a disabled checksum is a documented, legal state
/// (`GridData::init` sets it by default) and every reader must tolerate it.
const CHECKSUM_DISABLED: u64 = u64::MAX;
/// `Codec::NONE`.
const CODEC_NONE: u16 = 0;

/// `GridType::Float`.
const GRID_TYPE_FLOAT: u32 = 1;
/// `GridType::Half`.
const GRID_TYPE_HALF: u32 = 9;
/// `GridClass::FogVolume` - *DESIGN*, see the module doc.
const GRID_CLASS_FOG_VOLUME: u32 = 2;

/// `GridFlags::HasBBox | HasMinMax | HasAverage | HasStdDeviation |
/// IsBreadthFirst`. `HasLongGridName` is never set: names longer than 255
/// bytes are rejected outright rather than routed through blind-data
/// (`append_segment` enforces this).
const GRID_FLAGS: u32 = (1 << 1) | (1 << 2) | (1 << 3) | (1 << 4) | (1 << 5);

const FILE_HEADER_SIZE: usize = 16;
const FILE_META_SIZE: usize = 176;
const GRID_DATA_SIZE: usize = 672;
const TREE_DATA_SIZE: usize = 64;

/// Child fan-out (voxels per axis) at each level: leaf = 8³, one lower
/// child covers 16 leaves per axis = 128 voxels, one upper child covers 32
/// lowers per axis = 4096 voxels. `LOG2DIM` per level: leaf 3, lower 4,
/// upper 5 - the "8³/16³/32³" of the task brief and of `NanoVDB.h`'s
/// `NanoLeaf`/`NanoLower`/`NanoUpper` aliases.
const LEAF_LOG2DIM: u32 = 3;
const LOWER_LOG2DIM: u32 = 4;
const UPPER_LOG2DIM: u32 = 5;
const LEAF_DIM: usize = 1 << LEAF_LOG2DIM; // 8
const LOWER_DIM: usize = LEAF_DIM << LOWER_LOG2DIM; // 128
const UPPER_DIM: usize = LOWER_DIM << UPPER_LOG2DIM; // 4096
/// `ChildT::TOTAL` for the Upper node's root key (`CoordToKey` masks off
/// this many low bits): `UPPER_LOG2DIM + LOWER_LOG2DIM + LEAF_LOG2DIM`.
const UPPER_TOTAL: u32 = UPPER_LOG2DIM + LOWER_LOG2DIM + LEAF_LOG2DIM;

#[inline]
fn ceil_div(a: usize, b: usize) -> usize {
    (a + b - 1) / b
}

// ============================================================================
//  Generic C-struct layout - pads each field to the alignment a C++
//  compiler would use, so RootData/InternalData/LeafData/Tile offsets are
//  computed once, correctly, for either value width, instead of hard-coded
//  per-width by hand.
// ============================================================================

struct Layout {
    pos: usize,
}

impl Layout {
    fn new() -> Self {
        Self { pos: 0 }
    }

    /// Reserve `size` bytes aligned to `align`; return the field's offset.
    fn field(&mut self, size: usize, align: usize) -> usize {
        self.align_to(align);
        let off = self.pos;
        self.pos += size;
        off
    }

    fn align_to(&mut self, align: usize) {
        let rem = self.pos % align;
        if rem != 0 {
            self.pos += align - rem;
        }
    }

    /// The struct's `alignas(32)`-driven total size: alignment forces the
    /// size itself up to the next multiple of `align` too.
    fn finish(mut self, align: usize) -> usize {
        self.align_to(align);
        self.pos
    }
}

struct RootLayout {
    bbox: usize,
    table_size: usize,
    background: usize,
    minimum: usize,
    maximum: usize,
    average: usize,
    stddev: usize,
    size: usize,
}

/// `RootData<ChildT>`: `mBBox`(24) `mTableSize`(4) `mBackground/mMinimum/
/// mMaximum`(`ValueT`, width `w`) `mAverage/mStdDevi`(`float`, `FloatTraits`
/// maps any non-8-byte `ValueT` to `float`), `alignas(32)`.
fn root_layout(w: usize) -> RootLayout {
    let mut l = Layout::new();
    let bbox = l.field(24, 4);
    let table_size = l.field(4, 4);
    let background = l.field(w, w);
    let minimum = l.field(w, w);
    let maximum = l.field(w, w);
    let average = l.field(4, 4);
    let stddev = l.field(4, 4);
    let size = l.finish(32);
    RootLayout { bbox, table_size, background, minimum, maximum, average, stddev, size }
}

struct TileLayout {
    key: usize,
    child: usize,
    state: usize,
    value: usize,
    size: usize,
}

/// `RootData::Tile`: `key`(`KeyT` = `uint64_t`, `NANOVDB_USE_SINGLE_ROOT_KEY`
/// is the header's default) `child`(`int64_t`) `state`(`uint32_t`)
/// `value`(`ValueT`), itself `alignas(32)`.
fn root_tile_layout(w: usize) -> TileLayout {
    let mut l = Layout::new();
    let key = l.field(8, 8);
    let child = l.field(8, 8);
    let state = l.field(4, 4);
    let value = l.field(w, w);
    let size = l.finish(32);
    TileLayout { key, child, state, value, size }
}

struct InternalLayout {
    bbox: usize,
    flags: usize,
    /// Reserved for parity with the format (a Lower/Upper node's tile mask
    /// for *constant-value* tiles). This writer always subdivides a dense
    /// box down to Leaf level, so no Lower/Upper node ever has a
    /// constant-value tile and nothing ever sets or reads a bit here - the
    /// field earns its byte offset (it still has to be skipped so
    /// `child_mask` and the rest land in the right place) without earning a
    /// reader.
    #[allow(dead_code)]
    value_mask: usize,
    child_mask: usize,
    minimum: usize,
    maximum: usize,
    average: usize,
    stddev: usize,
    table: usize,
    #[allow(dead_code)] // kept for documentation; every caller uses `value_mask`/`child_mask` directly
    mask_bytes: usize,
    /// `sizeof(union { ValueT value; int64_t child; })`, which the
    /// `int64_t` floors at 8 for both value widths used here.
    entry_size: usize,
    /// `2^(3*LOG2DIM)` - the writer derives this itself from `LOG2DIM`
    /// where it needs it (the `0..32`/`0..16` loops in `build_upper`/
    /// `build_lower`); only the reader walks the table generically by
    /// `slots`.
    #[allow(dead_code)]
    slots: usize,
    size: usize,
}

/// `InternalData<ChildT, LOG2DIM>` (the Lower and Upper node types):
/// `mBBox`(24) `mFlags`(`uint64_t`) `mValueMask`/`mChildMask`(`Mask<LOG2DIM>`,
/// `2^(3*LOG2DIM)` bits each) `mMinimum/mMaximum`(`ValueT`) `mAverage/
/// mStdDevi`(`float`), then `alignas(32) Tile mTable[2^(3*LOG2DIM)]`
/// (the array field itself carries `alignas(32)`, so it starts at the next
/// 32-byte boundary regardless of what precedes it).
fn internal_layout(w: usize, log2dim: u32) -> InternalLayout {
    let slots = 1usize << (3 * log2dim);
    let mask_bytes = slots / 8;
    let mut l = Layout::new();
    let bbox = l.field(24, 4);
    let flags = l.field(8, 8);
    let value_mask = l.field(mask_bytes, 8);
    let child_mask = l.field(mask_bytes, 8);
    let minimum = l.field(w, w);
    let maximum = l.field(w, w);
    let average = l.field(4, 4);
    let stddev = l.field(4, 4);
    l.align_to(32);
    let entry_size = w.max(8);
    let table = l.field(entry_size * slots, 8);
    let size = l.finish(32);
    InternalLayout {
        bbox, flags, value_mask, child_mask, minimum, maximum, average, stddev, table,
        mask_bytes, entry_size, slots, size,
    }
}

struct LeafLayout {
    bbox_min: usize,
    bbox_dif: usize,
    flags: usize,
    value_mask: usize,
    minimum: usize,
    maximum: usize,
    average: usize,
    stddev: usize,
    values: usize,
    value_width: usize,
    size: usize,
}

/// `LeafData<ValueT>`: `mBBoxMin`(`Coord`,12) `mBBoxDif`(3x`uint8_t`)
/// `mFlags`(`uint8_t`) `mValueMask`(`Mask<3>`, fixed 64 B) `mMinimum/
/// mMaximum`(`ValueT`) `mAverage/mStdDevi`(`float`), then
/// `alignas(32) ValueT mValues[512]`.
fn leaf_layout(w: usize) -> LeafLayout {
    let mut l = Layout::new();
    let bbox_min = l.field(12, 4);
    let bbox_dif = l.field(3, 1);
    let flags = l.field(1, 1);
    let value_mask = l.field(64, 8); // Mask<3>: 512 bits = 64 B, fixed
    let minimum = l.field(w, w);
    let maximum = l.field(w, w);
    let average = l.field(4, 4);
    let stddev = l.field(4, 4);
    l.align_to(32);
    let values = l.field(w * 512, 8);
    let size = l.finish(32);
    LeafLayout {
        bbox_min, bbox_dif, flags, value_mask, minimum, maximum, average, stddev, values,
        value_width: w, size,
    }
}

// ============================================================================
//  Byte-buffer primitives (all little-endian, regardless of host)
// ============================================================================

fn put_bytes(b: &mut [u8], off: usize, bytes: &[u8]) {
    b[off..off + bytes.len()].copy_from_slice(bytes);
}
fn put_u16(b: &mut [u8], off: usize, v: u16) {
    put_bytes(b, off, &v.to_le_bytes());
}
fn put_u32(b: &mut [u8], off: usize, v: u32) {
    put_bytes(b, off, &v.to_le_bytes());
}
fn put_i32(b: &mut [u8], off: usize, v: i32) {
    put_bytes(b, off, &v.to_le_bytes());
}
fn put_u64(b: &mut [u8], off: usize, v: u64) {
    put_bytes(b, off, &v.to_le_bytes());
}
fn put_i64(b: &mut [u8], off: usize, v: i64) {
    put_bytes(b, off, &v.to_le_bytes());
}
fn put_f32(b: &mut [u8], off: usize, v: f32) {
    put_bytes(b, off, &v.to_le_bytes());
}
fn put_f64(b: &mut [u8], off: usize, v: f64) {
    put_bytes(b, off, &v.to_le_bytes());
}

// The `get_*` family and `mask_bit` are read-side only; the writer never
// reads back what it just wrote (`set_mask_bit` does a read-modify-write on
// a mask word, so `get_u64` alone is the one exception, needed unconditionally
// below). They exist for `mod reader`, `#[cfg(test)]`-only - see that
// module's doc.
#[cfg(test)]
fn get_u16(b: &[u8], off: usize) -> u16 {
    u16::from_le_bytes([b[off], b[off + 1]])
}
#[cfg(test)]
fn get_u32(b: &[u8], off: usize) -> u32 {
    u32::from_le_bytes([b[off], b[off + 1], b[off + 2], b[off + 3]])
}
#[cfg(test)]
fn get_i32(b: &[u8], off: usize) -> i32 {
    i32::from_le_bytes([b[off], b[off + 1], b[off + 2], b[off + 3]])
}
fn get_u64(b: &[u8], off: usize) -> u64 {
    let mut a = [0u8; 8];
    a.copy_from_slice(&b[off..off + 8]);
    u64::from_le_bytes(a)
}
#[cfg(test)]
fn get_i64(b: &[u8], off: usize) -> i64 {
    let mut a = [0u8; 8];
    a.copy_from_slice(&b[off..off + 8]);
    i64::from_le_bytes(a)
}
#[cfg(test)]
fn get_f32(b: &[u8], off: usize) -> f32 {
    f32::from_le_bytes([b[off], b[off + 1], b[off + 2], b[off + 3]])
}
#[cfg(test)]
fn get_f64(b: &[u8], off: usize) -> f64 {
    let mut a = [0u8; 8];
    a.copy_from_slice(&b[off..off + 8]);
    f64::from_le_bytes(a)
}

fn put_value(b: &mut [u8], off: usize, v: f64, precision: Precision) {
    match precision {
        Precision::F32 => put_f32(b, off, v as f32),
        Precision::F16 => put_u16(b, off, f32_to_f16_bits(v as f32)),
    }
}
#[cfg(test)]
fn get_value(b: &[u8], off: usize, precision: Precision) -> f64 {
    match precision {
        Precision::F32 => get_f32(b, off) as f64,
        Precision::F16 => f16_bits_to_f32(get_u16(b, off)) as f64,
    }
}

fn set_mask_bit(b: &mut [u8], mask_off: usize, n: u32) {
    let word_off = mask_off + 8 * (n as usize / 64);
    let bit = n % 64;
    let mut w = get_u64(b, word_off);
    w |= 1u64 << bit;
    put_u64(b, word_off, w);
}
#[cfg(test)]
fn mask_bit(b: &[u8], mask_off: usize, n: u32) -> bool {
    let word_off = mask_off + 8 * (n as usize / 64);
    let w = get_u64(b, word_off);
    (w >> (n % 64)) & 1 != 0
}

// ============================================================================
//  IEEE 754 binary16 <-> binary32, by hand, round-to-nearest-even.
//
//  Pure applied arithmetic from the IEEE 754 bit layout (1 sign / 5 exponent
//  / 10 mantissa bits, bias 15), not derived from any codebase's half-float
//  routine. See tests below for round-trip and rounding-boundary coverage.
// ============================================================================

fn f32_to_f16_bits(f: f32) -> u16 {
    let bits = f.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let mantissa = bits & 0x7f_ffff;

    if exp == 0xff {
        return if mantissa == 0 {
            sign | 0x7c00 // +/- infinity
        } else {
            sign | 0x7e00 // NaN (quiet, payload not preserved)
        };
    }
    if exp == 0 {
        // f32 subnormal: magnitude <= (2^-126), which underflows every
        // representable half (smallest half subnormal is 2^-24).
        return sign;
    }

    let unbiased = exp - 127;
    if unbiased > 15 {
        return sign | 0x7c00; // overflow -> infinity
    }
    if unbiased < -14 {
        // Half subnormal (or underflows to zero once rounded).
        let with_implicit = mantissa | 0x80_0000; // 24 significant bits
        let shift = (-unbiased - 1) as u32; // >= 14
        let m = round_shift_rne(with_implicit, shift);
        // `m` rolling over to 0x400 lands exactly on the smallest normal
        // half's bit pattern (exponent field 1, mantissa 0) - no special
        // case needed, the bits already say the right thing.
        return sign | (m as u16);
    }

    let half_exp = (unbiased + 15) as u32;
    let mut m = round_shift_rne(mantissa, 13);
    let mut e = half_exp;
    if m & 0x400 != 0 {
        m = 0;
        e += 1;
    }
    if e >= 31 {
        return sign | 0x7c00; // rounded into overflow
    }
    sign | ((e as u16) << 10) | (m as u16)
}

/// Round `value >> shift` to the nearest integer, ties to even. Only called
/// with `shift` in `1..=24` (`value` never exceeds 24 significant bits
/// here), which keeps `1u32 << (shift - 1)` in range.
fn round_shift_rne(value: u32, shift: u32) -> u32 {
    debug_assert!((1..=24).contains(&shift));
    let half = 1u32 << (shift - 1);
    let mask = (1u32 << shift) - 1;
    let remainder = value & mask;
    let mut result = value >> shift;
    if remainder > half || (remainder == half && result & 1 == 1) {
        result += 1;
    }
    result
}

/// Widening half -> float is always exact (more mantissa bits, same-or-wider
/// exponent range), so this can go through `f64` arithmetic instead of bit
/// surgery: every intermediate value (`mant/1024.0`, `2f64.powi(k)`, their
/// product) is exactly representable and exactly computed in binary
/// floating point.
///
/// Only the reader and the half-conversion tests need the reverse
/// direction - the writer only ever narrows - so this is `#[cfg(test)]`.
#[cfg(test)]
fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign: f64 = if bits & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exp = ((bits >> 10) & 0x1f) as i32;
    let mant = (bits & 0x3ff) as f64;
    let value = if exp == 0 {
        if mant == 0.0 { 0.0 } else { sign * mant * 2f64.powi(-24) }
    } else if exp == 31 {
        if mant == 0.0 { sign * f64::INFINITY } else { f64::NAN }
    } else {
        sign * (1.0 + mant / 1024.0) * 2f64.powi(exp - 15)
    };
    value as f32
}

// ============================================================================
//  Running statistics, aggregated bottom-up while the tree is built.
// ============================================================================

#[derive(Clone, Copy)]
struct Stats {
    count: u64,
    sum: f64,
    sumsq: f64,
    min: f64,
    max: f64,
}

impl Stats {
    fn one(v: f64) -> Self {
        Self { count: 1, sum: v, sumsq: v * v, min: v, max: v }
    }
    fn add(&mut self, v: f64) {
        self.count += 1;
        self.sum += v;
        self.sumsq += v * v;
        if v < self.min {
            self.min = v;
        }
        if v > self.max {
            self.max = v;
        }
    }
    fn merge(&mut self, o: Stats) {
        if o.count == 0 {
            return;
        }
        if self.count == 0 {
            *self = o;
            return;
        }
        self.count += o.count;
        self.sum += o.sum;
        self.sumsq += o.sumsq;
        self.min = self.min.min(o.min);
        self.max = self.max.max(o.max);
    }
    fn empty() -> Self {
        Self { count: 0, sum: 0.0, sumsq: 0.0, min: f64::INFINITY, max: f64::NEG_INFINITY }
    }
    /// `(min, max, average, stddev)`, average/stddev as `f32` (`FloatType`
    /// is always `float` here per `FloatTraits`).
    fn finalize(&self) -> (f64, f64, f32, f32) {
        if self.count == 0 {
            return (0.0, 0.0, 0.0, 0.0);
        }
        let n = self.count as f64;
        let avg = self.sum / n;
        let var = (self.sumsq / n - avg * avg).max(0.0);
        (self.min, self.max, avg as f32, var.sqrt() as f32)
    }
}

// ============================================================================
//  Writer
// ============================================================================

fn append_segment(
    out: &mut Vec<u8>,
    grid: &UniformGrid,
    name: &str,
    values: &[Scalar],
    precision: Precision,
) -> Result<()> {
    if name.as_bytes().len() >= 256 {
        return Err(Error::Field {
            field: name.to_string(),
            msg: "grid name is 256 bytes or longer; nvdb writer does not use the \
                  long-grid-name blind-data path"
                .to_string(),
        });
    }

    let blob = build_blob(grid, name, values, precision);

    let n = grid.n() as u64;
    let (ox, oy, oz) = (grid.origin.x as f64, grid.origin.y as f64, grid.origin.z as f64);
    let (dx, dy, dz) = (grid.spacing.x as f64, grid.spacing.y as f64, grid.spacing.z as f64);
    let world_min = [ox - 0.5 * dx, oy - 0.5 * dy, oz - 0.5 * dz];
    let world_max = [
        ox + dx * (grid.nx as f64 - 0.5),
        oy + dy * (grid.ny as f64 - 0.5),
        oz + dz * (grid.nz as f64 - 0.5),
    ];
    let (n_leaf, n_lower, n_upper) = node_counts(grid);
    let node_count = [
        (n_leaf[0] * n_leaf[1] * n_leaf[2]) as u32,
        (n_lower[0] * n_lower[1] * n_lower[2]) as u32,
        (n_upper[0] * n_upper[1] * n_upper[2]) as u32,
        1,
    ];

    // ---- FileHeader (16 B) -------------------------------------------------
    let mut header = [0u8; FILE_HEADER_SIZE];
    put_u64(&mut header, 0, MAGIC_FILE);
    put_u32(&mut header, 8, VERSION);
    put_u16(&mut header, 12, 1); // gridCount
    put_u16(&mut header, 14, CODEC_NONE);
    out.extend_from_slice(&header);

    // ---- FileMetaData (176 B) + name ---------------------------------------
    let mut meta = [0u8; FILE_META_SIZE];
    put_u64(&mut meta, 0, blob.len() as u64); // gridSize
    put_u64(&mut meta, 8, blob.len() as u64); // fileSize
    put_u64(&mut meta, 16, string_hash(name)); // nameKey
    put_u64(&mut meta, 24, n); // voxelCount
    let grid_type = match precision {
        Precision::F32 => GRID_TYPE_FLOAT,
        Precision::F16 => GRID_TYPE_HALF,
    };
    put_u32(&mut meta, 32, grid_type);
    put_u32(&mut meta, 36, GRID_CLASS_FOG_VOLUME);
    put_f64(&mut meta, 40, world_min[0]);
    put_f64(&mut meta, 48, world_min[1]);
    put_f64(&mut meta, 56, world_min[2]);
    put_f64(&mut meta, 64, world_max[0]);
    put_f64(&mut meta, 72, world_max[1]);
    put_f64(&mut meta, 80, world_max[2]);
    put_i32(&mut meta, 88, 0);
    put_i32(&mut meta, 92, 0);
    put_i32(&mut meta, 96, 0);
    put_i32(&mut meta, 100, grid.nx as i32 - 1);
    put_i32(&mut meta, 104, grid.ny as i32 - 1);
    put_i32(&mut meta, 108, grid.nz as i32 - 1);
    put_f64(&mut meta, 112, dx);
    put_f64(&mut meta, 120, dy);
    put_f64(&mut meta, 128, dz);
    let name_size = name.as_bytes().len() as u32 + 1;
    put_u32(&mut meta, 136, name_size);
    put_u32(&mut meta, 140, node_count[0]);
    put_u32(&mut meta, 144, node_count[1]);
    put_u32(&mut meta, 148, node_count[2]);
    put_u32(&mut meta, 152, node_count[3]);
    put_u32(&mut meta, 156, 0); // tileCount[0]
    put_u32(&mut meta, 160, 0); // tileCount[1]
    put_u32(&mut meta, 164, 0); // tileCount[2]
    put_u16(&mut meta, 168, CODEC_NONE);
    put_u16(&mut meta, 170, 0); // blindDataCount
    put_u32(&mut meta, 172, VERSION);
    out.extend_from_slice(&meta);
    out.extend_from_slice(name.as_bytes());
    out.push(0); // NUL terminator

    // ---- the grid blob itself -----------------------------------------------
    out.extend_from_slice(&blob);
    Ok(())
}

/// FNV-1a, 64-bit - `FileMetaData::nameKey` is only ever used by readers as
/// a fast pre-filter before comparing the actual name string, so any stable
/// hash is a legal choice; FNV-1a is public-domain-grade generic algorithm,
/// not derived from any specific implementation.
fn string_hash(s: &str) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in s.as_bytes() {
        h ^= *b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

fn node_counts(grid: &UniformGrid) -> ([usize; 3], [usize; 3], [usize; 3]) {
    let n_leaf = [
        ceil_div(grid.nx, LEAF_DIM),
        ceil_div(grid.ny, LEAF_DIM),
        ceil_div(grid.nz, LEAF_DIM),
    ];
    let n_lower = [
        ceil_div(grid.nx, LOWER_DIM),
        ceil_div(grid.ny, LOWER_DIM),
        ceil_div(grid.nz, LOWER_DIM),
    ];
    let n_upper = [
        ceil_div(grid.nx, UPPER_DIM),
        ceil_div(grid.ny, UPPER_DIM),
        ceil_div(grid.nz, UPPER_DIM),
    ];
    (n_leaf, n_lower, n_upper)
}

/// Everything the recursive node builders need, gathered so their
/// signatures stay readable.
struct Ctx<'a> {
    grid: &'a UniformGrid,
    values: &'a [Scalar],
    precision: Precision,
    leaf: LeafLayout,
    lower: InternalLayout,
    upper: InternalLayout,
    n_leaf: [usize; 3],
    n_lower: [usize; 3],
    leaf_block_off: usize,
    lower_block_off: usize,
}

fn build_blob(grid: &UniformGrid, name: &str, values: &[Scalar], precision: Precision) -> Vec<u8> {
    let w = precision.width();
    let (n_leaf, n_lower, n_upper) = node_counts(grid);
    let n_leaf_total = n_leaf[0] * n_leaf[1] * n_leaf[2];
    let n_lower_total = n_lower[0] * n_lower[1] * n_lower[2];
    let n_upper_total = n_upper[0] * n_upper[1] * n_upper[2];

    let leaf_lay = leaf_layout(w);
    let lower_lay = internal_layout(w, LOWER_LOG2DIM);
    let upper_lay = internal_layout(w, UPPER_LOG2DIM);
    let root_lay = root_layout(w);
    let tile_lay = root_tile_layout(w);

    let tree_off = GRID_DATA_SIZE;
    let root_off = tree_off + TREE_DATA_SIZE;
    let root_tiles_off = root_off + root_lay.size;
    let upper_block_off = root_tiles_off + n_upper_total * tile_lay.size;
    let lower_block_off = upper_block_off + n_upper_total * upper_lay.size;
    let leaf_block_off = lower_block_off + n_lower_total * lower_lay.size;
    let total = leaf_block_off + n_leaf_total * leaf_lay.size;

    let mut blob = vec![0u8; total];

    // ---- GridData (672 B) ----------------------------------------------
    put_u64(&mut blob, 0, MAGIC_GRID);
    put_u64(&mut blob, 8, CHECKSUM_DISABLED);
    put_u32(&mut blob, 16, VERSION);
    put_u32(&mut blob, 20, GRID_FLAGS);
    put_u32(&mut blob, 24, 0); // mGridIndex
    put_u32(&mut blob, 28, 1); // mGridCount
    put_u64(&mut blob, 32, total as u64); // mGridSize
    let name_bytes = name.as_bytes();
    blob[40..40 + name_bytes.len()].copy_from_slice(name_bytes);
    // blob[40+name_bytes.len()] is already 0 (the NUL) from the zero-init.
    write_map(&mut blob, 296, grid);
    let (ox, oy, oz) = (grid.origin.x as f64, grid.origin.y as f64, grid.origin.z as f64);
    let (dx, dy, dz) = (grid.spacing.x as f64, grid.spacing.y as f64, grid.spacing.z as f64);
    put_f64(&mut blob, 560, ox - 0.5 * dx);
    put_f64(&mut blob, 568, oy - 0.5 * dy);
    put_f64(&mut blob, 576, oz - 0.5 * dz);
    put_f64(&mut blob, 584, ox + dx * (grid.nx as f64 - 0.5));
    put_f64(&mut blob, 592, oy + dy * (grid.ny as f64 - 0.5));
    put_f64(&mut blob, 600, oz + dz * (grid.nz as f64 - 0.5));
    put_f64(&mut blob, 608, dx);
    put_f64(&mut blob, 616, dy);
    put_f64(&mut blob, 624, dz);
    put_u32(&mut blob, 632, GRID_CLASS_FOG_VOLUME);
    put_u32(
        &mut blob,
        636,
        match precision {
            Precision::F32 => GRID_TYPE_FLOAT,
            Precision::F16 => GRID_TYPE_HALF,
        },
    );
    put_i64(&mut blob, 640, total as i64); // mBlindMetadataOffset (no blind data)
    put_u32(&mut blob, 648, 0); // mBlindMetadataCount
    put_u32(&mut blob, 652, 0); // mData0
    put_u64(&mut blob, 656, 0); // mData1
    put_u64(&mut blob, 664, 0); // mData2

    // ---- TreeData (64 B) --------------------------------------------------
    put_i64(&mut blob, tree_off, (leaf_block_off - tree_off) as i64);
    put_i64(&mut blob, tree_off + 8, (lower_block_off - tree_off) as i64);
    put_i64(&mut blob, tree_off + 16, (upper_block_off - tree_off) as i64);
    put_i64(&mut blob, tree_off + 24, (root_off - tree_off) as i64);
    put_u32(&mut blob, tree_off + 32, n_leaf_total as u32);
    put_u32(&mut blob, tree_off + 36, n_lower_total as u32);
    put_u32(&mut blob, tree_off + 40, n_upper_total as u32);
    put_u32(&mut blob, tree_off + 44, 0);
    put_u32(&mut blob, tree_off + 48, 0);
    put_u32(&mut blob, tree_off + 52, 0);
    put_u64(&mut blob, tree_off + 56, grid.n() as u64);

    // ---- Root, Upper, Lower, Leaf (bottom-up stats, top-down bytes) -------
    let ctx = Ctx {
        grid,
        values,
        precision,
        leaf: leaf_lay,
        lower: lower_lay,
        upper: upper_lay,
        n_leaf,
        n_lower,
        leaf_block_off,
        lower_block_off,
    };

    let mut root_stats = Stats::empty();
    for uz in 0..n_upper[2] {
        for uy in 0..n_upper[1] {
            for ux in 0..n_upper[0] {
                let upper_idx = ux + n_upper[0] * (uy + n_upper[1] * uz);
                let upper_off = upper_block_off + upper_idx * ctx.upper.size;
                let origin = [ux * UPPER_DIM, uy * UPPER_DIM, uz * UPPER_DIM];
                let stats = build_upper(&ctx, &mut blob, upper_off, origin);
                root_stats.merge(stats);

                let tile_off = root_tiles_off + upper_idx * tile_lay.size;
                put_u64(&mut blob, tile_off + tile_lay.key, coord_to_key(origin, UPPER_TOTAL));
                put_i64(&mut blob, tile_off + tile_lay.child, (upper_off as i64) - (root_off as i64));
                put_u32(&mut blob, tile_off + tile_lay.state, 0);
                put_value(&mut blob, tile_off + tile_lay.value, 0.0, precision);
            }
        }
    }

    put_i32(&mut blob, root_off + root_lay.bbox, 0);
    put_i32(&mut blob, root_off + root_lay.bbox + 4, 0);
    put_i32(&mut blob, root_off + root_lay.bbox + 8, 0);
    put_i32(&mut blob, root_off + root_lay.bbox + 12, grid.nx as i32 - 1);
    put_i32(&mut blob, root_off + root_lay.bbox + 16, grid.ny as i32 - 1);
    put_i32(&mut blob, root_off + root_lay.bbox + 20, grid.nz as i32 - 1);
    put_u32(&mut blob, root_off + root_lay.table_size, n_upper_total as u32);
    put_value(&mut blob, root_off + root_lay.background, 0.0, precision);
    let (mn, mx, avg, sd) = root_stats.finalize();
    put_value(&mut blob, root_off + root_lay.minimum, mn, precision);
    put_value(&mut blob, root_off + root_lay.maximum, mx, precision);
    put_f32(&mut blob, root_off + root_lay.average, avg);
    put_f32(&mut blob, root_off + root_lay.stddev, sd);

    blob
}

/// `Map`: a uniform scale-plus-translation affine transform (see the module
/// doc for the index<->world convention). `map_off` is 296, the fixed
/// `GridData::mMap` offset.
fn write_map(blob: &mut [u8], map_off: usize, grid: &UniformGrid) {
    let (dx, dy, dz) = (grid.spacing.x as f64, grid.spacing.y as f64, grid.spacing.z as f64);
    let (ox, oy, oz) = (grid.origin.x as f64, grid.origin.y as f64, grid.origin.z as f64);
    let mat_f = [dx as f32, 0.0, 0.0, 0.0, dy as f32, 0.0, 0.0, 0.0, dz as f32];
    let inv_f = [
        1.0 / dx as f32, 0.0, 0.0, 0.0, 1.0 / dy as f32, 0.0, 0.0, 0.0, 1.0 / dz as f32,
    ];
    let vec_f = [ox as f32, oy as f32, oz as f32];
    let mat_d = [dx, 0.0, 0.0, 0.0, dy, 0.0, 0.0, 0.0, dz];
    let inv_d = [1.0 / dx, 0.0, 0.0, 0.0, 1.0 / dy, 0.0, 0.0, 0.0, 1.0 / dz];
    let vec_d = [ox, oy, oz];

    let mut off = map_off;
    for v in mat_f {
        put_f32(blob, off, v);
        off += 4;
    }
    for v in inv_f {
        put_f32(blob, off, v);
        off += 4;
    }
    for v in vec_f {
        put_f32(blob, off, v);
        off += 4;
    }
    put_f32(blob, off, 1.0);
    off += 4;
    for v in mat_d {
        put_f64(blob, off, v);
        off += 8;
    }
    for v in inv_d {
        put_f64(blob, off, v);
        off += 8;
    }
    for v in vec_d {
        put_f64(blob, off, v);
        off += 8;
    }
    put_f64(blob, off, 1.0);
}

/// `RootData<ChildT>::CoordToKey` under `NANOVDB_USE_SINGLE_ROOT_KEY`: pack
/// `x`/`y`/`z`, each right-shifted by `total` bits, into the upper/middle/
/// lower 21 bits of a `uint64_t`.
fn coord_to_key(origin: [usize; 3], total: u32) -> u64 {
    let x = (origin[0] as u64 >> total) & 0x1f_ffff;
    let y = (origin[1] as u64 >> total) & 0x1f_ffff;
    let z = origin[2] as u64 >> total;
    (x << 42) | (y << 21) | z
}

fn build_upper(ctx: &Ctx, blob: &mut [u8], off: usize, origin: [usize; 3]) -> Stats {
    let lay = &ctx.upper;
    let mut stats = Stats::empty();
    for c in 0..32usize {
        let lz = origin[2] / LOWER_DIM + c;
        if lz >= ctx.n_lower[2] {
            continue;
        }
        for b in 0..32usize {
            let ly = origin[1] / LOWER_DIM + b;
            if ly >= ctx.n_lower[1] {
                continue;
            }
            for a in 0..32usize {
                let lx = origin[0] / LOWER_DIM + a;
                if lx >= ctx.n_lower[0] {
                    continue;
                }
                let lower_idx = lx + ctx.n_lower[0] * (ly + ctx.n_lower[1] * lz);
                let lower_off = ctx.lower_block_off + lower_idx * ctx.lower.size;
                let lower_origin = [lx * LOWER_DIM, ly * LOWER_DIM, lz * LOWER_DIM];
                let child_stats = build_lower(ctx, blob, lower_off, lower_origin);
                stats.merge(child_stats);

                let n = ((a as u32) << 10) | ((b as u32) << 5) | (c as u32);
                set_mask_bit(blob, off + lay.child_mask, n);
                let table_off = off + lay.table + (n as usize) * lay.entry_size;
                put_i64(blob, table_off, (lower_off as i64) - (off as i64));
            }
        }
    }
    write_internal_node(blob, off, lay, ctx.precision, origin, UPPER_DIM, ctx.grid, &stats);
    stats
}

fn build_lower(ctx: &Ctx, blob: &mut [u8], off: usize, origin: [usize; 3]) -> Stats {
    let lay = &ctx.lower;
    let mut stats = Stats::empty();
    for f in 0..16usize {
        let fz = origin[2] / LEAF_DIM + f;
        if fz >= ctx.n_leaf[2] {
            continue;
        }
        for e in 0..16usize {
            let fy = origin[1] / LEAF_DIM + e;
            if fy >= ctx.n_leaf[1] {
                continue;
            }
            for d in 0..16usize {
                let fx = origin[0] / LEAF_DIM + d;
                if fx >= ctx.n_leaf[0] {
                    continue;
                }
                let leaf_idx = fx + ctx.n_leaf[0] * (fy + ctx.n_leaf[1] * fz);
                let leaf_off = ctx.leaf_block_off + leaf_idx * ctx.leaf.size;
                let leaf_origin = [fx * LEAF_DIM, fy * LEAF_DIM, fz * LEAF_DIM];
                let leaf_stats =
                    build_leaf(ctx, blob, leaf_off, leaf_origin);
                stats.merge(leaf_stats);

                let n = ((d as u32) << 8) | ((e as u32) << 4) | (f as u32);
                set_mask_bit(blob, off + lay.child_mask, n);
                let table_off = off + lay.table + (n as usize) * lay.entry_size;
                put_i64(blob, table_off, (leaf_off as i64) - (off as i64));
            }
        }
    }
    write_internal_node(blob, off, lay, ctx.precision, origin, LOWER_DIM, ctx.grid, &stats);
    stats
}

fn write_internal_node(
    blob: &mut [u8],
    off: usize,
    lay: &InternalLayout,
    precision: Precision,
    origin: [usize; 3],
    span: usize,
    grid: &UniformGrid,
    stats: &Stats,
) {
    put_i32(blob, off + lay.bbox, origin[0] as i32);
    put_i32(blob, off + lay.bbox + 4, origin[1] as i32);
    put_i32(blob, off + lay.bbox + 8, origin[2] as i32);
    put_i32(blob, off + lay.bbox + 12, (origin[0] + span - 1).min(grid.nx - 1) as i32);
    put_i32(blob, off + lay.bbox + 16, (origin[1] + span - 1).min(grid.ny - 1) as i32);
    put_i32(blob, off + lay.bbox + 20, (origin[2] + span - 1).min(grid.nz - 1) as i32);
    put_u64(blob, off + lay.flags, 0);
    // mValueMask stays all-zero: this writer never places a constant-value
    // tile at Lower/Upper level (a dense box always subdivides to Leaf).
    let (mn, mx, avg, sd) = stats.finalize();
    put_value(blob, off + lay.minimum, mn, precision);
    put_value(blob, off + lay.maximum, mx, precision);
    put_f32(blob, off + lay.average, avg);
    put_f32(blob, off + lay.stddev, sd);
}

fn build_leaf(ctx: &Ctx, blob: &mut [u8], off: usize, origin: [usize; 3]) -> Stats {
    let lay = &ctx.leaf;
    let grid = ctx.grid;
    put_i32(blob, off + lay.bbox_min, origin[0] as i32);
    put_i32(blob, off + lay.bbox_min + 4, origin[1] as i32);
    put_i32(blob, off + lay.bbox_min + 8, origin[2] as i32);

    let max_x = (origin[0] + LEAF_DIM - 1).min(grid.nx - 1);
    let max_y = (origin[1] + LEAF_DIM - 1).min(grid.ny - 1);
    let max_z = (origin[2] + LEAF_DIM - 1).min(grid.nz - 1);
    blob[off + lay.bbox_dif] = (max_x - origin[0]) as u8;
    blob[off + lay.bbox_dif + 1] = (max_y - origin[1]) as u8;
    blob[off + lay.bbox_dif + 2] = (max_z - origin[2]) as u8;
    blob[off + lay.flags] = 0x12; // hasBBox (bit1) | hasStats (bit4)

    let mut stats = Stats::empty();
    for lx in 0..LEAF_DIM {
        let gx = origin[0] + lx;
        if gx >= grid.nx {
            continue;
        }
        for ly in 0..LEAF_DIM {
            let gy = origin[1] + ly;
            if gy >= grid.ny {
                continue;
            }
            for lz in 0..LEAF_DIM {
                let gz = origin[2] + lz;
                if gz >= grid.nz {
                    continue;
                }
                let n = ((lx as u32) << 6) | ((ly as u32) << 3) | (lz as u32);
                set_mask_bit(blob, off + lay.value_mask, n);
                let v = ctx.values[grid.idx(gx, gy, gz)] as f64;
                put_value(blob, off + lay.values + (n as usize) * lay.value_width, v, ctx.precision);
                if stats.count == 0 {
                    stats = Stats::one(v);
                } else {
                    stats.add(v);
                }
            }
        }
    }

    let (mn, mx, avg, sd) = stats.finalize();
    put_value(blob, off + lay.minimum, mn, ctx.precision);
    put_value(blob, off + lay.maximum, mx, ctx.precision);
    put_f32(blob, off + lay.average, avg);
    put_f32(blob, off + lay.stddev, sd);
    stats
}

// ============================================================================
//  Internal reader - for round-trip tests only. Genuinely walks the tree
//  (root tiles -> child masks -> child masks -> value masks) rather than
//  recomputing existence from the same formulas the writer used, so it
//  actually exercises what got written to the mask/child/key bytes.
// ============================================================================

#[cfg(test)]
mod reader {
    use super::*;

    pub struct ReadSegment {
        pub name: String,
        pub grid_type: u32,
        pub grid_class: u32,
        pub voxel_count: u64,
        pub node_count: [u32; 4],
        pub index_bbox: [i32; 6],
        pub world_bbox: [f64; 6],
        pub voxel_size: [f64; 3],
        pub nx: usize,
        pub ny: usize,
        pub nz: usize,
        pub values: Vec<f64>, // dense, idx = i + nx*(j + ny*k); NAN where inactive
    }

    /// Read every segment in an nvdb file written by [`super::write`].
    pub fn read_all(bytes: &[u8]) -> Vec<ReadSegment> {
        let mut out = Vec::new();
        let mut pos = 0usize;
        while pos + FILE_HEADER_SIZE <= bytes.len() {
            let magic = get_u64(bytes, pos);
            assert_eq!(magic, MAGIC_FILE, "not a NanoVDB file segment at byte {pos}");
            let grid_count = get_u16(bytes, pos + 12);
            assert_eq!(grid_count, 1, "reader only handles one grid per segment");
            let mut p = pos + FILE_HEADER_SIZE;

            let grid_size = get_u64(bytes, p);
            let grid_type = get_u32(bytes, p + 32);
            let grid_class = get_u32(bytes, p + 36);
            let world_bbox = [
                get_f64(bytes, p + 40), get_f64(bytes, p + 48), get_f64(bytes, p + 56),
                get_f64(bytes, p + 64), get_f64(bytes, p + 72), get_f64(bytes, p + 80),
            ];
            let index_bbox = [
                get_i32(bytes, p + 88), get_i32(bytes, p + 92), get_i32(bytes, p + 96),
                get_i32(bytes, p + 100), get_i32(bytes, p + 104), get_i32(bytes, p + 108),
            ];
            let voxel_size = [get_f64(bytes, p + 112), get_f64(bytes, p + 120), get_f64(bytes, p + 128)];
            let name_size = get_u32(bytes, p + 136) as usize;
            let node_count = [
                get_u32(bytes, p + 140), get_u32(bytes, p + 144),
                get_u32(bytes, p + 148), get_u32(bytes, p + 152),
            ];
            p += FILE_META_SIZE;
            let name = String::from_utf8_lossy(&bytes[p..p + name_size - 1]).into_owned();
            p += name_size;

            let grid_off = p;
            let tree_off = grid_off + GRID_DATA_SIZE;
            let voxel_count = get_u64(bytes, tree_off + 56); // TreeData::mVoxelCount
            let precision = match grid_type {
                GRID_TYPE_FLOAT => Precision::F32,
                GRID_TYPE_HALF => Precision::F16,
                other => panic!("unexpected grid type {other}"),
            };

            let nx = (index_bbox[3] - index_bbox[0] + 1) as usize;
            let ny = (index_bbox[4] - index_bbox[1] + 1) as usize;
            let nz = (index_bbox[5] - index_bbox[2] + 1) as usize;
            let mut values = vec![f64::NAN; nx * ny * nz];

            let w = precision.width();
            let leaf_lay = leaf_layout(w);
            let lower_lay = internal_layout(w, LOWER_LOG2DIM);
            let upper_lay = internal_layout(w, UPPER_LOG2DIM);
            let root_lay = root_layout(w);
            let tile_lay = root_tile_layout(w);

            let root_off = tree_off + get_i64(bytes, tree_off + 24) as usize;
            let table_size = get_u32(bytes, root_off + root_lay.table_size) as usize;

            for t in 0..table_size {
                let tile_off = root_off + root_lay.size + t * tile_lay.size;
                let key = get_u64(bytes, tile_off + tile_lay.key);
                let child = get_i64(bytes, tile_off + tile_lay.child);
                let upper_off = (root_off as i64 + child) as usize;
                let upper_origin = key_to_coord(key, UPPER_TOTAL);
                read_internal(
                    bytes, upper_off, &upper_lay, UPPER_LOG2DIM, upper_origin, UPPER_DIM / 32,
                    precision, index_bbox, nx, ny, &lower_lay, &leaf_lay, &mut values,
                    NodeKind::Upper,
                );
            }

            out.push(ReadSegment {
                name, grid_type, grid_class, voxel_count,
                node_count, index_bbox, world_bbox, voxel_size, nx, ny, nz, values,
            });
            pos = grid_off + grid_size as usize;
        }
        out
    }

    enum NodeKind {
        Upper,
        Lower,
    }

    #[allow(clippy::too_many_arguments)]
    fn read_internal(
        bytes: &[u8],
        off: usize,
        lay: &InternalLayout,
        log2dim: u32,
        origin: [i32; 3],
        child_span: usize,
        precision: Precision,
        index_bbox: [i32; 6],
        nx: usize,
        ny: usize,
        lower_lay: &InternalLayout,
        leaf_lay: &LeafLayout,
        values: &mut [f64],
        kind: NodeKind,
    ) {
        let fan = 1u32 << log2dim;
        for n in 0..lay.slots as u32 {
            if !mask_bit(bytes, off + lay.child_mask, n) {
                continue;
            }
            let a = n >> (2 * log2dim);
            let b = (n >> log2dim) & (fan - 1);
            let c = n & (fan - 1);
            let child_origin = [
                origin[0] + (a as i32) * child_span as i32,
                origin[1] + (b as i32) * child_span as i32,
                origin[2] + (c as i32) * child_span as i32,
            ];
            let table_off = off + lay.table + (n as usize) * lay.entry_size;
            let child_rel = get_i64(bytes, table_off);
            let child_off = (off as i64 + child_rel) as usize;
            match kind {
                NodeKind::Upper => {
                    read_internal(
                        bytes, child_off, lower_lay, LOWER_LOG2DIM, child_origin, LEAF_DIM,
                        precision, index_bbox, nx, ny, lower_lay, leaf_lay, values,
                        NodeKind::Lower,
                    );
                }
                NodeKind::Lower => {
                    read_leaf(bytes, child_off, leaf_lay, child_origin, precision, index_bbox, nx, ny, values);
                }
            }
        }
    }

    fn read_leaf(
        bytes: &[u8],
        off: usize,
        lay: &LeafLayout,
        origin: [i32; 3],
        precision: Precision,
        index_bbox: [i32; 6],
        nx: usize,
        ny: usize,
        values: &mut [f64],
    ) {
        for lx in 0..LEAF_DIM as u32 {
            for ly in 0..LEAF_DIM as u32 {
                for lz in 0..LEAF_DIM as u32 {
                    let n = (lx << 6) | (ly << 3) | lz;
                    if !mask_bit(bytes, off + lay.value_mask, n) {
                        continue;
                    }
                    let gx = origin[0] + lx as i32;
                    let gy = origin[1] + ly as i32;
                    let gz = origin[2] + lz as i32;
                    let i = (gx - index_bbox[0]) as usize;
                    let j = (gy - index_bbox[1]) as usize;
                    let k = (gz - index_bbox[2]) as usize;
                    let v = get_value(bytes, off + lay.values + (n as usize) * lay.value_width, precision);
                    values[i + nx * (j + ny * k)] = v;
                }
            }
        }
    }

    /// Inverse of `coord_to_key`.
    fn key_to_coord(key: u64, total: u32) -> [i32; 3] {
        let mask = 0x1f_ffffu64;
        let x = ((key >> 42) & mask) << total;
        let y = ((key >> 21) & mask) << total;
        let z = (key & mask) << total;
        [x as i32, y as i32, z as i32]
    }
}

// ============================================================================
//  Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::reader::read_all;
    use super::*;
    use std::path::PathBuf;

    fn scratch(name: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!("ofgpu_nvdb_test_{name}_{}.nvdb", std::process::id()));
        p
    }

    // ---- layout sanity ------------------------------------------------

    #[test]
    fn layout_sizes_match_the_header() {
        assert_eq!(root_layout(4).size, 64);
        assert_eq!(root_layout(2).size, 64);
        assert_eq!(root_tile_layout(4).size, 32);
        assert_eq!(root_tile_layout(2).size, 32);
        assert_eq!(internal_layout(4, LOWER_LOG2DIM).size, 33_856);
        assert_eq!(internal_layout(2, LOWER_LOG2DIM).size, 33_856);
        assert_eq!(internal_layout(4, UPPER_LOG2DIM).size, 270_400);
        assert_eq!(internal_layout(2, UPPER_LOG2DIM).size, 270_400);
        assert_eq!(leaf_layout(4).size, 2144);
        assert_eq!(leaf_layout(2).size, 1120);
        // Structural facts stated in the task brief / NanoVDB.h comments.
        assert_eq!(FILE_HEADER_SIZE, 16);
        assert_eq!(FILE_META_SIZE, 176);
        assert_eq!(GRID_DATA_SIZE, 672);
        assert_eq!(TREE_DATA_SIZE, 64);
        assert_eq!(MAGIC_FILE, 0x324244566f6e614e);
        for size in [
            root_layout(4).size, root_tile_layout(4).size,
            internal_layout(4, LOWER_LOG2DIM).size, internal_layout(4, UPPER_LOG2DIM).size,
            leaf_layout(4).size, leaf_layout(2).size,
        ] {
            assert_eq!(size % 32, 0, "every NANOVDB_ALIGN'd struct must be a multiple of 32 bytes");
        }
    }

    #[test]
    fn node_counts_are_the_analytic_ceiling_division() {
        let grid = UniformGrid { nx: 20, ny: 5, nz: 33, origin: Vec3::ZERO, spacing: Vec3::new(1.0, 1.0, 1.0) };
        let (leaf, lower, upper) = node_counts(&grid);
        assert_eq!(leaf, [ceil_div(20, 8), ceil_div(5, 8), ceil_div(33, 8)]);
        assert_eq!(leaf, [3, 1, 5]);
        assert_eq!(lower, [ceil_div(20, 128), ceil_div(5, 128), ceil_div(33, 128)]);
        assert_eq!(lower, [1, 1, 1]);
        assert_eq!(upper, [1, 1, 1]);
    }

    // ---- half-float conversion -----------------------------------------

    #[test]
    fn half_conversion_round_trips_exact_values() {
        // Every one of these is exactly representable in binary16: integers
        // and halves well inside its 10-bit mantissa / 5-bit exponent range.
        // (-293.15 and 1173.15, tempting since they are §9's plume/ambient
        // temperatures, are *not* exactly representable in binary16 - their
        // fractional part needs more than 10 mantissa bits at that
        // magnitude - so they belong in a rounding test, not an exactness
        // one; see `half_conversion_rounds_a_temperature_to_its_nearest_half`.)
        for &v in &[0.0f32, 1.0, -1.0, 2.0, 0.5, -0.5, 100.0, -293.0, 1173.0, 65504.0] {
            let bits = f32_to_f16_bits(v);
            let back = f16_bits_to_f32(bits);
            assert_eq!(back, v, "round trip of {v}");
        }
    }

    #[test]
    fn half_conversion_rounds_a_temperature_to_its_nearest_half() {
        // 1173.15 K sits between the two representable halves 1173.0 and
        // 1174.0 (ULP = 1 in that range); this pins that it lands on the
        // nearer one rather than silently drifting.
        let half = f16_bits_to_f32(f32_to_f16_bits(1173.15));
        assert_eq!(half, 1173.0);
    }

    #[test]
    fn half_conversion_known_bit_patterns() {
        assert_eq!(f32_to_f16_bits(1.0), 0x3c00);
        assert_eq!(f32_to_f16_bits(-1.0), 0xbc00);
        assert_eq!(f32_to_f16_bits(2.0), 0x4000);
        assert_eq!(f32_to_f16_bits(0.0), 0x0000);
        assert_eq!(f32_to_f16_bits(-0.0), 0x8000);
        assert_eq!(f32_to_f16_bits(f32::INFINITY), 0x7c00);
        assert_eq!(f32_to_f16_bits(f32::NEG_INFINITY), 0xfc00);
        assert!(f16_bits_to_f32(f32_to_f16_bits(f32::NAN)).is_nan());
    }

    #[test]
    fn half_conversion_rounds_to_nearest_even() {
        // Half's mantissa step at exponent 0 (values in [1,2)) is 2^-10; a
        // tie sits exactly half a step past a given mantissa, i.e. at
        // `1 + m*2^-10 + 2^-11`. m=0 is already even, so its tie rounds
        // down (stays 0); m=1 is odd, so its tie rounds up to the even
        // neighbour, 2.
        let step = 2f32.powi(-10);
        let half_step = 2f32.powi(-11);
        let tie_at_even_mantissa = 1.0f32 + 0.0 * step + half_step;
        let tie_at_odd_mantissa = 1.0f32 + 1.0 * step + half_step;
        let b_even = f32_to_f16_bits(tie_at_even_mantissa);
        let b_odd = f32_to_f16_bits(tie_at_odd_mantissa);
        assert_eq!(b_even & 0x3ff, 0, "tie at mantissa 0 stays at the even mantissa (0)");
        assert_eq!(b_odd & 0x3ff, 2, "tie at mantissa 1 rounds up to the even mantissa (2)");
    }

    #[test]
    fn half_conversion_smallest_subnormal_and_underflow() {
        let smallest = 2f32.powi(-24);
        assert_eq!(f32_to_f16_bits(smallest), 0x0001);
        assert_eq!(f16_bits_to_f32(0x0001), smallest);
        // Half of the smallest subnormal rounds to even (zero).
        assert_eq!(f32_to_f16_bits(2f32.powi(-25)), 0x0000);
        // Just enough to round up to the smallest subnormal.
        assert_eq!(f32_to_f16_bits(2f32.powi(-25) * 1.5), 0x0001);
    }

    #[test]
    fn half_conversion_overflow_to_infinity() {
        assert_eq!(f32_to_f16_bits(70000.0), 0x7c00);
        assert_eq!(f32_to_f16_bits(65520.0), 0x7c00); // rounds up past the largest finite half
    }

    // ---- end-to-end write + internal read --------------------------------

    fn make_grid(nx: usize, ny: usize, nz: usize) -> UniformGrid {
        UniformGrid {
            nx, ny, nz,
            origin: Vec3::new(1.5, -2.0, 0.25),
            spacing: Vec3::new(0.1, 0.2, 0.05),
        }
    }

    fn ramp_field(grid: &UniformGrid) -> Vec<Scalar> {
        let mut v = vec![0.0 as Scalar; grid.n()];
        for k in 0..grid.nz {
            for j in 0..grid.ny {
                for i in 0..grid.nx {
                    let val = (i as Scalar) + 100.0 * (j as Scalar) + 10_000.0 * (k as Scalar) + 0.5;
                    v[grid.idx(i, j, k)] = val;
                }
            }
        }
        v
    }

    #[test]
    fn round_trips_every_voxel_bit_exact_at_fp32_small_grid() {
        let grid = make_grid(5, 4, 3);
        let field = ramp_field(&grid);
        let path = scratch("small_fp32");
        write(
            &path,
            &grid,
            &[OutputField::scalar("T", &field)],
            Precision::F32,
        )
        .expect("write");

        let bytes = std::fs::read(&path).expect("read back");
        let segments = read_all(&bytes);
        assert_eq!(segments.len(), 1);
        let seg = &segments[0];
        assert_eq!(seg.name, "T");
        assert_eq!((seg.nx, seg.ny, seg.nz), (grid.nx, grid.ny, grid.nz));
        for k in 0..grid.nz {
            for j in 0..grid.ny {
                for i in 0..grid.nx {
                    let want = field[grid.idx(i, j, k)] as f64;
                    let got = seg.values[i + grid.nx * (j + grid.ny * k)];
                    assert_eq!(got, want, "voxel ({i},{j},{k})");
                }
            }
        }
        let _ = std::fs::remove_file(&path);
    }

    /// A grid large enough to force more than one node at every level
    /// (multiple leaves per axis, multiple lower nodes per axis), still
    /// small enough to run fast, exercising boundary leaves/lowers whose
    /// local extent is clipped by the domain.
    #[test]
    fn round_trips_every_voxel_bit_exact_at_fp32_multi_node_grid() {
        let grid = make_grid(37, 21, 17); // spans several 8-leaves and crosses a 16-leaf lower boundary
        let field = ramp_field(&grid);
        let path = scratch("multi_fp32");
        write(&path, &grid, &[OutputField::scalar("rho", &field)], Precision::F32).expect("write");

        let bytes = std::fs::read(&path).expect("read back");
        let segments = read_all(&bytes);
        let seg = &segments[0];
        let (n_leaf, n_lower, n_upper) = node_counts(&grid);
        assert_eq!(seg.node_count[0], (n_leaf[0] * n_leaf[1] * n_leaf[2]) as u32);
        assert_eq!(seg.node_count[1], (n_lower[0] * n_lower[1] * n_lower[2]) as u32);
        assert_eq!(seg.node_count[2], (n_upper[0] * n_upper[1] * n_upper[2]) as u32);
        assert_eq!(seg.node_count[3], 1);

        for k in 0..grid.nz {
            for j in 0..grid.ny {
                for i in 0..grid.nx {
                    let want = field[grid.idx(i, j, k)] as f64;
                    let got = seg.values[i + grid.nx * (j + grid.ny * k)];
                    assert_eq!(got, want, "voxel ({i},{j},{k})");
                }
            }
        }
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn fp16_round_trips_through_this_writers_own_reader() {
        let grid = make_grid(6, 5, 4);
        let mut field = vec![0.0 as Scalar; grid.n()];
        for (i, v) in field.iter_mut().enumerate() {
            *v = (293.15 + i as Scalar * 0.37) as Scalar;
        }
        let path = scratch("fp16");
        write(&path, &grid, &[OutputField::scalar("T", &field)], Precision::F16).expect("write");

        let bytes = std::fs::read(&path).expect("read back");
        let seg = &read_all(&bytes)[0];
        for idx in 0..grid.n() {
            let want = f16_bits_to_f32(f32_to_f16_bits(field[idx] as f32)) as f64;
            assert_eq!(seg.values[idx], want, "voxel {idx}");
        }
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn vector_field_becomes_four_component_segments() {
        let grid = make_grid(4, 3, 2);
        let mut u = vec![Vec3::ZERO; grid.n()];
        for k in 0..grid.nz {
            for j in 0..grid.ny {
                for i in 0..grid.nx {
                    u[grid.idx(i, j, k)] = Vec3::new(i as Scalar, j as Scalar, k as Scalar);
                }
            }
        }
        let path = scratch("vector");
        write(&path, &grid, &[OutputField::vector("U", &u)], Precision::F32).expect("write");

        let bytes = std::fs::read(&path).expect("read back");
        let segments = read_all(&bytes);
        let names: Vec<&str> = segments.iter().map(|s| s.name.as_str()).collect();
        assert_eq!(names, ["U.x", "U.y", "U.z", "U.mag"]);

        let seg_x = &segments[0];
        let seg_mag = &segments[3];
        for k in 0..grid.nz {
            for j in 0..grid.ny {
                for i in 0..grid.nx {
                    let idx = i + grid.nx * (j + grid.ny * k);
                    assert_eq!(seg_x.values[idx], i as f64);
                    // Precision::F32 narrows every value to f32 on the way
                    // into the file, so the expected magnitude must go
                    // through that same narrowing before comparison.
                    let want_mag = (u[grid.idx(i, j, k)].mag() as f32) as f64;
                    assert_eq!(seg_mag.values[idx], want_mag);
                }
            }
        }
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn world_bbox_index_bbox_and_voxel_size_are_correct() {
        let grid = make_grid(10, 8, 6);
        let field = ramp_field(&grid);
        let path = scratch("bbox");
        write(&path, &grid, &[OutputField::scalar("p", &field)], Precision::F32).expect("write");

        let bytes = std::fs::read(&path).expect("read back");
        let seg = &read_all(&bytes)[0];
        assert_eq!(seg.index_bbox, [0, 0, 0, 9, 7, 5]);
        assert_eq!(seg.voxel_size, [0.1f64, 0.2, 0.05]);
        assert_eq!(seg.grid_type, GRID_TYPE_FLOAT);
        assert_eq!(seg.grid_class, GRID_CLASS_FOG_VOLUME);
        assert_eq!(seg.voxel_count, grid.n() as u64);

        let (ox, oy, oz) = (1.5f64, -2.0, 0.25);
        let (dx, dy, dz) = (0.1f64, 0.2, 0.05);
        let want_min = [ox - 0.5 * dx, oy - 0.5 * dy, oz - 0.5 * dz];
        let want_max = [
            ox + dx * (10.0 - 0.5),
            oy + dy * (8.0 - 0.5),
            oz + dz * (6.0 - 0.5),
        ];
        for a in 0..3 {
            assert!((seg.world_bbox[a] - want_min[a]).abs() < 1e-12);
            assert!((seg.world_bbox[a + 3] - want_max[a]).abs() < 1e-12);
        }
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn rejects_a_field_length_that_does_not_match_the_grid() {
        let grid = make_grid(3, 3, 3);
        let field = vec![0.0 as Scalar; 5];
        let path = scratch("bad_len");
        let err = write(&path, &grid, &[OutputField::scalar("bad", &field)], Precision::F32).unwrap_err();
        assert!(err.to_string().contains("bad"));
    }

    #[test]
    fn rejects_empty_dims_and_empty_field_list() {
        let field = vec![1.0 as Scalar; 1];
        let empty_grid = UniformGrid { nx: 0, ny: 1, nz: 1, origin: Vec3::ZERO, spacing: Vec3::new(1.0, 1.0, 1.0) };
        assert!(write("x.nvdb", &empty_grid, &[OutputField::scalar("a", &field)], Precision::F32).is_err());

        let grid = make_grid(2, 2, 2);
        assert!(write("x.nvdb", &grid, &[], Precision::F32).is_err());
    }
}
