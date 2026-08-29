// meteor-cfd - Copyright (c) 2026 주식회사 메테오시뮬레이션 (Meteo Simulation Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

//! OpenVDB (.vdb) file writer, pure Rust, uncompressed. No formal spec
//! exists; structural validation only until externally verified.
//!
//! Written from the Apache-2.0 OpenVDB source (not a header-only spec like
//! NanoVDB - the `.vdb` format has no published byte-layout document, so the
//! implementation itself is the only authority), fetched from
//! `github.com/AcademySoftwareFoundation/openvdb` at the `v13.0.0` tag:
//! `openvdb/openvdb/io/Archive.cc` (`writeHeader`, `Archive::write`,
//! `writeGrid`, `setGridCompression`), `io/GridDescriptor.cc` (per-grid
//! header + stream-position framing), `util/Name.h` (`writeString`),
//! `Metadata.h`/`Metadata.cc` (the `MetaMap` entry format),
//! `math/Transform.cc` + `math/Maps.h` (`AffineMap`), `math/Mat.h` (`Mat4`
//! raw layout), `tree/RootNode.h`, `tree/InternalNode.h`, `tree/LeafNode.h`
//! (topology/buffer I/O), `util/NodeMasks.h` (bitmask raw layout),
//! `io/Compression.h` (`writeCompressedValues` under `COMPRESS_NONE`) and
//! `version.h.in` (magic/version constants). Every byte offset and field
//! order below is transcribed from those `.write`/`.writeTopology`/
//! `.writeBuffers` methods, cited inline; no other CFD or CAE code was
//! consulted, and this file is an independent writer built from that
//! format knowledge, not a translation of OpenVDB's node/tree classes.
//!
//! # Why this file writes real data instead of stopping at a stub
//!
//! SPEC-LIT's task brief budgets `.vdb` as a "budget-boxed attempt" with an
//! explicit escape hatch: ship the NanoVDB writer (`nvdb.rs`) alone if the
//! `.vdb` structural writer does not land in the session. It landed: the
//! format turned out to be fully pinned down by the source above, including
//! the one detail that would have been a silent-corruption trap - the
//! per-grid *compression* flags are not a metadata string (an earlier,
//! commented-out line in `Archive.cc` suggests that was tried and abandoned)
//! but a plain `uint32_t` written to the stream by `Archive::
//! setGridCompression`, immediately after the grid's three stream-position
//! placeholders and read back by `Archive::readGridCompression` before
//! anything else about the grid is parsed. Writing `0` (`COMPRESS_NONE`)
//! there is what makes every node's value buffer that follows a plain array
//! of raw values with no dependency on the reading application's own
//! default codec.
//!
//! # What is written
//!
//! One `.vdb` file (a single `io::Archive` "segment", `OPENVDB_FILE_VERSION
//! = 225`), no file-level metadata, one grid per scalar field - a
//! `FieldValues::Vector` field becomes four grids (`name.x`, `name.y`,
//! `name.z`, `name.mag`) for the same reason `nvdb.rs` does: a dense
//! `Vec3f` grid is the documented Omniverse/Blender split failure in
//! `docs/05-io-redesign.md` §8 Q4, and it applies to OpenVDB's own
//! `Vec3fGrid` exactly as much as to NanoVDB's. Every grid is
//! `Tree_float_5_4_3` (OpenVDB's own default `FloatTree` configuration -
//! Upper 32³, Lower 16³, Leaf 8³ fan-out, the same shape `nvdb.rs` builds),
//! `class = "fog volume"`, transformed by an `AffineMap` (a general 4x4,
//! not `UniformScaleMap`, so anisotropic `dx/dy/dz` - already ordinary in
//! this crate's own `pressure::cartesian::CartesianGrid` - are represented
//! exactly rather than approximated). Every LeafNode is written in full
//! (`NO_MASK_AND_ALL_VALS`, §"Compression" below), so the file is large but
//! requires no compression codec, matching what "uncompressed" was asked
//! for. Voxels are `f32` by default and `f16` under
//! [`crate::io::nvdb::Precision::F16`] - see the `fp16` section below. [`UniformGrid`] is `nvdb`'s type, reused rather than redefined - see
//! its own doc for the coordinate convention.
//!
//! # Compression: always `COMPRESS_NONE`, and what that byte format is
//!
//! `io::writeCompressedValues` (`Compression.h`) with `compress ==
//! COMPRESS_NONE` never engages `COMPRESS_ACTIVE_MASK`, so it always takes
//! the simplest branch: write one `int8_t` metadata byte, always
//! `NO_MASK_AND_ALL_VALS` (`= 6`), then every one of the node's values as
//! raw, uncompressed `f32`s (`writeData` with `compression == 0` is a plain
//! `os.write`). No mask-selected subsetting, no inactive-value table - the
//! full dense array, every time. This is also why a Lower/Upper
//! `InternalNode`'s "value" array is written even though this writer never
//! places a value there (every non-child slot is simply background/off):
//! the format writes the full `NUM_VALUES`-length array regardless, so it
//! is trivially all zero here.
//!
//! # `RootNode`'s child order must be `Coord`-sorted; `InternalNode`'s must
//! not be
//!
//! `RootNode`'s children live in a `std::map<Coord, Tile>`, so on a real
//! read they always iterate lexicographically by `(x, y, z)` regardless of
//! file order - `writeTopology`'s child loop and `writeBuffers`' child loop
//! are *two separate passes* over the same live map, so they only agree if
//! both list children in that same sorted order (`math::Coord::operator<`,
//! `Coord.h`: lexicographic, `x` most significant). This writer therefore
//! enumerates Upper nodes with `x` outermost and `z` innermost. `Internal
//! Node`'s children, by contrast, live in a fixed-size array indexed by the
//! bit-packed offset `(x<<2L)|(y<<L)|z`
//! (`InternalNode::coordToOffset`, identical to NanoVDB's own, since they
//! are the same project's two encodings of the same tree shape) - so Lower/
//! Upper children are enumerated in that packed order, not sorted-by-
//! coordinate, matching `cbeginChildOn()`'s traversal.
//!
//! # `fp16`: the `_HalfFloat` save preference - SPEC-LIT S45
//!
//! OpenVDB has no half-float grid TYPE. It has a per-grid save preference
//! (`GridBase::setSaveFloatAsHalf`, `Grid.cc`), and four things change on the
//! way out when it is set - transcribed from the same `v13.0.0` sources as
//! everything above:
//!
//! 1. `GridDescriptor::writeHeader` appends `HALF_FLOAT_TYPENAME_SUFFIX`
//!    (`"_HalfFloat"`, `GridDescriptor.cc`) to the grid type string. That
//!    suffix is the ONLY thing a reader keys on: `GridDescriptor::read`
//!    strips it and calls `io::setHalfFloat(is, ...)` before a single node is
//!    parsed.
//! 2. `InternalNode::writeTopology(os, toHalf)` passes `toHalf` into
//!    `io::writeCompressedValues` for its own `NUM_VALUES` value array, so
//!    that array becomes 2 bytes per entry.
//! 3. `LeafNode::writeBuffers(os, toHalf)` does the same for its 512 values.
//! 4. `GridBase::setSaveFloatAsHalf` inserts `META_SAVE_HALF_FLOAT`
//!    (`"is_saved_as_half_float"`, a `BoolMetadata`) into the grid's metadata
//!    map, so `Grid::writeMeta` emits it - sorted after `"class"`, since
//!    `MetaMap` is a `std::map<Name, ...>`.
//!
//! What does NOT change: `RootNode::writeTopology` writes its background
//! through `io::truncateRealToHalf` and then as `sizeof(ValueType)` - four
//! bytes still, rounded to half precision, which for this writer's `0.0`
//! background is the identity. Masks, the `AffineMap` transform, the
//! stream-position triple and the `COMPRESS_NONE` word are all untouched.
//!
//! Point 2 is the trap: an internal node's value array is written by
//! `writeTopology`, not `writeBuffers`, and it is easy to halve only the leaf
//! buffers. The contents there are all zero either way (binary16 zero is
//! `0x0000`, and this writer places no constant tiles), so only the LENGTH
//! changes - which a value-comparing test cannot see and [`reader`]'s
//! position asserts can.
//!
//! # Externally unverified
//!
//! No OpenVDB build, Blender, or ParaView is available in this environment
//! to open the files this module writes. [`tests`] verifies both precisions
//! of the writer against [`reader`], an internal reader built from the same source
//! citations above, independently walking `RootNode`/`InternalNode`/
//! `LeafNode` topology and buffers rather than re-deriving node existence
//! from the same formulas the writer used - so it is a genuine structural
//! check, not a tautology - and every voxel round-trips bit-exact. That
//! proves this writer and this reader agree with each other about what the
//! byte-layout research above says the format is; it does not prove a real
//! OpenVDB reader agrees. Mark this "structurally validated, externally
//! unverified" until it has been opened in Blender or ParaView.
//!
//! No GPL-licensed source was consulted.

use crate::error::{Error, IoContext, Result};
use crate::io::nvdb::{f32_to_f16_bits, Precision, UniformGrid};
use crate::io::output_types::{FieldValues, OutputField};
use crate::Scalar;
use std::path::Path;

// ============================================================================
//  Format constants - openvdb/openvdb/version.h.in, io/Compression.h, v13.0.0
// ============================================================================

/// `OPENVDB_MAGIC`, written as the low 32 bits of an `int64_t` (`Archive::
/// writeHeader`: `int64_t magic = OPENVDB_MAGIC; os.write(&magic, 8)` -
/// sign-extension of a positive `int32_t` is zero-extension).
const MAGIC: u64 = 0x5644_4220;
/// `OPENVDB_FILE_VERSION` at the `v13.0.0` tag (>= 224, the task's floor -
/// `OPENVDB_FILE_VERSION_MULTIPASS_IO`).
const FILE_VERSION: u32 = 225;
/// `OPENVDB_LIBRARY_MAJOR_VERSION_NUMBER` / `_MINOR_VERSION_NUMBER` at the
/// `v13.0.0` tag. Purely informational - no reader branches on these.
const LIB_MAJOR: u32 = 13;
const LIB_MINOR: u32 = 0;
/// `io::COMPRESS_NONE`.
const COMPRESS_NONE: u32 = 0;
/// The per-node-buffer indicator byte meaning "no mask compression, the
/// full array follows" (`io::Compression.h`, the seventh and last entry of
/// the anonymous `enum` documented as "> 2 inactive vals, so no mask
/// compression at all"). The only one this writer ever emits.
const NO_MASK_AND_ALL_VALS: i8 = 6;
/// `io::HALF_FLOAT_TYPENAME_SUFFIX` (`GridDescriptor.cc`), appended to the
/// grid type string by `GridDescriptor::writeHeader` when the grid's
/// `saveFloatAsHalf` preference is set. SPEC-LIT S45.1(1).
const HALF_FLOAT_TYPENAME_SUFFIX: &str = "_HalfFloat";
/// `GridBase::META_SAVE_HALF_FLOAT` (`Grid.cc`) - the `BoolMetadata` entry
/// `setSaveFloatAsHalf` inserts, which `Grid::writeMeta` then emits.
/// SPEC-LIT S45.1(4).
const META_SAVE_HALF_FLOAT: &str = "is_saved_as_half_float";
/// The base grid type: OpenVDB's own default `FloatTree` configuration.
const TREE_TYPE: &str = "Tree_float_5_4_3";

/// Bytes per voxel on disk, per SPEC-LIT S45.1: `f32` normally, `f16` under
/// the `_HalfFloat` save preference. `Precision::width` is private to
/// `nvdb` and describes NanoVDB's own node layout, which has nothing in
/// common with this one beyond the number, so it is stated here.
fn value_width(p: Precision) -> usize {
    match p {
        Precision::F32 => 4,
        Precision::F16 => 2,
    }
}

/// One voxel, in whichever of the two on-disk widths is in force.
///
/// The `f16` conversion is `nvdb`'s own (SPEC-LIT S45.2) - not a second
/// implementation of IEEE 754 binary16 living in this file.
fn put_value(b: &mut [u8], at: usize, v: f32, p: Precision) {
    match p {
        Precision::F32 => b[at..at + 4].copy_from_slice(&v.to_le_bytes()),
        Precision::F16 => b[at..at + 2].copy_from_slice(&f32_to_f16_bits(v).to_le_bytes()),
    }
}

const LEAF_LOG2DIM: u32 = 3;
const LOWER_LOG2DIM: u32 = 4;
const UPPER_LOG2DIM: u32 = 5;
const LEAF_DIM: usize = 1 << LEAF_LOG2DIM; // 8
const LOWER_DIM: usize = LEAF_DIM << LOWER_LOG2DIM; // 128
const UPPER_DIM: usize = LOWER_DIM << UPPER_LOG2DIM; // 4096

#[inline]
fn ceil_div(a: usize, b: usize) -> usize {
    (a + b - 1) / b
}

// ============================================================================
//  Public API
// ============================================================================

/// Write `fields` on `grid` to `path` as a single `.vdb` archive - see the
/// module doc. A `FieldValues::Vector` field becomes four grids (`name.x`,
/// `name.y`, `name.z`, `name.mag`); a `FieldValues::Scalar` field becomes
/// one.
///
/// `precision` is SPEC-LIT S44.3's `output.visualisation.precision`:
/// [`Precision::F32`] is the plain `FloatTree` every `.vdb` this crate has
/// ever written, [`Precision::F16`] is OpenVDB's `_HalfFloat` save
/// preference (S45).
pub fn write(
    path: impl AsRef<Path>,
    grid: &UniformGrid,
    fields: &[OutputField],
    precision: Precision,
) -> Result<()> {
    let path = path.as_ref();
    if grid.nx == 0 || grid.ny == 0 || grid.nz == 0 {
        return Err(Error::Config(format!(
            "vdb: grid dims must all be positive, got {}x{}x{}",
            grid.nx, grid.ny, grid.nz
        )));
    }
    if fields.is_empty() {
        return Err(Error::Config("vdb: no fields to write".into()));
    }

    let n = grid.n();
    let mut segments: Vec<(String, Vec<Scalar>)> = Vec::new();
    for field in fields {
        match &field.values {
            FieldValues::Scalar(v) => {
                check_len(field.name, v.len(), n)?;
                segments.push((field.name.to_string(), v.to_vec()));
            }
            FieldValues::Vector(v) => {
                check_len(field.name, v.len(), n)?;
                segments.push((format!("{}.x", field.name), v.iter().map(|p| p.x).collect()));
                segments.push((format!("{}.y", field.name), v.iter().map(|p| p.y).collect()));
                segments.push((format!("{}.z", field.name), v.iter().map(|p| p.z).collect()));
                segments.push((format!("{}.mag", field.name), v.iter().map(|p| p.mag()).collect()));
            }
        }
    }

    let mut buf = Vec::new();
    write_header(&mut buf);
    put_u32(&mut buf, 0); // file-level MetaMap: no global metadata

    put_i32(&mut buf, segments.len() as i32); // grid count

    for (name, values) in &segments {
        write_grid(&mut buf, grid, name, values, precision);
    }

    std::fs::write(path, &buf).path(path)
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
//  Byte-buffer primitives (little-endian; `os.write(reinterpret_cast<...>)`
//  on the native machine OpenVDB was built for, universally x86/ARM little-
//  endian in practice)
// ============================================================================

fn put_u8(b: &mut Vec<u8>, v: u8) {
    b.push(v);
}
fn put_bytes(b: &mut Vec<u8>, v: &[u8]) {
    b.extend_from_slice(v);
}
fn put_u32(b: &mut Vec<u8>, v: u32) {
    put_bytes(b, &v.to_le_bytes());
}
fn put_i32(b: &mut Vec<u8>, v: i32) {
    put_bytes(b, &v.to_le_bytes());
}
fn put_u64(b: &mut Vec<u8>, v: u64) {
    put_bytes(b, &v.to_le_bytes());
}
fn put_i64(b: &mut Vec<u8>, v: i64) {
    put_bytes(b, &v.to_le_bytes());
}
fn put_f32(b: &mut Vec<u8>, v: f32) {
    put_bytes(b, &v.to_le_bytes());
}
fn put_f64(b: &mut Vec<u8>, v: f64) {
    put_bytes(b, &v.to_le_bytes());
}
/// `util::writeString` (`util/Name.h`): a `uint32_t` byte length, then the
/// raw bytes - no NUL terminator.
fn put_string(b: &mut Vec<u8>, s: &str) {
    put_u32(b, s.as_bytes().len() as u32);
    put_bytes(b, s.as_bytes());
}
/// Patch an already-written `i64` at `at` (used for the three stream-
/// position placeholders `GridDescriptor::writeStreamPos` leaves behind to
/// be rewritten once the real positions are known).
fn patch_i64(b: &mut [u8], at: usize, v: i64) {
    b[at..at + 8].copy_from_slice(&v.to_le_bytes());
}

fn set_mask_bit(b: &mut [u8], mask_off: usize, n: u32) {
    let word_off = mask_off + 8 * (n as usize / 64);
    let bit = n % 64;
    let mut w = u64::from_le_bytes([0, 0, 0, 0, 0, 0, 0, 0]);
    w |= u64::from_le_bytes(b[word_off..word_off + 8].try_into().unwrap_or([0; 8]));
    w |= 1u64 << bit;
    b[word_off..word_off + 8].copy_from_slice(&w.to_le_bytes());
}

// ============================================================================
//  File / grid framing - Archive.cc, GridDescriptor.cc
// ============================================================================

/// `Archive::writeHeader`: magic(i64) + fileVersion(u32) + libMajor(u32) +
/// libMinor(u32) + hasGridOffsets(1 byte, `true` - this is a seekable file
/// with real offsets, not a bare stream) + a 36-byte UUID string. The UUID
/// is cosmetic (compared only against itself, for detecting a changed input
/// file on re-read) - Version 4 layout (`xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx`
/// is not required by the reader, which just stores the string opaquely) -
/// so this writes a fixed, valid-shaped placeholder rather than pulling in
/// a randomness dependency for no behavioural difference.
fn write_header(b: &mut Vec<u8>) {
    put_u64(b, MAGIC);
    put_u32(b, FILE_VERSION);
    put_u32(b, LIB_MAJOR);
    put_u32(b, LIB_MINOR);
    put_u8(b, 1); // hasGridOffsets = seekable = true
    put_bytes(b, b"00000000-0000-0000-0000-000000000000");
}

/// One grid: `GridDescriptor::writeHeader` + `writeStreamPos` (patched) +
/// `Archive::writeGrid`'s body (`setGridCompression`, `writeMeta`,
/// `writeTransform`, `writeTopology`, `writeBuffers`), in that order.
fn write_grid(
    b: &mut Vec<u8>,
    grid: &UniformGrid,
    name: &str,
    values: &[Scalar],
    precision: Precision,
) {
    put_string(b, name); // GridDescriptor::mUniqueName
    // `GridDescriptor::writeHeader`: the tree type, plus HALF_FLOAT_TYPENAME_
    // SUFFIX when the grid's saveFloatAsHalf preference is set. SPEC-LIT
    // S45.1(1) - this suffix is the whole of what tells a reader the node
    // buffers below are two bytes wide.
    let grid_type = match precision {
        Precision::F32 => TREE_TYPE.to_string(),
        Precision::F16 => format!("{TREE_TYPE}{HALF_FLOAT_TYPENAME_SUFFIX}"),
    };
    put_string(b, &grid_type);
    put_string(b, ""); // instance parent name: never an instance here

    let offset_pos = b.len();
    put_i64(b, 0); // gridPos placeholder
    put_i64(b, 0); // blockPos placeholder
    put_i64(b, 0); // endPos placeholder
    let grid_pos = b.len() as i64;

    put_u32(b, COMPRESS_NONE); // Archive::setGridCompression's stream word

    write_grid_metadata(b, precision);
    write_transform(b, grid);
    write_topology(b, grid, values, precision);
    let block_pos = b.len() as i64;
    write_buffers(b, grid, values, precision);
    let end_pos = b.len() as i64;

    patch_i64(b, offset_pos, grid_pos);
    patch_i64(b, offset_pos + 8, block_pos);
    patch_i64(b, offset_pos + 16, end_pos);
}

/// `GridBase::writeMeta` -> `MetaMap::writeMeta`: a `uint32_t` entry count,
/// then per entry `writeString(name)` + `writeString(typeName)` +
/// `Metadata::write` (`writeSize` u32 + `writeValue` raw bytes - see
/// `StringMetadata::size`/`writeValue`, `Metadata.h`). One entry here:
/// `GridBase::META_GRID_CLASS` ("class") = `gridClassToString(GRID_FOG_VOLUME)`
/// = "fog volume" (`Grid.cc`) - *DESIGN*, same choice `nvdb.rs` makes and
/// for the same reason (a plain scalar/component field renders sensibly as
/// a volume/fog density by default).
fn write_grid_metadata(b: &mut Vec<u8>, precision: Precision) {
    // `MetaMap` is a `std::map<Name, Metadata::Ptr>`, so entries come out in
    // ascending name order: "class" before "is_saved_as_half_float".
    let n = match precision {
        Precision::F32 => 1,
        Precision::F16 => 2,
    };
    put_u32(b, n); // metaCount
    put_string(b, "class");
    put_string(b, "string");
    let value = b"fog volume";
    put_u32(b, value.len() as u32); // StringMetadata::size()
    put_bytes(b, value);

    // SPEC-LIT S45.1(4). `GridBase::setSaveFloatAsHalf` (`Grid.cc`) inserts
    // this `BoolMetadata`; `TypedMetadata<bool>` uses the generic
    // `size() = sizeof(bool) = 1` and `writeValue` = one raw byte
    // (`Metadata.h`). An fp32 grid never had `setSaveFloatAsHalf` called on
    // it and so carries no such entry at all - which is what keeps the fp32
    // byte stream exactly what it was before S45.
    if precision == Precision::F16 {
        put_string(b, META_SAVE_HALF_FLOAT);
        put_string(b, "bool");
        put_u32(b, 1); // TypedMetadata<bool>::size()
        put_u8(b, 1);
    }
}

/// `Grid::writeTransform` -> `Transform::write`: `writeString(map->type())`
/// + `map->write()`. Always an `AffineMap` (`Maps.h::AffineMap::mapType() =
/// "AffineMap"`), whose `write` is `Mat4d::write` (`Mat.h`: 16 raw `f64`,
/// row-major, `mm[i*4+j]`). `operator*(Vec3d, Mat4)` (`Mat4.h`) evaluates
/// `v*M` as a row-vector on the left, so the translation lives in row 3 and
/// each axis's scale is the corresponding diagonal entry - see the module
/// doc for the `origin`/`spacing` convention this encodes (same one
/// `nvdb.rs` uses: `origin` is the world position of cell `(0,0,0)`'s
/// centre).
fn write_transform(b: &mut Vec<u8>, grid: &UniformGrid) {
    put_string(b, "AffineMap");
    let (dx, dy, dz) = (grid.spacing.x as f64, grid.spacing.y as f64, grid.spacing.z as f64);
    let (ox, oy, oz) = (grid.origin.x as f64, grid.origin.y as f64, grid.origin.z as f64);
    #[rustfmt::skip]
    let m: [f64; 16] = [
        dx,  0.0, 0.0, 0.0,
        0.0, dy,  0.0, 0.0,
        0.0, 0.0, dz,  0.0,
        ox,  oy,  oz,  1.0,
    ];
    for v in m {
        put_f64(b, v);
    }
}

fn node_counts(grid: &UniformGrid) -> ([usize; 3], [usize; 3], [usize; 3]) {
    let n_leaf = [
        ceil_div(grid.nx, LEAF_DIM), ceil_div(grid.ny, LEAF_DIM), ceil_div(grid.nz, LEAF_DIM),
    ];
    let n_lower = [
        ceil_div(grid.nx, LOWER_DIM), ceil_div(grid.ny, LOWER_DIM), ceil_div(grid.nz, LOWER_DIM),
    ];
    let n_upper = [
        ceil_div(grid.nx, UPPER_DIM), ceil_div(grid.ny, UPPER_DIM), ceil_div(grid.nz, UPPER_DIM),
    ];
    (n_leaf, n_lower, n_upper)
}

// ============================================================================
//  Topology - RootNode::writeTopology / InternalNode::writeTopology /
//  LeafNode::writeTopology (tree/{Root,Internal,Leaf}Node.h)
// ============================================================================

/// `RootNode<ChildT>::writeTopology`: `background`(`f32`) + `numTiles`(u32,
/// always 0 - this writer never places a root-level constant tile) +
/// `numChildren`(u32), then, per child (`std::map<Coord,_>` order -
/// lexicographic by `(x,y,z)`, see the module doc): `origin`(3x`i32`) +
/// the child's own `writeTopology`.
fn write_topology(b: &mut Vec<u8>, grid: &UniformGrid, values: &[Scalar], precision: Precision) {
    let (_n_leaf, n_lower, n_upper) = node_counts(grid);
    put_f32(b, 0.0); // background
    put_u32(b, 0); // numTiles
    put_u32(b, (n_upper[0] * n_upper[1] * n_upper[2]) as u32); // numChildren

    // x outermost, z innermost: ascending Coord order.
    for ux in 0..n_upper[0] {
        for uy in 0..n_upper[1] {
            for uz in 0..n_upper[2] {
                let origin = [ux * UPPER_DIM, uy * UPPER_DIM, uz * UPPER_DIM];
                put_i32(b, origin[0] as i32);
                put_i32(b, origin[1] as i32);
                put_i32(b, origin[2] as i32);
                write_internal_topology(
                    b, grid, values, origin, UPPER_LOG2DIM, LOWER_DIM, n_lower, true, precision,
                );
            }
        }
    }
}

/// `InternalNode<ChildT,Log2Dim>::writeTopology`: `mChildMask.save` (raw
/// words) + `mValueMask.save` (raw words, always all-zero here - no
/// Lower/Upper-level constant tiles in a dense box) + `io::
/// writeCompressedValues` on the full `NUM_VALUES` array under
/// `COMPRESS_NONE` (metadata byte `NO_MASK_AND_ALL_VALS` + `NUM_VALUES` raw
/// `f32`s, all zero: every non-child slot is background), then each ON
/// child in bit-packed offset order (`coordToOffset`/`cbeginChildOn`, *not*
/// `Coord`-sorted - see the module doc).
#[allow(clippy::too_many_arguments)]
fn write_internal_topology(
    b: &mut Vec<u8>,
    grid: &UniformGrid,
    values: &[Scalar],
    origin: [usize; 3],
    log2dim: u32,
    child_span: usize,
    n_child: [usize; 3], // child-grid dimensions in the CHILD's own units (Lower or Leaf)
    is_upper: bool,
    precision: Precision,
) {
    let slots = 1usize << (3 * log2dim);
    let mask_words = slots / 64;
    let child_mask_off = b.len();
    put_bytes(b, &vec![0u8; mask_words * 8]); // mChildMask, filled in below
    put_bytes(b, &vec![0u8; mask_words * 8]); // mValueMask: always zero

    put_u8(b, NO_MASK_AND_ALL_VALS as u8);
    // NUM_VALUES entries, all zero (no tiles) - and `toHalf` reaches THIS
    // array too, through `InternalNode::writeTopology`'s own
    // `io::writeCompressedValues` call, so under fp16 it is half as long.
    // Binary16 zero is 0x0000, so only the length changes. SPEC-LIT S45.1(2).
    put_bytes(b, &vec![0u8; slots * value_width(precision)]);

    let (n_leaf, _, _) = node_counts(grid);
    let fan = 1usize << log2dim;
    for a in 0..fan {
        let cx = origin[0] / child_span + a;
        if cx >= n_child[0] {
            continue;
        }
        for bb in 0..fan {
            let cy = origin[1] / child_span + bb;
            if cy >= n_child[1] {
                continue;
            }
            for c in 0..fan {
                let cz = origin[2] / child_span + c;
                if cz >= n_child[2] {
                    continue;
                }
                let n = ((a as u32) << (2 * log2dim)) | ((bb as u32) << log2dim) | (c as u32);
                set_mask_bit(b, child_mask_off, n);
                let child_origin = [cx * child_span, cy * child_span, cz * child_span];
                if is_upper {
                    write_internal_topology(
                        b, grid, values, child_origin, LOWER_LOG2DIM, LEAF_DIM, n_leaf, false,
                        precision,
                    );
                } else {
                    write_leaf_topology(b, grid, child_origin);
                }
            }
        }
    }
}

/// `LeafNode::writeTopology`: just `mValueMask.save` - the raw 64-byte
/// (8-word) bit array, bit `n = (x<<6)|(y<<3)|z` (`LeafNode::coordToOffset`,
/// identical convention to NanoVDB's).
fn write_leaf_topology(b: &mut Vec<u8>, grid: &UniformGrid, origin: [usize; 3]) {
    let mask_off = b.len();
    put_bytes(b, &[0u8; 64]);
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
                set_mask_bit(b, mask_off, n);
            }
        }
    }
}

// ============================================================================
//  Buffers - RootNode::writeBuffers / InternalNode::writeBuffers /
//  LeafNode::writeBuffers
// ============================================================================

/// `RootNode::writeBuffers`: recurse into children only (a root has no
/// buffer of its own), same order as `write_topology`.
fn write_buffers(b: &mut Vec<u8>, grid: &UniformGrid, values: &[Scalar], precision: Precision) {
    let (n_leaf, n_lower, n_upper) = node_counts(grid);
    for ux in 0..n_upper[0] {
        for uy in 0..n_upper[1] {
            for uz in 0..n_upper[2] {
                let origin = [ux * UPPER_DIM, uy * UPPER_DIM, uz * UPPER_DIM];
                write_internal_buffers(
                    b, grid, values, origin, UPPER_LOG2DIM, LOWER_DIM, n_lower, true, n_leaf,
                    precision,
                );
            }
        }
    }
}

/// `InternalNode::writeBuffers`: recurse into children only, in bit-packed
/// offset order (must match `write_internal_topology`'s child order).
#[allow(clippy::too_many_arguments)]
fn write_internal_buffers(
    b: &mut Vec<u8>,
    grid: &UniformGrid,
    values: &[Scalar],
    origin: [usize; 3],
    log2dim: u32,
    child_span: usize,
    n_child: [usize; 3],
    is_upper: bool,
    n_leaf: [usize; 3],
    precision: Precision,
) {
    let fan = 1usize << log2dim;
    for a in 0..fan {
        let cx = origin[0] / child_span + a;
        if cx >= n_child[0] {
            continue;
        }
        for bb in 0..fan {
            let cy = origin[1] / child_span + bb;
            if cy >= n_child[1] {
                continue;
            }
            for c in 0..fan {
                let cz = origin[2] / child_span + c;
                if cz >= n_child[2] {
                    continue;
                }
                let child_origin = [cx * child_span, cy * child_span, cz * child_span];
                if is_upper {
                    write_internal_buffers(
                        b, grid, values, child_origin, LOWER_LOG2DIM, LEAF_DIM, n_leaf, false,
                        n_leaf, precision,
                    );
                } else {
                    write_leaf_buffers(b, grid, values, child_origin, precision);
                }
            }
        }
    }
}

/// `LeafNode::writeBuffers`: `mValueMask.save` again (yes, a second copy -
/// OpenVDB's multi-pass I/O writes topology and buffers as two full tree
/// traversals, and the mask happens to be needed, and written, in both) +
/// `io::writeCompressedValues`: metadata byte `NO_MASK_AND_ALL_VALS` + all
/// 512 values raw, background (`0.0`) where inactive.
fn write_leaf_buffers(
    b: &mut Vec<u8>,
    grid: &UniformGrid,
    values: &[Scalar],
    origin: [usize; 3],
    precision: Precision,
) {
    let w = value_width(precision);
    let mask_off = b.len();
    put_bytes(b, &[0u8; 64]);
    let values_off = b.len();
    put_u8(b, NO_MASK_AND_ALL_VALS as u8);
    put_bytes(b, &vec![0u8; 512 * w]);

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
                set_mask_bit(b, mask_off, n);
                let v = values[grid.idx(gx, gy, gz)] as f32;
                let at = values_off + 1 + (n as usize) * w;
                put_value(b, at, v, precision);
            }
        }
    }
}

// ============================================================================
//  Internal reader - for the round-trip test only. Genuinely walks the
//  format: it builds an in-memory tree from the topology pass (reading the
//  same child/value masks the writer set) and then walks *that* tree for
//  the buffers pass, exactly as a real OpenVDB reader must (the buffers
//  pass does not repeat the child mask - `InternalNode::writeBuffers` is
//  pure recursion over the live tree the topology pass already built). This
//  is therefore a genuine structural check of the mask/metadata bytes, not
//  a tautology that recomputes node existence from the writer's own
//  ceil-division formulas.
// ============================================================================

#[cfg(test)]
mod reader {
    use super::*;

    pub struct ReadGrid {
        pub name: String,
        /// As written, `_HalfFloat` suffix and all - so a test can assert
        /// the suffix rather than infer it.
        pub grid_type: String,
        /// What the reader DERIVED from that suffix, exactly as
        /// `GridDescriptor::read` does before parsing a single node.
        pub precision: Precision,
        pub class_metadata: Option<String>,
        /// `is_saved_as_half_float`, when the grid carries it.
        pub half_metadata: Option<bool>,
        pub voxel_size: [f64; 3],
        pub origin: [f64; 3],
        pub nx: usize,
        pub ny: usize,
        pub nz: usize,
        pub values: Vec<f64>, // dense, idx = i + nx*(j + ny*k); NaN where inactive
    }

    fn get_u32(b: &[u8], off: usize) -> u32 {
        u32::from_le_bytes([b[off], b[off + 1], b[off + 2], b[off + 3]])
    }
    fn get_i32(b: &[u8], off: usize) -> i32 {
        i32::from_le_bytes([b[off], b[off + 1], b[off + 2], b[off + 3]])
    }
    fn get_u64(b: &[u8], off: usize) -> u64 {
        let mut a = [0u8; 8];
        a.copy_from_slice(&b[off..off + 8]);
        u64::from_le_bytes(a)
    }
    fn get_i64(b: &[u8], off: usize) -> i64 {
        let mut a = [0u8; 8];
        a.copy_from_slice(&b[off..off + 8]);
        i64::from_le_bytes(a)
    }
    fn get_f32(b: &[u8], off: usize) -> f32 {
        f32::from_le_bytes([b[off], b[off + 1], b[off + 2], b[off + 3]])
    }
    fn get_u16(b: &[u8], off: usize) -> u16 {
        u16::from_le_bytes([b[off], b[off + 1]])
    }
    /// One voxel at `off`, in whichever width the grid type's suffix said.
    /// The `f16` direction is `nvdb`'s own inverse, for the same reason the
    /// writer's forward direction is (SPEC-LIT S45.2).
    fn get_value(b: &[u8], off: usize, p: Precision) -> f64 {
        match p {
            Precision::F32 => get_f32(b, off) as f64,
            Precision::F16 => crate::io::nvdb::f16_bits_to_f32(get_u16(b, off)) as f64,
        }
    }
    fn get_f64(b: &[u8], off: usize) -> f64 {
        let mut a = [0u8; 8];
        a.copy_from_slice(&b[off..off + 8]);
        f64::from_le_bytes(a)
    }
    fn mask_bit(b: &[u8], mask_off: usize, n: u32) -> bool {
        let word_off = mask_off + 8 * (n as usize / 64);
        let w = get_u64(b, word_off);
        (w >> (n % 64)) & 1 != 0
    }
    fn get_string(b: &[u8], pos: &mut usize) -> String {
        let len = get_u32(b, *pos) as usize;
        *pos += 4;
        let s = String::from_utf8_lossy(&b[*pos..*pos + len]).into_owned();
        *pos += len;
        s
    }

    /// The tree this reader reconstructs from the topology pass, in exactly
    /// the child order the writer used (bit-packed offset for Internal
    /// nodes), so the buffers pass below can walk it without needing to
    /// re-derive anything.
    enum Node {
        Internal { children: Vec<Node> },
        Leaf { origin: [i32; 3], active: [bool; 512] },
    }

    pub fn read_all(bytes: &[u8]) -> Vec<ReadGrid> {
        let magic = get_u64(bytes, 0);
        assert_eq!(magic, MAGIC, "not a .vdb file");
        let mut pos = 8 + 4 + 4 + 4; // magic, fileVersion, libMajor, libMinor
        pos += 1; // hasGridOffsets
        pos += 36; // uuid

        let meta_count = get_u32(bytes, pos);
        pos += 4;
        assert_eq!(meta_count, 0, "reader only handles files with no file-level metadata");

        let grid_count = get_i32(bytes, pos) as usize;
        pos += 4;

        let mut out = Vec::new();
        for _ in 0..grid_count {
            let name = get_string(bytes, &mut pos);
            let grid_type = get_string(bytes, &mut pos);
            // `GridDescriptor::read`: strip HALF_FLOAT_TYPENAME_SUFFIX and
            // set the stream's half flag from it, BEFORE anything else about
            // the grid is parsed. SPEC-LIT S45.1(1).
            let (base_type, precision) = match grid_type.strip_suffix(HALF_FLOAT_TYPENAME_SUFFIX) {
                Some(base) => (base.to_string(), Precision::F16),
                None => (grid_type.clone(), Precision::F32),
            };
            assert_eq!(base_type, TREE_TYPE, "reader only handles {TREE_TYPE}");
            let width = value_width(precision);
            let _instance_parent = get_string(bytes, &mut pos);

            let grid_pos = get_i64(bytes, pos) as usize;
            let block_pos = get_i64(bytes, pos + 8) as usize;
            let end_pos = get_i64(bytes, pos + 16) as usize;
            pos += 24;
            assert_eq!(pos, grid_pos, "gridPos must point right after the stream-position triple");

            let compression = get_u32(bytes, pos);
            pos += 4;
            assert_eq!(compression, COMPRESS_NONE, "reader only handles COMPRESS_NONE");

            let meta_count = get_u32(bytes, pos);
            pos += 4;
            let mut class_metadata = None;
            let mut half_metadata = None;
            for _ in 0..meta_count {
                let mname = get_string(bytes, &mut pos);
                let mtype = get_string(bytes, &mut pos);
                let len = get_u32(bytes, pos) as usize;
                pos += 4;
                match mtype.as_str() {
                    "string" => {
                        let value =
                            String::from_utf8_lossy(&bytes[pos..pos + len]).into_owned();
                        if mname == "class" {
                            class_metadata = Some(value);
                        }
                    }
                    "bool" => {
                        assert_eq!(len, 1, "TypedMetadata<bool>::size() is sizeof(bool)");
                        if mname == META_SAVE_HALF_FLOAT {
                            half_metadata = Some(bytes[pos] != 0);
                        }
                    }
                    other => panic!("reader does not handle {other} grid metadata"),
                }
                pos += len;
            }

            let map_type = get_string(bytes, &mut pos);
            assert_eq!(map_type, "AffineMap");
            let mut m = [0f64; 16];
            for v in m.iter_mut() {
                *v = get_f64(bytes, pos);
                pos += 8;
            }
            let voxel_size = [m[0], m[5], m[10]];
            let origin = [m[12], m[13], m[14]];

            // SPEC-LIT S45.1(5): the background stays `sizeof(ValueType)`
            // even under `toHalf` - `RootNode::writeTopology` truncates the
            // VALUE to half precision and writes it back as a float.
            let background = get_f32(bytes, pos);
            assert_eq!(background, 0.0, "this writer's background is always 0");
            pos += 4;
            let num_tiles = get_u32(bytes, pos);
            pos += 4;
            let num_children = get_u32(bytes, pos);
            pos += 4;
            assert_eq!(num_tiles, 0, "reader only handles a tile-free root");

            let mut max_xyz = [0i32; 3];
            let mut roots = Vec::new();
            for _ in 0..num_children {
                let child_origin =
                    [get_i32(bytes, pos), get_i32(bytes, pos + 4), get_i32(bytes, pos + 8)];
                pos += 12;
                let (node, new_pos) = read_internal_topology(
                    bytes, pos, UPPER_LOG2DIM, child_origin, &mut max_xyz, width,
                );
                pos = new_pos;
                roots.push(node);
            }
            assert_eq!(pos, block_pos, "topology must end exactly at blockPos");

            let (nx, ny, nz) =
                ((max_xyz[0] + 1) as usize, (max_xyz[1] + 1) as usize, (max_xyz[2] + 1) as usize);
            let mut values = vec![f64::NAN; nx * ny * nz];

            let mut bpos = block_pos;
            for node in &roots {
                bpos = read_buffers(bytes, bpos, node, nx, ny, &mut values, precision);
            }
            assert_eq!(bpos, end_pos, "buffers must end exactly at endPos");

            pos = end_pos;
            out.push(ReadGrid {
                name, grid_type, precision, class_metadata, half_metadata, voxel_size, origin,
                nx, ny, nz, values,
            });
        }
        out
    }

    /// Read one Internal node's topology (`mChildMask` + `mValueMask` +
    /// the `NO_MASK_AND_ALL_VALS` value array, all per `InternalNode::
    /// writeTopology`), recursing into each ON child in the same
    /// bit-packed order the writer used, and return the reconstructed
    /// subtree plus the byte position right after it.
    fn read_internal_topology(
        bytes: &[u8],
        pos: usize,
        log2dim: u32,
        origin: [i32; 3],
        max_xyz: &mut [i32; 3],
        width: usize,
    ) -> (Node, usize) {
        let slots = 1usize << (3 * log2dim);
        let mask_words = slots / 8;
        let child_mask_off = pos;
        let value_off = pos + 2 * mask_words;
        let metadata = bytes[value_off];
        assert_eq!(metadata, NO_MASK_AND_ALL_VALS as u8, "this reader only handles NO_MASK_AND_ALL_VALS");
        // SPEC-LIT S45.1(2): `width`, not 4. Halving only the leaf buffers
        // lands here as a position mismatch, which is what the `assert_eq!`
        // on `blockPos` in `read_all` then reports.
        let mut cursor = value_off + 1 + slots * width;

        let fan = 1u32 << log2dim;
        let child_span = if log2dim == UPPER_LOG2DIM { LOWER_DIM } else { LEAF_DIM } as i32;
        let mut children = Vec::new();
        for n in 0..slots as u32 {
            if !mask_bit(bytes, child_mask_off, n) {
                continue;
            }
            let a = n >> (2 * log2dim);
            let bb = (n >> log2dim) & (fan - 1);
            let c = n & (fan - 1);
            let child_origin = [
                origin[0] + a as i32 * child_span,
                origin[1] + bb as i32 * child_span,
                origin[2] + c as i32 * child_span,
            ];
            if log2dim == UPPER_LOG2DIM {
                let (node, new_cursor) = read_internal_topology(
                    bytes, cursor, LOWER_LOG2DIM, child_origin, max_xyz, width,
                );
                children.push(node);
                cursor = new_cursor;
            } else {
                let (node, new_cursor) = read_leaf_topology(bytes, cursor, child_origin, max_xyz);
                children.push(node);
                cursor = new_cursor;
            }
        }
        (Node::Internal { children }, cursor)
    }

    /// `LeafNode::writeTopology`: just the 64-byte value mask.
    fn read_leaf_topology(
        bytes: &[u8],
        pos: usize,
        origin: [i32; 3],
        max_xyz: &mut [i32; 3],
    ) -> (Node, usize) {
        let mask_off = pos;
        let mut active = [false; 512];
        for n in 0..512u32 {
            if !mask_bit(bytes, mask_off, n) {
                continue;
            }
            active[n as usize] = true;
            let lx = (n >> 6) as i32;
            let ly = ((n >> 3) & 7) as i32;
            let lz = (n & 7) as i32;
            max_xyz[0] = max_xyz[0].max(origin[0] + lx);
            max_xyz[1] = max_xyz[1].max(origin[1] + ly);
            max_xyz[2] = max_xyz[2].max(origin[2] + lz);
        }
        (Node::Leaf { origin, active }, pos + 64)
    }

    /// Walk the tree built by the topology pass, consuming the buffers
    /// region (`InternalNode::writeBuffers` recurses with no bytes of its
    /// own; `LeafNode::writeBuffers` repeats the value mask then the
    /// `NO_MASK_AND_ALL_VALS` value array) and scattering active values
    /// into the dense `values` array.
    fn read_buffers(
        bytes: &[u8],
        pos: usize,
        node: &Node,
        nx: usize,
        ny: usize,
        values: &mut [f64],
        precision: Precision,
    ) -> usize {
        match node {
            Node::Internal { children } => {
                let mut cursor = pos;
                for child in children {
                    cursor = read_buffers(bytes, cursor, child, nx, ny, values, precision);
                }
                cursor
            }
            Node::Leaf { origin, active } => {
                let mask_off = pos;
                // Cross-check: the buffers pass's own mask must agree with
                // the one the topology pass already read - a real integrity
                // check on the two independent mask writes in the file.
                for n in 0..512u32 {
                    assert_eq!(
                        mask_bit(bytes, mask_off, n), active[n as usize],
                        "topology and buffers value masks disagree at bit {n}"
                    );
                }
                let value_off = pos + 64;
                let metadata = bytes[value_off];
                assert_eq!(metadata, NO_MASK_AND_ALL_VALS as u8);
                let values_start = value_off + 1;
                for n in 0..512u32 {
                    if !active[n as usize] {
                        continue;
                    }
                    let lx = (n >> 6) as i32;
                    let ly = ((n >> 3) & 7) as i32;
                    let lz = (n & 7) as i32;
                    let gx = (origin[0] + lx) as usize;
                    let gy = (origin[1] + ly) as usize;
                    let gz = (origin[2] + lz) as usize;
                    let w = value_width(precision);
                    let v = get_value(bytes, values_start + (n as usize) * w, precision);
                    values[gx + nx * (gy + ny * gz)] = v;
                }
                values_start + 512 * value_width(precision)
            }
        }
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
        p.push(format!("ofgpu_vdb_test_{name}_{}.vdb", std::process::id()));
        p
    }

    fn make_grid(nx: usize, ny: usize, nz: usize) -> UniformGrid {
        UniformGrid {
            nx, ny, nz,
            origin: crate::Vec3::new(1.5, -2.0, 0.25),
            spacing: crate::Vec3::new(0.1, 0.2, 0.05),
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
    fn round_trips_every_voxel_on_a_small_grid() {
        let grid = make_grid(5, 4, 3);
        let field = ramp_field(&grid);
        let path = scratch("small");
        write(&path, &grid, &[OutputField::scalar("T", &field)], Precision::F32).expect("write");

        let bytes = std::fs::read(&path).expect("read back");
        let grids = read_all(&bytes);
        assert_eq!(grids.len(), 1);
        let g = &grids[0];
        assert_eq!(g.name, "T");
        assert_eq!(g.grid_type, "Tree_float_5_4_3");
        assert_eq!(g.class_metadata.as_deref(), Some("fog volume"));
        assert_eq!((g.nx, g.ny, g.nz), (grid.nx, grid.ny, grid.nz));
        for k in 0..grid.nz {
            for j in 0..grid.ny {
                for i in 0..grid.nx {
                    let want = field[grid.idx(i, j, k)] as f32 as f64;
                    let got = g.values[i + grid.nx * (j + grid.ny * k)];
                    assert_eq!(got, want, "voxel ({i},{j},{k})");
                }
            }
        }
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn round_trips_a_multi_node_grid_crossing_lower_boundaries() {
        let grid = make_grid(37, 21, 17); // crosses several Leaf and Lower boundaries
        let field = ramp_field(&grid);
        let path = scratch("multi");
        write(&path, &grid, &[OutputField::scalar("rho", &field)], Precision::F32).expect("write");

        let bytes = std::fs::read(&path).expect("read back");
        let g = &read_all(&bytes)[0];
        for k in 0..grid.nz {
            for j in 0..grid.ny {
                for i in 0..grid.nx {
                    let want = field[grid.idx(i, j, k)] as f32 as f64;
                    let got = g.values[i + grid.nx * (j + grid.ny * k)];
                    assert_eq!(got, want, "voxel ({i},{j},{k})");
                }
            }
        }
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn vector_field_becomes_four_component_grids() {
        let grid = make_grid(4, 3, 2);
        let mut u = vec![crate::Vec3::ZERO; grid.n()];
        for k in 0..grid.nz {
            for j in 0..grid.ny {
                for i in 0..grid.nx {
                    u[grid.idx(i, j, k)] = crate::Vec3::new(i as Scalar, j as Scalar, k as Scalar);
                }
            }
        }
        let path = scratch("vector");
        write(&path, &grid, &[OutputField::vector("U", &u)], Precision::F32).expect("write");

        let bytes = std::fs::read(&path).expect("read back");
        let grids = read_all(&bytes);
        let names: Vec<&str> = grids.iter().map(|g| g.name.as_str()).collect();
        assert_eq!(names, ["U.x", "U.y", "U.z", "U.mag"]);

        let gx = &grids[0];
        let gmag = &grids[3];
        for k in 0..grid.nz {
            for j in 0..grid.ny {
                for i in 0..grid.nx {
                    let idx = i + grid.nx * (j + grid.ny * k);
                    assert_eq!(gx.values[idx], i as f64);
                    let want_mag = (u[grid.idx(i, j, k)].mag() as f32) as f64;
                    assert_eq!(gmag.values[idx], want_mag);
                }
            }
        }
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn transform_round_trips_origin_and_spacing() {
        let grid = make_grid(6, 5, 4);
        let field = ramp_field(&grid);
        let path = scratch("transform");
        write(&path, &grid, &[OutputField::scalar("p", &field)], Precision::F32).expect("write");

        let bytes = std::fs::read(&path).expect("read back");
        let g = &read_all(&bytes)[0];
        assert_eq!(g.voxel_size, [0.1f64, 0.2, 0.05]);
        assert_eq!(g.origin, [1.5f64, -2.0, 0.25]);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn rejects_a_field_length_that_does_not_match_the_grid() {
        let grid = make_grid(3, 3, 3);
        let field = vec![0.0 as Scalar; 5];
        let err = write("x.vdb", &grid, &[OutputField::scalar("bad", &field)], Precision::F32).unwrap_err();
        assert!(err.to_string().contains("bad"));
    }

    #[test]
    fn rejects_empty_dims_and_empty_field_list() {
        let field = vec![1.0 as Scalar; 1];
        let empty_grid = UniformGrid {
            nx: 0, ny: 1, nz: 1, origin: crate::Vec3::ZERO, spacing: crate::Vec3::new(1.0, 1.0, 1.0),
        };
        assert!(write("x.vdb", &empty_grid, &[OutputField::scalar("a", &field)], Precision::F32).is_err());

        let grid = make_grid(2, 2, 2);
        assert!(write("x.vdb", &grid, &[], Precision::F32).is_err());
    }

    // ========================================================================
    //  SPEC-LIT S45 - the `_HalfFloat` save preference
    // ========================================================================

    /// S45.3: the type-name suffix and the metadata bool, in BOTH
    /// directions - because "fp32 carries no such entry" is what keeps every
    /// `.vdb` this crate wrote before S45 byte-identical.
    #[test]
    fn the_half_float_preference_is_the_suffix_and_the_metadata_bool() {
        let grid = make_grid(9, 5, 4);
        let field = ramp_field(&grid);

        let p32 = scratch("half_off");
        write(&p32, &grid, &[OutputField::scalar("T", &field)], Precision::F32).expect("write");
        let g32 = read_all(&std::fs::read(&p32).unwrap()).remove(0);
        assert_eq!(g32.grid_type, "Tree_float_5_4_3");
        assert_eq!(g32.precision, Precision::F32);
        assert_eq!(g32.half_metadata, None, "fp32 must carry no is_saved_as_half_float entry");

        let p16 = scratch("half_on");
        write(&p16, &grid, &[OutputField::scalar("T", &field)], Precision::F16).expect("write");
        let g16 = read_all(&std::fs::read(&p16).unwrap()).remove(0);
        assert_eq!(g16.grid_type, "Tree_float_5_4_3_HalfFloat");
        assert_eq!(g16.precision, Precision::F16);
        assert_eq!(g16.half_metadata, Some(true));
        assert_eq!(g16.class_metadata.as_deref(), Some("fog volume"), "class survives");
        assert_eq!((g16.nx, g16.ny, g16.nz), (grid.nx, grid.ny, grid.nz));

        let _ = std::fs::remove_file(&p32);
        let _ = std::fs::remove_file(&p16);
    }

    /// **The test that catches halving only the leaves.** SPEC-LIT S45.1(2):
    /// `toHalf` reaches an InternalNode's own `NUM_VALUES` array through
    /// `writeTopology`, not only the LeafNode buffers through `writeBuffers`.
    /// Both arrays are all-zero here, so nothing about the VALUES can tell
    /// the two apart - only the total length can, and this is the length,
    /// derived rather than recorded:
    ///
    /// ```text
    /// shrink = 2 * (n_leaf*512 + n_lower*4096 + n_upper*32768)
    /// growth = 10  ("_HalfFloat" appended to the type string)
    ///        + 39  (the is_saved_as_half_float BoolMetadata entry:
    ///               4+22 name, 4+4 type, 4 size, 1 value)
    /// ```
    ///
    /// A writer that halved only the leaf buffers would come out
    /// `2*(n_lower*4096 + n_upper*32768)` bytes too large and fail here.
    #[test]
    fn fp16_shrinks_the_internal_node_arrays_as_well_as_the_leaves() {
        // A Lower node spans 128 cells per axis, so a grid has to be LONGER
        // than that for the Lower term to be non-degenerate - 37x21x17
        // (this file's other multi-node case) fits inside a single one, and
        // the first version of this test asserted `n_lower > 1` on it and
        // failed. That assert stays, below, for exactly that reason.
        let grid = make_grid(130, 9, 9);
        let field = ramp_field(&grid);

        let p32 = scratch("size32");
        let p16 = scratch("size16");
        write(&p32, &grid, &[OutputField::scalar("T", &field)], Precision::F32).expect("write");
        write(&p16, &grid, &[OutputField::scalar("T", &field)], Precision::F16).expect("write");
        let n32 = std::fs::metadata(&p32).unwrap().len() as i64;
        let n16 = std::fs::metadata(&p16).unwrap().len() as i64;

        let (n_leaf, n_lower, n_upper) = node_counts(&grid);
        let prod = |a: [usize; 3]| (a[0] * a[1] * a[2]) as i64;
        let leaf_values = prod(n_leaf) * 512;
        let lower_values = prod(n_lower) * (1 << (3 * LOWER_LOG2DIM));
        let upper_values = prod(n_upper) * (1 << (3 * UPPER_LOG2DIM));
        assert!(prod(n_lower) > 1, "the grid must span more than one Lower node");

        let shrink = 2 * (leaf_values + lower_values + upper_values);
        let growth = 10 + 39;
        assert_eq!(
            n32 - n16,
            shrink - growth,
            "fp16 must halve the LeafNode buffers ({leaf_values} values), the Lower              InternalNode arrays ({lower_values}) and the Upper ones ({upper_values}) -              halving only the leaves would leave the file {} bytes too large",
            2 * (lower_values + upper_values)
        );

        let _ = std::fs::remove_file(&p32);
        let _ = std::fs::remove_file(&p16);
    }

    /// Every voxel comes back at half precision - and the grid geometry
    /// comes back EXACTLY, because the transform is `f64` either way.
    #[test]
    fn fp16_round_trips_every_voxel_at_half_precision() {
        let grid = make_grid(11, 9, 6);
        let field = ramp_field(&grid);
        let path = scratch("half_round");
        write(&path, &grid, &[OutputField::scalar("T", &field)], Precision::F16).expect("write");

        let g = read_all(&std::fs::read(&path).unwrap()).remove(0);
        assert_eq!(g.voxel_size, [0.1f64, 0.2, 0.05]);
        assert_eq!(g.origin, [1.5f64, -2.0, 0.25]);
        for k in 0..grid.nz {
            for j in 0..grid.ny {
                for i in 0..grid.nx {
                    let v = field[grid.idx(i, j, k)] as f32;
                    let want = crate::io::nvdb::f16_bits_to_f32(f32_to_f16_bits(v)) as f64;
                    let got = g.values[i + grid.nx * (j + grid.ny * k)];
                    assert_eq!(got, want, "voxel ({i},{j},{k})");
                }
            }
        }
        let _ = std::fs::remove_file(&path);
    }

    /// SPEC-LIT S45.2: `vdb`'s fp16 voxels are `nvdb::f32_to_f16_bits`'s
    /// output, not a second binary16 implementation living in this file.
    ///
    /// The sweep is the part that makes this a test rather than a
    /// restatement: `+-0` (told apart by BITS, since `-0.0 == 0.0`),
    /// subnormals below `2^-14`, the largest finite half, the first value
    /// that overflows to infinity, and a round-to-nearest-EVEN tie.
    #[test]
    fn the_half_conversion_is_nvdb_s_own() {
        let sweep: [f32; 12] = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            1.0009766,      // 1 + 2^-10, exactly representable
            1.0004883,      // a tie, halfway between 1 and 1 + 2^-10
            6.0e-8,         // below the smallest subnormal -> 0
            6.0e-5,         // subnormal
            65504.0,        // the largest finite half
            65520.0,        // the first value that rounds to infinity
            -65520.0,
            2.7182817, // an ordinary value with a full 24-bit mantissa
        ];
        let grid = UniformGrid {
            nx: sweep.len(), ny: 1, nz: 1,
            origin: crate::Vec3::ZERO,
            spacing: crate::Vec3::new(1.0, 1.0, 1.0),
        };
        let field: Vec<Scalar> = sweep.iter().map(|&v| v as Scalar).collect();
        let path = scratch("half_sweep");
        write(&path, &grid, &[OutputField::scalar("s", &field)], Precision::F16).expect("write");

        let g = read_all(&std::fs::read(&path).unwrap()).remove(0);
        for (i, &v) in sweep.iter().enumerate() {
            let want = crate::io::nvdb::f16_bits_to_f32(f32_to_f16_bits(v as Scalar as f32)) as f64;
            assert_eq!(
                g.values[i].to_bits(),
                want.to_bits(),
                "{v} must round exactly as nvdb::f32_to_f16_bits rounds it"
            );
        }
        // The sign of zero is a real bit and the writer must not lose it.
        assert_eq!(g.values[0].to_bits(), 0.0f64.to_bits());
        assert_eq!(g.values[1].to_bits(), (-0.0f64).to_bits());
        assert!(g.values[9].is_infinite() && g.values[9] > 0.0, "65520 overflows to +inf");
        assert!(g.values[10].is_infinite() && g.values[10] < 0.0);

        let _ = std::fs::remove_file(&path);
    }
}
