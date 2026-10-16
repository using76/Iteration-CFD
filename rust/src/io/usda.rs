// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

//! Hand-written USDA scene emitter (UsdVolVolume + boundary UsdGeomMesh
//! with per-patch UsdGeomSubset). No OpenUSD dependency.
//!
//! Written from `docs/05-io-redesign.md`:
//!   * section 4.5 ("Omniverse / Isaac Sim") for the minimal `#usda 1.0`
//!     schema, the verbatim `Volume` + `OpenVDBAsset` example this module
//!     reproduces the shape of, and the decision that the USD scene is a
//!     thin pointer at `.vdb`/`.vtu` plus a renderable boundary surface -
//!     never a translation of the volume mesh into USD geometry;
//!   * section 4.2 ("지오메트리") for the patch-identity convention this
//!     format is chosen to carry: `UsdGeomSubset`, `familyName =
//!     "materialBind"`, `familyType = "partition"` - literally "every
//!     boundary face belongs to exactly one patch".
//!
//! No library: USD's ASCII form (`.usda`) is a documented text grammar, not
//! a binary one, so every prim below is one `format!` call. This is the
//! approach section 4.5 recommends explicitly ("Rust `format!`으로
//! 충분합니다").
//!
//! **Known approximation** - the synthesised square is now only
//! [`boundary_surface`]'s fallback, for a caller that holds no raw mesh
//! (`io/writer.rs`): per boundary face it reconstructs a planar
//! quadrilateral centred at the face centroid, orthogonal to the face
//! normal, with area equal to `b_mag_sf`, fanned into two triangles - a
//! correct bounding box and a plausible preview surface, but not the exact
//! patch outline. When the raw mesh IS available, [`boundary_surface_loops`]
//! fans each boundary face's true point loop from its first vertex instead
//! (SPEC-LIT section 117.12, docs/17 VIO-07): the polyMesh order keeps each
//! face's right-handed normal out of the fluid, a planar convex face's fan
//! area equals `|b_sf|`, and a non-convex face may fold under a fan -
//! accepted, the surface is for display and is not repaired. The per-patch
//! `UsdGeomSubset` partition stays exact in both paths.
//!
//! No GPL-licensed source was consulted.

use std::path::Path;

use crate::error::{Error, IoContext, Result};
use crate::mesh::HostMesh;
use crate::types::{to_vec3, DVec3};
use crate::{Scalar, Vec3};

// ==========================================================================
//  Types
// ==========================================================================

/// One `.vdb` file bound to a USD `timeSamples` time, in seconds.
#[derive(Debug, Clone)]
pub struct VdbFrame {
    pub time: f64,
    /// Path written verbatim inside `@...@` - relative paths are the norm
    /// (`./plume_0040.vdb`), matching section 4.5's example.
    pub path: String,
}

/// Where a volume's voxel data lives on disk.
#[derive(Debug, Clone)]
pub enum VdbSource {
    /// A single, static grid: `asset filePath = @...@`.
    Single(String),
    /// A time-varying grid: `asset filePath.timeSamples = { t: @...@, ... }`.
    /// Must be non-empty and is written in the given order (callers sort by
    /// time first).
    Series(Vec<VdbFrame>),
}

/// One `UsdVolVolume` + child `OpenVDBAsset`, per section 4.5's example.
#[derive(Debug, Clone)]
pub struct VolumeAsset {
    /// Prim name, e.g. `"plume"`. Must be a valid USD identifier (ASCII
    /// letters, digits, `_`, not starting with a digit) - not re-validated
    /// here, callers control it.
    pub name: String,
    /// Both the `OpenVDBAsset` child's name and its `fieldName` token, and
    /// what `rel field:<field_name>` points at.
    pub field_name: String,
    pub vdb: VdbSource,
    /// `float3[] extent` on the `Volume` prim. Omitted from the prim when
    /// `None` - USD does not require it, it is only a render-time hint.
    pub extent: Option<(Vec3, Vec3)>,
}

/// One `UsdGeomSubset` worth of triangle faces, all belonging to the same
/// mesh patch.
#[derive(Debug, Clone)]
pub struct PatchSubset {
    pub name: String,
    /// Indices into the boundary mesh's flattened face list (i.e. into
    /// `SurfaceOut::face_vertex_counts`), NOT into `points`.
    pub face_indices: Vec<u32>,
}

/// A triangulated boundary surface ready to become one `UsdGeomMesh`, as
/// produced by [`boundary_surface`].
#[derive(Debug, Clone)]
pub struct SurfaceOut {
    pub points: Vec<Vec3>,
    /// One entry per emitted face; always `3` (the emitter only ever fans
    /// into triangles).
    pub face_vertex_counts: Vec<u32>,
    /// Flattened, `sum(face_vertex_counts)` long.
    pub face_vertex_indices: Vec<u32>,
    pub extent: (Vec3, Vec3),
    /// One [`PatchSubset`] per `HostMesh` patch, in `HostMesh::patches`
    /// order. Every face index in `0..face_vertex_counts.len()` appears in
    /// exactly one subset's `face_indices` - the section 4.2 invariant.
    pub patches: Vec<PatchSubset>,
}

/// Everything one `write_usda_scene` call emits: any number of volume
/// assets plus (optionally) one boundary surface.
#[derive(Debug, Clone, Default)]
pub struct SceneSpec {
    pub volumes: Vec<VolumeAsset>,
    pub boundary: Option<SurfaceOut>,
}

// ==========================================================================
//  Boundary surface construction
// ==========================================================================

/// Build the renderable boundary surface from a [`HostMesh`] alone - the
/// fallback for a caller that holds no raw mesh: one synthesised quad per
/// boundary face (see the module doc's "Known approximation"), fanned into
/// two triangles, grouped into one [`PatchSubset`] per patch. Prefer
/// [`boundary_surface_loops`] when the raw mesh is at hand.
pub fn boundary_surface(m: &HostMesh) -> SurfaceOut {
    let n_bf = m.n_boundary_faces.min(m.b_cf.len()).min(m.b_sf.len());
    let mut points = Vec::with_capacity(n_bf * 4);
    let mut face_vertex_counts = Vec::with_capacity(n_bf * 2);
    let mut face_vertex_indices = Vec::with_capacity(n_bf * 6);
    let mut patch_faces: Vec<Vec<u32>> = vec![Vec::new(); m.patches.len()];

    for bf in 0..n_bf {
        // The emitted centroid is the WORLD position (SPEC-LIT 118.7); the
        // frame below is built from `sf` alone, and a translation moves no
        // normal, so nothing else here shifts.
        let c = to_vec3(m.world(m.b_cf[bf]));
        let sf = m.b_sf[bf];
        let mag = m.b_mag_sf.get(bf).copied().unwrap_or_else(|| sf.mag());
        // Degenerate face (zero area): nothing sane to draw, skip it rather
        // than emit a zero-size quad that would corrupt the extent.
        if !(mag > 0.0) {
            continue;
        }
        let n = sf / mag;

        // An orthonormal in-plane frame (u, v) with n = u x v. `reference`
        // just needs to not be near-parallel to n; the 0.9 threshold keeps
        // the Gram-Schmidt subtraction well conditioned for every n.
        let reference = if n.x.abs() < 0.9 {
            Vec3::new(1.0, 0.0, 0.0)
        } else {
            Vec3::new(0.0, 1.0, 0.0)
        };
        let u = (reference - n * n.dot(reference)).normalised();
        let v = n.cross(u);

        // Square quad of area `mag`, centred on the true face centroid.
        let half = mag.sqrt() * 0.5;
        let p0 = c - u * half - v * half;
        let p1 = c + u * half - v * half;
        let p2 = c + u * half + v * half;
        let p3 = c - u * half + v * half;

        let base = points.len() as u32;
        points.push(p0);
        points.push(p1);
        points.push(p2);
        points.push(p3);

        // Fan the quad from vertex 0, winding so the two triangles' normals
        // agree with `n` (checked in tests): (p1-p0) x (p3-p0) = 4*half^2*n.
        let tri0 = face_vertex_counts.len() as u32;
        face_vertex_counts.push(3);
        face_vertex_indices.extend_from_slice(&[base, base + 1, base + 2]);

        let tri1 = face_vertex_counts.len() as u32;
        face_vertex_counts.push(3);
        face_vertex_indices.extend_from_slice(&[base, base + 2, base + 3]);

        let patch = m.b_patch.get(bf).copied().unwrap_or(-1);
        if patch >= 0 {
            if let Some(faces) = patch_faces.get_mut(patch as usize) {
                faces.push(tri0);
                faces.push(tri1);
            }
        }
    }

    let extent = bbox(&points);
    let patches = m
        .patches
        .iter()
        .enumerate()
        .map(|(i, p)| PatchSubset {
            name: p.name.clone(),
            face_indices: patch_faces.get(i).cloned().unwrap_or_default(),
        })
        .collect();

    SurfaceOut { points, face_vertex_counts, face_vertex_indices, extent, patches }
}

/// SPEC-LIT section 117.12 (docs/17 VIO-07): the boundary surface from the
/// TRUE face polygons. Each boundary face's point loop `p_0 ... p_{m-1}` is
/// fanned from its first vertex into `(p_0, p_k, p_{k+1})` for
/// `k = 1 ... m-2`; the polyMesh order makes each face's right-handed normal
/// point out of the fluid and the fan keeps that order, so a planar convex
/// face's fan area equals `|b_sf|`. A non-convex face can fold under a fan;
/// the surface is a display surface, so that is accepted and not repaired.
/// The points are compacted to the ones the boundary faces reference, in
/// ascending raw point index. A malformed raw mesh is refused by name, not
/// assembled into a wrong surface; the `UsdGeomSubset` partition stays
/// exact (one [`PatchSubset`] per patch, every triangle in exactly one).
pub fn boundary_surface_loops(raw: &crate::io::polymesh::PolyMeshRaw) -> Result<SurfaceOut> {
    let err =
        |msg: String| crate::error::Error::Mesh(format!("boundary_surface_loops: {msg}"));

    let n_if = raw.neighbour.len();
    if n_if > raw.faces.len() {
        return Err(err(format!(
            "neighbour has {} entries, more than the {} faces",
            n_if,
            raw.faces.len()
        )));
    }
    let n_bf = raw.faces.len() - n_if;

    // The patches must tile the boundary faces: every boundary face in
    // exactly one patch, no patch range past the last boundary face.
    let mut patch_of = vec![usize::MAX; n_bf];
    for (pi, p) in raw.patches.iter().enumerate() {
        let end = p.start.checked_add(p.size).unwrap_or(usize::MAX);
        if end > n_bf {
            return Err(err(format!(
                "the patches do not tile the boundary: patch '{}' covers boundary faces {}..{}, past the last boundary face {}",
                p.name,
                p.start,
                end,
                n_bf.saturating_sub(1)
            )));
        }
        for bf in p.start..end {
            if patch_of[bf] != usize::MAX {
                return Err(err(format!(
                    "the patches do not tile the boundary: boundary face {} is in both patch '{}' and patch '{}'",
                    bf,
                    raw.patches[patch_of[bf]].name,
                    p.name
                )));
            }
            patch_of[bf] = pi;
        }
    }
    for bf in 0..n_bf {
        if patch_of[bf] == usize::MAX {
            return Err(err(format!(
                "the patches do not tile the boundary: boundary face {} is in no patch",
                bf
            )));
        }
    }

    // Pass 1: every boundary face needs at least 3 points, every point
    // label in range - and mark which raw points the boundary references.
    let mut used = vec![false; raw.points.len()];
    for bf in 0..n_bf {
        let face = &raw.faces[n_if + bf];
        if face.len() < 3 {
            return Err(err(format!(
                "boundary face {} has fewer than 3 points ({})",
                bf,
                face.len()
            )));
        }
        for &label in face {
            if label < 0 || label as usize >= raw.points.len() {
                return Err(err(format!(
                    "boundary face {} references point {}, but the mesh has {} points",
                    bf,
                    label,
                    raw.points.len()
                )));
            }
            used[label as usize] = true;
        }
    }

    // The compacted points: exactly the referenced raw points, ascending
    // raw index, each converted to `Vec3` once (SPEC-LIT §118.2).
    let mut new_of_old = vec![u32::MAX; raw.points.len()];
    let mut points: Vec<Vec3> = Vec::new();
    for (i, &u) in used.iter().enumerate() {
        if u {
            new_of_old[i] = points.len() as u32;
            points.push(crate::types::to_vec3(raw.points[i]));
        }
    }

    // Pass 2: fan each face from its first vertex, in the face's own point
    // order - the triangles (p_0, p_k, p_{k+1}) for k = 1 ... m-2.
    let mut face_vertex_counts: Vec<u32> = Vec::new();
    let mut face_vertex_indices: Vec<u32> = Vec::new();
    let mut patch_faces: Vec<Vec<u32>> = vec![Vec::new(); raw.patches.len()];
    for bf in 0..n_bf {
        let face = &raw.faces[n_if + bf];
        let base_tri = face_vertex_counts.len() as u32;
        let first = new_of_old[face[0] as usize];
        for k in 1..face.len() - 1 {
            face_vertex_counts.push(3);
            face_vertex_indices.extend_from_slice(&[
                first,
                new_of_old[face[k] as usize],
                new_of_old[face[k + 1] as usize],
            ]);
        }
        let n_new = (face.len() - 2) as u32;
        for t in 0..n_new {
            patch_faces[patch_of[bf]].push(base_tri + t);
        }
    }

    let extent = bbox(&points);
    let patches = raw
        .patches
        .iter()
        .enumerate()
        .map(|(i, p)| PatchSubset {
            name: p.name.clone(),
            face_indices: patch_faces[i].clone(),
        })
        .collect();

    Ok(SurfaceOut {
        points,
        face_vertex_counts,
        face_vertex_indices,
        extent,
        patches,
    })
}

fn bbox(points: &[Vec3]) -> (Vec3, Vec3) {
    if points.is_empty() {
        return (Vec3::ZERO, Vec3::ZERO);
    }
    let mut lo = points[0];
    let mut hi = points[0];
    for p in &points[1..] {
        lo.x = lo.x.min(p.x);
        lo.y = lo.y.min(p.y);
        lo.z = lo.z.min(p.z);
        hi.x = hi.x.max(p.x);
        hi.y = hi.y.max(p.y);
        hi.z = hi.z.max(p.z);
    }
    (lo, hi)
}

// ==========================================================================
//  USDA text emission
// ==========================================================================

/// Write a complete `.usda` scene: the header from section 4.5 (`#usda
/// 1.0`, `metersPerUnit = 1`, `upAxis = "Z"`), one `Volume` prim per
/// `spec.volumes` entry, and (if present) one `Mesh` prim for
/// `spec.boundary` with one `GeomSubset` per patch.
pub fn write_usda_scene(path: &Path, spec: &SceneSpec) -> Result<()> {
    let mut s = String::new();
    s.push_str("#usda 1.0\n");
    s.push_str("(\n");
    s.push_str("    metersPerUnit = 1\n");
    s.push_str("    upAxis = \"Z\"\n");
    s.push_str(")\n");

    for vol in &spec.volumes {
        s.push('\n');
        write_volume(&mut s, vol);
    }

    if let Some(surf) = &spec.boundary {
        s.push('\n');
        write_boundary_mesh(&mut s, surf);
    }

    if let Some(dir) = path.parent() {
        if !dir.as_os_str().is_empty() {
            std::fs::create_dir_all(dir).path(dir)?;
        }
    }
    std::fs::write(path, s.as_bytes()).path(path)
}

fn write_volume(s: &mut String, vol: &VolumeAsset) {
    s.push_str(&format!("def Volume \"{}\"\n", vol.name));
    s.push_str("{\n");
    if let Some((lo, hi)) = vol.extent {
        s.push_str(&format!("    float3[] extent = [{}, {}]\n", fmt_v3(lo), fmt_v3(hi)));
    }
    s.push_str(&format!(
        "    rel field:{field} = </{name}/{field}>\n",
        name = vol.name,
        field = vol.field_name
    ));
    s.push('\n');
    s.push_str(&format!("    def OpenVDBAsset \"{}\"\n", vol.field_name));
    s.push_str("    {\n");
    match &vol.vdb {
        VdbSource::Single(p) => {
            s.push_str(&format!("        asset filePath = @{}@\n", p));
        }
        VdbSource::Series(frames) => {
            s.push_str("        asset filePath.timeSamples = {\n");
            for f in frames {
                s.push_str(&format!("            {}: @{}@,\n", fmt_time(f.time), f.path));
            }
            s.push_str("        }\n");
        }
    }
    s.push_str(&format!("        token fieldName = \"{}\"\n", vol.field_name));
    s.push_str("    }\n");
    s.push_str("}\n");
}

fn write_boundary_mesh(s: &mut String, surf: &SurfaceOut) {
    s.push_str("def Mesh \"boundary\"\n");
    s.push_str("{\n");
    // Family-level metadata (UsdGeomSubset::SetFamilyType lives on the
    // owning prim, keyed by family name) - section 4.2's "familyType =
    // partition" invariant, spelled the way the schema actually stores it.
    s.push_str("    uniform token subsetFamily:materialBind:familyType = \"partition\"\n");
    s.push_str(&format!(
        "    point3f[] points = [{}]\n",
        surf.points.iter().map(|p| fmt_v3(*p)).collect::<Vec<_>>().join(", ")
    ));
    s.push_str(&format!(
        "    int[] faceVertexCounts = [{}]\n",
        join_ints(&surf.face_vertex_counts)
    ));
    s.push_str(&format!(
        "    int[] faceVertexIndices = [{}]\n",
        join_ints(&surf.face_vertex_indices)
    ));
    s.push_str(&format!(
        "    float3[] extent = [{}, {}]\n",
        fmt_v3(surf.extent.0),
        fmt_v3(surf.extent.1)
    ));

    for patch in &surf.patches {
        s.push('\n');
        s.push_str(&format!("    def GeomSubset \"{}\"\n", patch.name));
        s.push_str("    {\n");
        s.push_str("        uniform token elementType = \"face\"\n");
        s.push_str("        uniform token familyName = \"materialBind\"\n");
        s.push_str(&format!("        int[] indices = [{}]\n", join_ints(&patch.face_indices)));
        s.push_str("    }\n");
    }
    s.push_str("}\n");
}

fn join_ints(v: &[u32]) -> String {
    v.iter().map(|i| i.to_string()).collect::<Vec<_>>().join(", ")
}

/// `{:?}` on `f64` always prints a decimal point (`1.0`, not `1`), which is
/// what makes the value unambiguously a USD `float`/`double`, and uses the
/// shortest round-tripping representation otherwise.
fn fmt_f(v: Scalar) -> String {
    format!("{:?}", v as f64)
}

fn fmt_v3(v: Vec3) -> String {
    format!("({}, {}, {})", fmt_f(v.x), fmt_f(v.y), fmt_f(v.z))
}

fn fmt_time(t: f64) -> String {
    // USD timeSamples keys are plain numbers, not quoted.
    if t.fract() == 0.0 {
        format!("{}", t as i64)
    } else {
        format!("{:?}", t)
    }
}

// ==========================================================================
//  The Isaac scene (SPEC-LIT 117.9, VIO-09)
// ==========================================================================

/// Which renderer the Isaac scene addresses (117.9's table). `Index` is
/// NVIDIA IndeX, the RTX volume path, on a `def Volume` + `OpenVDBAsset`;
/// `Pathtrace` is the RTX path tracer, on a `Mesh` box carrying
/// `primvars:isVolume` bound to an `OmniVolumeDensity` material.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum IsaacProfile {
    #[default]
    Index,
    Pathtrace,
}

impl IsaacProfile {
    /// The case key's spelling (117.9's table), for refusals and banners.
    pub fn name(self) -> &'static str {
        match self {
            Self::Index => "index",
            Self::Pathtrace => "pathtrace",
        }
    }
}

/// The stage metadata a viewer needs before it reads any prim (117.9's
/// table): the up axis and the unit scale, `"Z"` / `1.0` as `usda.rs` has
/// always written them.
#[derive(Debug, Clone, PartialEq)]
pub struct StageMeta {
    pub up_axis: String,
    pub meters_per_unit: f64,
}

impl Default for StageMeta {
    fn default() -> Self {
        Self { up_axis: "Z".to_string(), meters_per_unit: 1.0 }
    }
}

/// Everything one Isaac scene names: the profile, the stage metadata, the
/// `.vdb` the density lives in (a Single asset or a `timeSamples` Series),
/// the extent box of the volume, and the boundary surface, if the run has
/// one to show. Both profiles reference the `.vdb` only - `.nvdb` is never
/// referenced, and [`isaac_scene_text`] refuses such a path by name.
#[derive(Debug, Clone)]
pub struct IsaacScene {
    pub profile: IsaacProfile,
    pub stage: StageMeta,
    pub vdb: VdbSource,
    /// `float3[] extent` - the box of the volume, in world metres.
    pub extent: (DVec3, DVec3),
    pub boundary: Option<SurfaceOut>,
}

/// `{:?}` on a plain `f64`, the Isaac scene's number style (`1.0`,
/// `0.5`): a decimal point, so the value is unambiguously a float.
fn fmt_f64(v: f64) -> String {
    format!("{v:?}")
}

fn fmt_d3(v: DVec3) -> String {
    format!("({}, {}, {})", fmt_f64(v.x), fmt_f64(v.y), fmt_f64(v.z))
}

/// One `asset <attr>` binding, single or `timeSamples`, at `indent` spaces.
/// A Series frames' times go through the same [`fmt_time`] the plain scene
/// uses; the frames are written in the order given.
fn write_isaac_asset(s: &mut String, indent: usize, attr: &str, vdb: &VdbSource) {
    let pad = " ".repeat(indent);
    match vdb {
        VdbSource::Single(p) => s.push_str(&format!("{pad}asset {attr} = @{p}@\n")),
        VdbSource::Series(frames) => {
            s.push_str(&format!("{pad}asset {attr}.timeSamples = {{\n"));
            for f in frames {
                s.push_str(&format!(
                    "{pad}    {t}: @{p}@,\n",
                    pad = pad,
                    t = fmt_time(f.time),
                    p = f.path
                ));
            }
            s.push_str(&format!("{pad}}}\n"));
        }
    }
}

/// The `index` profile's volume prim (117.9's table): `def Volume
/// "density"` with `rel field:density` to a child `def OpenVDBAsset
/// "density"`, `token fieldName = "density"`, `asset filePath` = the
/// `.vdb` - the shape `write_usda_scene` writes, under `World`.
fn write_isaac_index_prim(s: &mut String, scene: &IsaacScene) {
    s.push_str("    def Volume \"density\"\n");
    s.push_str("    {\n");
    s.push_str(&format!(
        "        float3[] extent = [{}, {}]\n",
        fmt_d3(scene.extent.0),
        fmt_d3(scene.extent.1)
    ));
    s.push_str("        rel field:density = </World/density/density>\n");
    s.push('\n');
    s.push_str("        def OpenVDBAsset \"density\"\n");
    s.push_str("        {\n");
    write_isaac_asset(s, 12, "filePath", &scene.vdb);
    s.push_str("            token fieldName = \"density\"\n");
    s.push_str("        }\n");
    s.push_str("    }\n");
}

/// The `pathtrace` profile's prims (117.9's table): `def Mesh "density"`,
/// the box of the extent, with `primvars:isVolume` true and
/// `MaterialBindingAPI` applied, bound to a Material whose Shader names
/// `OmniVolumeDensity.mdl` (sub-identifier `OmniVolumeDensity`) with input
/// `volume_density_texture` = the `.vdb`, and `volume_density_scale` 1.0 so
/// sigma is used as written. The eight points are the extent box's corners
/// in the fixed order the face indices below assume; the handoff's working
/// path-traced scene is the topology's reference (117.9).
fn write_isaac_pathtrace_prims(s: &mut String, scene: &IsaacScene) {
    let (lo, hi) = (scene.extent.0, scene.extent.1);
    let pts = [
        (lo.x, lo.y, lo.z),
        (hi.x, lo.y, lo.z),
        (lo.x, lo.y, hi.z),
        (hi.x, lo.y, hi.z),
        (lo.x, hi.y, lo.z),
        (hi.x, hi.y, lo.z),
        (hi.x, hi.y, hi.z),
        (lo.x, hi.y, hi.z),
    ];
    let points = pts
        .iter()
        .map(|(x, y, z)| format!("({}, {}, {})", fmt_f64(*x), fmt_f64(*y), fmt_f64(*z)))
        .collect::<Vec<_>>()
        .join(", ");
    s.push_str("    def Mesh \"density\" (\n");
    s.push_str("        prepend apiSchemas = [\"MaterialBindingAPI\"]\n");
    s.push_str("    )\n");
    s.push_str("    {\n");
    s.push_str(&format!(
        "        float3[] extent = [{}, {}]\n",
        fmt_d3(lo),
        fmt_d3(hi)
    ));
    s.push_str("        int[] faceVertexCounts = [4, 4, 4, 4, 4, 4]\n");
    s.push_str(
        "        int[] faceVertexIndices = [0, 1, 3, 2, 0, 4, 5, 1, 1, 5, 6, 3, 2, 3, 6, 7, 0, 2, 7, 4, 4, 7, 6, 5]\n",
    );
    s.push_str(&format!("        point3f[] points = [{points}]\n"));
    s.push_str("        bool primvars:isVolume = 1\n");
    s.push_str("        uniform token subdivisionScheme = \"none\"\n");
    s.push_str("        rel material:binding = </World/Looks/density_material>\n");
    s.push_str("    }\n");
    s.push('\n');
    s.push_str("    def Scope \"Looks\"\n");
    s.push_str("    {\n");
    s.push_str("        def Material \"density_material\"\n");
    s.push_str("        {\n");
    s.push_str(
        "            token outputs:mdl:displacement.connect = </World/Looks/density_material/Shader.outputs:out>\n",
    );
    s.push_str(
        "            token outputs:mdl:surface.connect = </World/Looks/density_material/Shader.outputs:out>\n",
    );
    s.push_str(
        "            token outputs:mdl:volume.connect = </World/Looks/density_material/Shader.outputs:out>\n",
    );
    s.push('\n');
    s.push_str("            def Shader \"Shader\"\n");
    s.push_str("            {\n");
    s.push_str("                uniform token info:implementationSource = \"sourceAsset\"\n");
    s.push_str("                uniform asset info:mdl:sourceAsset = @OmniVolumeDensity.mdl@\n");
    s.push_str(
        "                uniform token info:mdl:sourceAsset:subIdentifier = \"OmniVolumeDensity\"\n",
    );
    write_isaac_asset(s, 16, "inputs:volume_density_texture", &scene.vdb);
    s.push_str("                float inputs:volume_density_scale = 1.0\n");
    s.push_str("                token outputs:out\n");
    s.push_str("            }\n");
    s.push_str("        }\n");
    s.push_str("    }\n");
}

/// The Isaac scene text (117.9): `#usda 1.0` with `defaultPrim = "World"`
/// and the stage metadata of [`IsaacScene::stage`], one `def Xform
/// "World"` holding the profile's density prim (plus the Looks material of
/// the pathtrace profile) and, when present, the boundary surface -
/// [`write_boundary_mesh`]'s output with every non-empty line indented by
/// four spaces, after one empty line. Refused by name: a `.vdb` path that
/// ends in `.nvdb` (an Isaac scene never references `.nvdb`, 117.9) and an
/// empty Series.
pub fn isaac_scene_text(scene: &IsaacScene) -> Result<String> {
    let refuse_nvdb = |p: &str| {
        if p.ends_with(".nvdb") {
            return Err(Error::Config(format!(
                "isaac scene: the volume source \"{p}\" ends in .nvdb - an Isaac scene never \
                 references .nvdb (SPEC-LIT 117.9); name the .vdb the run writes"
            )));
        }
        Ok(())
    };
    match &scene.vdb {
        VdbSource::Single(p) => refuse_nvdb(p)?,
        VdbSource::Series(frames) => {
            if frames.is_empty() {
                return Err(Error::Config(
                    "isaac scene: a Series of frames is empty - one frame at least, or a Single"
                        .into(),
                ));
            }
            for f in frames {
                refuse_nvdb(&f.path)?;
            }
        }
    }

    let mut s = String::new();
    s.push_str("#usda 1.0\n");
    s.push_str("(\n");
    s.push_str("    defaultPrim = \"World\"\n");
    s.push_str(&format!("    metersPerUnit = {}\n", fmt_f64(scene.stage.meters_per_unit)));
    s.push_str(&format!("    upAxis = \"{}\"\n", scene.stage.up_axis));
    s.push_str(")\n");
    s.push('\n');
    s.push_str("def Xform \"World\"\n");
    s.push_str("{\n");
    match scene.profile {
        IsaacProfile::Index => write_isaac_index_prim(&mut s, scene),
        IsaacProfile::Pathtrace => write_isaac_pathtrace_prims(&mut s, scene),
    }
    if let Some(surf) = &scene.boundary {
        s.push('\n');
        let mut b = String::new();
        write_boundary_mesh(&mut b, surf);
        for line in b.lines() {
            if line.is_empty() {
                s.push('\n');
            } else {
                s.push_str("    ");
                s.push_str(line);
                s.push('\n');
            }
        }
    }
    s.push_str("}\n");
    Ok(s)
}

/// Write the Isaac scene of [`isaac_scene_text`] to `path`, creating the
/// parent directory like [`write_usda_scene`].
pub fn write_isaac_scene(path: &Path, scene: &IsaacScene) -> Result<()> {
    let s = isaac_scene_text(scene)?;
    if let Some(dir) = path.parent() {
        if !dir.as_os_str().is_empty() {
            std::fs::create_dir_all(dir).path(dir)?;
        }
    }
    std::fs::write(path, s.as_bytes()).path(path)
}

// ==========================================================================
//  Tests
// ==========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::{PatchInfo, PatchKind};

    /// A tiny mesh: two boundary faces, two patches ("inlet", "outlet"),
    /// one boundary face each - axis-aligned unit-area faces so the
    /// reconstructed quads and extent are easy to check by hand.
    fn two_patch_mesh() -> HostMesh {
        let mut m = HostMesh::default();
        m.n_boundary_faces = 2;
        m.b_cf = vec![Vec3::new(0.0, 0.0, 0.0), Vec3::new(2.0, 0.0, 0.0)];
        m.b_sf = vec![Vec3::new(1.0, 0.0, 0.0), Vec3::new(-1.0, 0.0, 0.0)];
        m.b_mag_sf = vec![1.0, 1.0];
        m.b_patch = vec![0, 1];
        m.patches = vec![
            PatchInfo {
                name: "inlet".into(),
                type_name: "patch".into(),
                kind: PatchKind::Generic,
                start: 0,
                size: 1,
                nbr_patch: None,
            },
            PatchInfo {
                name: "outlet".into(),
                type_name: "patch".into(),
                kind: PatchKind::Generic,
                start: 1,
                size: 1,
                nbr_patch: None,
            },
        ];
        m
    }

    #[test]
    fn boundary_surface_partitions_faces_exactly() {
        let m = two_patch_mesh();
        let surf = boundary_surface(&m);

        // 2 boundary faces -> 4 triangle faces (fan of a quad).
        assert_eq!(surf.face_vertex_counts.len(), 4);
        assert!(surf.face_vertex_counts.iter().all(|&c| c == 3));
        assert_eq!(surf.patches.len(), 2);

        // Every face index appears in exactly one subset.
        let n_faces = surf.face_vertex_counts.len();
        let mut seen = vec![0u32; n_faces];
        for p in &surf.patches {
            for &i in &p.face_indices {
                seen[i as usize] += 1;
            }
        }
        assert!(seen.iter().all(|&c| c == 1), "partition violated: {:?}", seen);

        assert_eq!(surf.patches[0].face_indices, vec![0, 1]);
        assert_eq!(surf.patches[1].face_indices, vec![2, 3]);
    }

    #[test]
    fn boundary_surface_extent_matches_points_bbox() {
        let m = two_patch_mesh();
        let surf = boundary_surface(&m);
        let (lo, hi) = bbox(&surf.points);
        assert_eq!(surf.extent.0, lo);
        assert_eq!(surf.extent.1, hi);
        // Both faces have their normal along x, so each quad lies flat in
        // the y-z plane through its centroid: extent.x must span exactly
        // the two centroids (x=0 and x=2), and each quad's own half-width
        // (0.5) shows up in y and z instead.
        assert!((surf.extent.0.x - 0.0).abs() < 1e-6);
        assert!((surf.extent.1.x - 2.0).abs() < 1e-6);
        assert!((surf.extent.0.y - (-0.5)).abs() < 1e-6);
        assert!((surf.extent.1.y - 0.5).abs() < 1e-6);
    }

    #[test]
    fn write_usda_scene_has_header_and_one_subset_per_patch() {
        let m = two_patch_mesh();
        let boundary = boundary_surface(&m);
        let spec = SceneSpec { volumes: vec![], boundary: Some(boundary) };

        let dir = std::env::temp_dir().join("usda_test_header");
        let path = dir.join("scene.usda");
        write_usda_scene(&path, &spec).unwrap();
        let text = std::fs::read_to_string(&path).unwrap();

        assert!(text.starts_with("#usda 1.0\n"));
        assert!(text.contains("metersPerUnit = 1"));
        assert!(text.contains("upAxis = \"Z\""));
        assert_eq!(text.matches("def GeomSubset").count(), 2);
        assert!(text.contains("def GeomSubset \"inlet\""));
        assert!(text.contains("def GeomSubset \"outlet\""));
        assert!(text.contains("familyName = \"materialBind\""));
        assert!(text.contains("familyType = \"partition\""));

        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_dir(&dir);
    }

    #[test]
    fn write_usda_scene_extent_matches_bbox() {
        let m = two_patch_mesh();
        let boundary = boundary_surface(&m);
        let expected_extent = boundary.extent;
        let spec = SceneSpec { volumes: vec![], boundary: Some(boundary) };

        let dir = std::env::temp_dir().join("usda_test_extent");
        let path = dir.join("scene.usda");
        write_usda_scene(&path, &spec).unwrap();
        let text = std::fs::read_to_string(&path).unwrap();

        let expected = format!(
            "float3[] extent = [{}, {}]",
            fmt_v3(expected_extent.0),
            fmt_v3(expected_extent.1)
        );
        assert!(text.contains(&expected), "expected `{}` in:\n{}", expected, text);

        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_dir(&dir);
    }

    #[test]
    fn write_usda_scene_series_emits_time_samples() {
        let vol = VolumeAsset {
            name: "plume".into(),
            field_name: "temperature".into(),
            vdb: VdbSource::Series(vec![
                VdbFrame { time: 0.0, path: "./plume_0000.vdb".into() },
                VdbFrame { time: 1.0, path: "./plume_0001.vdb".into() },
            ]),
            extent: Some((Vec3::new(-7.3, -3.1, 0.0), Vec3::new(7.3, 3.1, 3.0))),
        };
        let spec = SceneSpec { volumes: vec![vol], boundary: None };

        let dir = std::env::temp_dir().join("usda_test_series");
        let path = dir.join("scene.usda");
        write_usda_scene(&path, &spec).unwrap();
        let text = std::fs::read_to_string(&path).unwrap();

        assert!(text.contains("def Volume \"plume\""));
        assert!(text.contains("def OpenVDBAsset \"temperature\""));
        assert!(text.contains("asset filePath.timeSamples = {"));
        assert!(text.contains("0: @./plume_0000.vdb@,"));
        assert!(text.contains("1: @./plume_0001.vdb@,"));
        assert!(text.contains("token fieldName = \"temperature\""));

        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_dir(&dir);
    }

    #[test]
    fn write_usda_scene_single_vdb_has_no_time_samples() {
        let vol = VolumeAsset {
            name: "plume".into(),
            field_name: "temperature".into(),
            vdb: VdbSource::Single("./plume_0040.vdb".into()),
            extent: Some((Vec3::new(-7.3, -3.1, 0.0), Vec3::new(7.3, 3.1, 3.0))),
        };
        let spec = SceneSpec { volumes: vec![vol], boundary: None };

        let dir = std::env::temp_dir().join("usda_test_single");
        let path = dir.join("scene.usda");
        write_usda_scene(&path, &spec).unwrap();
        let text = std::fs::read_to_string(&path).unwrap();

        assert!(text.contains("asset filePath = @./plume_0040.vdb@"));
        assert!(!text.contains("timeSamples"));

        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_dir(&dir);
    }

    // ---- the Isaac scenes (SPEC-LIT 117.9, VIO-09) -----------------------

    /// The scene both profile tests compare against, byte for byte.
    fn isaac(profile: IsaacProfile, vdb: VdbSource) -> IsaacScene {
        IsaacScene {
            profile,
            stage: StageMeta::default(),
            vdb,
            extent: (DVec3::new(0.0, 0.0, 0.0), DVec3::new(1.0, 2.0, 3.0)),
            boundary: None,
        }
    }

    const INDEX_SCENE: &str = "\
#usda 1.0
(
    defaultPrim = \"World\"
    metersPerUnit = 1.0
    upAxis = \"Z\"
)

def Xform \"World\"
{
    def Volume \"density\"
    {
        float3[] extent = [(0.0, 0.0, 0.0), (1.0, 2.0, 3.0)]
        rel field:density = </World/density/density>

        def OpenVDBAsset \"density\"
        {
            asset filePath = @./VDB/s_000000.vdb@
            token fieldName = \"density\"
        }
    }
}
";

    const PATHTRACE_SCENE: &str = "\
#usda 1.0
(
    defaultPrim = \"World\"
    metersPerUnit = 1.0
    upAxis = \"Z\"
)

def Xform \"World\"
{
    def Mesh \"density\" (
        prepend apiSchemas = [\"MaterialBindingAPI\"]
    )
    {
        float3[] extent = [(0.0, 0.0, 0.0), (1.0, 2.0, 3.0)]
        int[] faceVertexCounts = [4, 4, 4, 4, 4, 4]
        int[] faceVertexIndices = [0, 1, 3, 2, 0, 4, 5, 1, 1, 5, 6, 3, 2, 3, 6, 7, 0, 2, 7, 4, 4, 7, 6, 5]
        point3f[] points = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 3.0), (1.0, 0.0, 3.0), (0.0, 2.0, 0.0), (1.0, 2.0, 0.0), (1.0, 2.0, 3.0), (0.0, 2.0, 3.0)]
        bool primvars:isVolume = 1
        uniform token subdivisionScheme = \"none\"
        rel material:binding = </World/Looks/density_material>
    }

    def Scope \"Looks\"
    {
        def Material \"density_material\"
        {
            token outputs:mdl:displacement.connect = </World/Looks/density_material/Shader.outputs:out>
            token outputs:mdl:surface.connect = </World/Looks/density_material/Shader.outputs:out>
            token outputs:mdl:volume.connect = </World/Looks/density_material/Shader.outputs:out>

            def Shader \"Shader\"
            {
                uniform token info:implementationSource = \"sourceAsset\"
                uniform asset info:mdl:sourceAsset = @OmniVolumeDensity.mdl@
                uniform token info:mdl:sourceAsset:subIdentifier = \"OmniVolumeDensity\"
                asset inputs:volume_density_texture = @./VDB/s_000000.vdb@
                float inputs:volume_density_scale = 1.0
                token outputs:out
            }
        }
    }
}
";

    #[test]
    fn isaac_index_scene() {
        let scene = isaac(IsaacProfile::Index, VdbSource::Single("./VDB/s_000000.vdb".into()));
        assert_eq!(isaac_scene_text(&scene).unwrap(), INDEX_SCENE);

        // A Series of two frames replaces the single asset line by the
        // timeSamples block, same indent, frames in order.
        let mut scene = scene;
        scene.vdb = VdbSource::Series(vec![
            VdbFrame { time: 0.0, path: "./VDB/s_000000.vdb".into() },
            VdbFrame { time: 0.5, path: "./VDB/s_000001.vdb".into() },
        ]);
        let text = isaac_scene_text(&scene).unwrap();
        assert_eq!(
            text,
            INDEX_SCENE.replace(
                "            asset filePath = @./VDB/s_000000.vdb@\n",
                "            asset filePath.timeSamples = {\n                0: @./VDB/s_000000.vdb@,\n                0.5: @./VDB/s_000001.vdb@,\n            }\n",
            )
        );

        // With a boundary: after the last prim inside World, one empty
        // line, then write_boundary_mesh's output indented by four, before
        // World's `}`.
        let mut scene = isaac(IsaacProfile::Index, VdbSource::Single("./VDB/s_000000.vdb".into()));
        scene.boundary = Some(boundary_surface(&two_patch_mesh()));
        let text = isaac_scene_text(&scene).unwrap();
        let b = {
            let mut b = String::new();
            write_boundary_mesh(&mut b, &scene.boundary.clone().unwrap());
            b
        };
        let indented: String = b
            .lines()
            .map(|l| if l.is_empty() { "\n".to_string() } else { format!("    {l}\n") })
            .collect();
        // World's closing `}` is the text's last line; the boundary goes
        // before it, after one empty line.
        let head = &INDEX_SCENE[..INDEX_SCENE.len() - "}\n".len()];
        assert_eq!(text, format!("{head}\n{indented}}}\n"));
        assert!(!text.contains("nvdb"));
    }

    #[test]
    fn isaac_pathtrace_scene() {
        let scene =
            isaac(IsaacProfile::Pathtrace, VdbSource::Single("./VDB/s_000000.vdb".into()));
        assert_eq!(isaac_scene_text(&scene).unwrap(), PATHTRACE_SCENE);

        let mut scene = scene;
        scene.vdb = VdbSource::Series(vec![
            VdbFrame { time: 0.0, path: "./VDB/s_000000.vdb".into() },
            VdbFrame { time: 0.5, path: "./VDB/s_000001.vdb".into() },
        ]);
        let text = isaac_scene_text(&scene).unwrap();
        assert_eq!(
            text,
            PATHTRACE_SCENE.replace(
                "                asset inputs:volume_density_texture = @./VDB/s_000000.vdb@\n",
                "                asset inputs:volume_density_texture.timeSamples = {\n                    0: @./VDB/s_000000.vdb@,\n                    0.5: @./VDB/s_000001.vdb@,\n                }\n",
            )
        );

        let mut scene =
            isaac(IsaacProfile::Pathtrace, VdbSource::Single("./VDB/s_000000.vdb".into()));
        scene.boundary = Some(boundary_surface(&two_patch_mesh()));
        let text = isaac_scene_text(&scene).unwrap();
        assert!(text.contains("    def Mesh \"boundary\"\n"), "boundary indented into World");
        assert!(!text.contains("nvdb"));

        // upAxis / metersPerUnit come from StageMeta.
        let mut scene = isaac(IsaacProfile::Index, VdbSource::Single("./VDB/s_000000.vdb".into()));
        scene.stage = StageMeta { up_axis: "Y".to_string(), meters_per_unit: 0.01 };
        let text = isaac_scene_text(&scene).unwrap();
        assert!(text.contains("    metersPerUnit = 0.01\n"), "{text}");
        assert!(text.contains("    upAxis = \"Y\"\n"), "{text}");
    }

    #[test]
    fn isaac_scene_refuses_nvdb() {
        let mut scene =
            isaac(IsaacProfile::Index, VdbSource::Single("./VDB/s_000000.nvdb".into()));
        let e = isaac_scene_text(&scene).unwrap_err().to_string();
        assert!(e.contains(".nvdb") && e.contains("117.9"), "{e}");

        scene.vdb = VdbSource::Series(vec![
            VdbFrame { time: 0.0, path: "./VDB/s_000000.vdb".into() },
            VdbFrame { time: 0.5, path: "./VDB/s_000001.nvdb".into() },
        ]);
        let e = isaac_scene_text(&scene).unwrap_err().to_string();
        assert!(e.contains(".nvdb"), "{e}");
    }
}
