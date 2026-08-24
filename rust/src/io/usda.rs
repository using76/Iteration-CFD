// meteor-cfd - Copyright (c) 2026 주식회사 메테오시뮬레이션 (Meteo Simulation Co., Ltd.)
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
//! **Known approximation** - `HostMesh` (`src/mesh.rs`) keeps only the
//! *aggregate* geometry of a boundary face (its centroid `b_cf`, area
//! vector `b_sf`, magnitude `b_mag_sf`): the original point loop is
//! consumed by `compute_geometry` and not retained. [`boundary_surface`]
//! therefore cannot fan-triangulate the *true* face polygon; instead it
//! reconstructs, per boundary face, a planar quadrilateral centred at the
//! face centroid, orthogonal to the face normal, with area equal to
//! `b_mag_sf`, and fans that quad into two triangles. This keeps the
//! documented per-patch `UsdGeomSubset` partition invariant exactly (every
//! generated triangle face belongs to the same patch as the boundary face
//! it came from) and gives a correct bounding box and a plausible preview
//! surface; it does not reproduce the exact patch outline. Should
//! `HostMesh` ever retain the boundary face's point loop, this function is
//! the only place that needs to change: fan the real loop from vertex 0
//! instead of a synthesised quad.
//!
//! No GPL-licensed source was consulted.

use std::path::Path;

use crate::error::{IoContext, Result};
use crate::mesh::HostMesh;
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

/// Build the renderable boundary surface for a mesh: one quad per boundary
/// face (see the module doc's "Known approximation"), fanned into two
/// triangles, grouped into one [`PatchSubset`] per patch.
pub fn boundary_surface(m: &HostMesh) -> SurfaceOut {
    let n_bf = m.n_boundary_faces.min(m.b_cf.len()).min(m.b_sf.len());
    let mut points = Vec::with_capacity(n_bf * 4);
    let mut face_vertex_counts = Vec::with_capacity(n_bf * 2);
    let mut face_vertex_indices = Vec::with_capacity(n_bf * 6);
    let mut patch_faces: Vec<Vec<u32>> = vec![Vec::new(); m.patches.len()];

    for bf in 0..n_bf {
        let c = m.b_cf[bf];
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
}
