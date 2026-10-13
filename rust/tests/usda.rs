// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! VIO-07's integration test: `boundary_surface_loops` fans each boundary
//! face's true point loop from its first vertex (117.12). The fan keeps the
//! `UsdGeomSubset` partition exact (one subset per patch, ascending, every
//! triangle in exactly one subset), reproduces `geom64()`'s per-patch areas
//! and per-face area vectors to 1e-10 with every single triangle wound
//! outward, compacts the points the boundary references in ascending raw
//! order with the extent the bit-exact bbox of ALL raw points, refuses a
//! malformed raw mesh by name, and writes through `write_usda_scene`.
//!
//! Written from SPEC-LIT section 117.12.
//! Provenance: ORIGINAL.
//! No GPL-licensed source was consulted.

use ofgpu::blockgen::{self, BlockSpec, GradedAxis};
use ofgpu::error::Error;
use ofgpu::io::polymesh::{build_host_mesh, PolyMeshRaw};
use ofgpu::io::usda::{boundary_surface_loops, SceneSpec, write_usda_scene};
use ofgpu::mesh::refined::refined_core;
use ofgpu::mesh::{HostMesh, PatchInfo, PatchKind};
use ofgpu::types::{to_vec3, DVec3, Vec3};
use ofgpu::{Label, Scalar};

// ==========================================================================
//  Fixtures
// ==========================================================================

/// ONE cell: a hexagonal prism, 12 points, 8 faces, no internal face.
/// Bottom points 0..5 at z = 0, top points 6..11 the same xy at z = 3.
/// Patches: bottom (face 0), top (face 1), sides (faces 2..7), and an
/// empty `unused` patch - which must still appear as a subset.
fn prism() -> (PolyMeshRaw, HostMesh) {
    let xy = [(2.0, 0.0), (4.0, 0.0), (6.0, 2.0), (4.0, 4.0), (2.0, 4.0), (0.0, 2.0)];
    let mut points = Vec::with_capacity(12);
    for &(x, y) in &xy {
        points.push(DVec3::new(x, y, 0.0));
    }
    for &(x, y) in &xy {
        points.push(DVec3::new(x, y, 3.0));
    }
    let mut faces: Vec<Vec<Label>> = Vec::new();
    faces.push(vec![0, 5, 4, 3, 2, 1]);
    faces.push(vec![6, 7, 8, 9, 10, 11]);
    for (i, j) in [(0usize, 1usize), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)] {
        faces.push(vec![
            i as Label,
            j as Label,
            j as Label + 6,
            i as Label + 6,
        ]);
    }
    let patch = |name: &str, start: usize, size: usize| PatchInfo {
        name: name.to_string(),
        type_name: "patch".to_string(),
        kind: PatchKind::Generic,
        start,
        size,
        nbr_patch: None,
    };
    let raw = PolyMeshRaw {
        points,
        faces,
        owner: vec![0; 8],
        neighbour: Vec::new(),
        patches: vec![
            patch("bottom", 0, 1),
            patch("top", 1, 1),
            patch("sides", 2, 6),
            patch("unused", 8, 0),
        ],
    };
    let hm = build_host_mesh(&raw).expect("build_host_mesh on the prism");
    (raw, hm)
}

/// A uniform axis of a block.
fn axis(lo: f64, hi: f64, n: usize) -> GradedAxis {
    GradedAxis {
        lo: lo as Scalar,
        hi: hi as Scalar,
        n,
        expansion: 1.0,
        two_sided: false,
    }
}

/// The unit cube, `n` cells of `1/n` per axis. Cell index `i + n (j + n k)`.
fn cube(n: usize) -> (PolyMeshRaw, HostMesh) {
    let spec = BlockSpec {
        x: axis(0.0, 1.0, n),
        y: axis(0.0, 1.0, n),
        z: axis(0.0, 1.0, n),
        ..Default::default()
    };
    let raw = blockgen::raw_mesh(&spec).expect("raw_mesh");
    let hm = build_host_mesh(&raw).expect("build_host_mesh");
    (raw, hm)
}

/// The 2:1 refined box of `mesh/refined.rs`: 120 cells, 56 of edge 1/4 and
/// 64 of edge 1/8, with faces, edges and corners on the interface.
fn refined() -> (PolyMeshRaw, HostMesh) {
    let rb = refined_core([4, 4, 4], Vec3::new(0.25, 0.25, 0.25), 0.3, 1).expect("refined_core");
    let points: Vec<DVec3> = rb
        .points
        .iter()
        .map(|&p| DVec3::new(p.x as f64, p.y as f64, p.z as f64))
        .collect();
    let mut owner = rb.mesh.owner.clone();
    owner.extend(rb.mesh.b_face_cells.iter().copied());
    let raw = PolyMeshRaw {
        points,
        faces: rb.faces.clone(),
        owner,
        neighbour: rb.mesh.neighbour.clone(),
        patches: rb.mesh.patches.clone(),
    };
    let hm = build_host_mesh(&raw).expect("build_host_mesh");
    (raw, hm)
}

// ==========================================================================
//  The test's own oracles
// ==========================================================================

/// The test's own sorted, de-duplicated list of the point labels of the
/// boundary faces - the map from a compacted index back to a raw point id.
fn boundary_point_ids(raw: &PolyMeshRaw) -> Vec<usize> {
    let n_if = raw.neighbour.len();
    let mut ids: Vec<usize> = Vec::new();
    for f in n_if..raw.faces.len() {
        for &p in &raw.faces[f] {
            let p = p as usize;
            if !ids.contains(&p) {
                ids.push(p);
            }
        }
    }
    ids.sort_unstable();
    ids
}

/// The test's own triangles of each patch: face `bf`'s fan starts at the
/// triangle `sum over bf' < bf of (m_bf' - 2)`, and holds `m_bf - 2`.
fn patch_triangles(raw: &PolyMeshRaw) -> Vec<Vec<u32>> {
    let n_if = raw.neighbour.len();
    let n_bf = raw.faces.len() - n_if;
    let mut base_of = Vec::with_capacity(n_bf);
    let mut base = 0u32;
    for bf in 0..n_bf {
        base_of.push(base);
        base += (raw.faces[n_if + bf].len() - 2) as u32;
    }
    raw.patches
        .iter()
        .map(|p| {
            let mut v = Vec::new();
            for bf in p.start..p.start + p.size {
                let m = raw.faces[n_if + bf].len() - 2;
                for t in 0..m {
                    v.push(base_of[bf] + t as u32);
                }
            }
            v
        })
        .collect()
}

/// A fan triangle's area vector `0.5 (b - a) x (c - a)`, in f64 from raw
/// points (never from `SurfaceOut::points`, which are `Scalar`).
fn area_vec(pts: &[DVec3], t: [usize; 3]) -> DVec3 {
    let (a, b, c) = (pts[t[0]], pts[t[1]], pts[t[2]]);
    let (u, v) = (b - a, c - a);
    DVec3::new(
        0.5 * (u.y * v.z - u.z * v.y),
        0.5 * (u.z * v.x - u.x * v.z),
        0.5 * (u.x * v.y - u.y * v.x),
    )
}

/// The f64 magnitude through the dot product.
fn mag64(v: DVec3) -> f64 {
    v.dot(v).sqrt()
}

// ==========================================================================
//  Tests
// ==========================================================================

/// The partition (117.12): all triangles, ascending, one subset per patch,
/// every triangle in exactly one subset - and the F1/F2 exact counts.
#[test]
fn true_loops_partition() {
    let cases = [("F1", prism()), ("F2", cube(3)), ("F3", refined())];
    for (name, (raw, _hm)) in cases {
        let surf = boundary_surface_loops(&raw).expect("boundary_surface_loops");
        let n_if = raw.neighbour.len();
        let want_tris: usize =
            (n_if..raw.faces.len()).map(|f| raw.faces[f].len() - 2).sum();
        assert!(
            surf.face_vertex_counts.iter().all(|&c| c == 3),
            "{name}: face_vertex_counts must be all 3"
        );
        assert_eq!(surf.face_vertex_counts.len(), want_tris, "{name}: triangles");
        assert_eq!(surf.face_vertex_indices.len(), 3 * want_tris, "{name}: indices");
        assert_eq!(surf.patches.len(), raw.patches.len(), "{name}: subsets");
        let mut seen = vec![0u32; want_tris];
        for sp in &surf.patches {
            assert!(
                sp.face_indices.windows(2).all(|w| w[0] < w[1]),
                "{name}: subset '{}' not ascending",
                sp.name
            );
            for &t in &sp.face_indices {
                seen[t as usize] += 1;
            }
        }
        assert!(
            seen.iter().all(|&c| c == 1),
            "{name}: partition violated: {seen:?}"
        );
        let want = patch_triangles(&raw);
        for (sp, rp) in surf.patches.iter().zip(&raw.patches) {
            assert_eq!(sp.name, rp.name, "{name}: subset order");
        }
        for (sp, w) in surf.patches.iter().zip(&want) {
            assert_eq!(sp.face_indices, w.as_slice(), "{name}: patch triangles");
        }
        if name == "F1" {
            assert_eq!(want_tris, 20, "F1 triangles");
            assert_eq!(surf.patches[0].face_indices, vec![0, 1, 2, 3], "bottom");
            assert_eq!(surf.patches[1].face_indices, vec![4, 5, 6, 7], "top");
            assert_eq!(
                surf.patches[2].face_indices,
                (8..=19u32).collect::<Vec<u32>>(),
                "sides"
            );
            assert!(surf.patches[3].face_indices.is_empty(), "unused");
            assert_eq!(
                &surf.face_vertex_indices[0..12],
                &[0, 5, 4, 0, 4, 3, 0, 3, 2, 0, 2, 1],
                "F1: bottom face's fan"
            );
        }
        if name == "F2" {
            assert_eq!(want_tris, 108, "F2 triangles");
            assert_eq!(surf.points.len(), 56, "F2 points");
        }
    }
}

/// The fan's area equals `geom64()`'s, per patch, to 1e-10 relative - and
/// on F1 also against the closed forms.
#[test]
fn true_loops_area() {
    let cases = [("F1", prism()), ("F2", cube(3)), ("F3", refined())];
    for (name, (raw, hm)) in cases {
        let surf = boundary_surface_loops(&raw).expect("boundary_surface_loops");
        let g = hm.geom64();
        let pts: Vec<DVec3> =
            boundary_point_ids(&raw).iter().map(|&i| raw.points[i]).collect();
        let verts = |t: u32| {
            [
                surf.face_vertex_indices[3 * t as usize] as usize,
                surf.face_vertex_indices[3 * t as usize + 1] as usize,
                surf.face_vertex_indices[3 * t as usize + 2] as usize,
            ]
        };
        for (pi, sp) in surf.patches.iter().enumerate() {
            let p = &raw.patches[pi];
            let a_ref: f64 = (p.start..p.start + p.size).map(|bf| g.b_mag_sf[bf]).sum();
            if a_ref <= 0.0 {
                continue;
            }
            let a_fan: f64 = sp
                .face_indices
                .iter()
                .map(|&t| mag64(area_vec(&pts, verts(t))))
                .sum();
            assert!(
                (a_fan - a_ref).abs() <= 1e-10 * a_ref,
                "{name} patch '{}': A_fan {a_fan} vs A_ref {a_ref}",
                p.name
            );
            if name == "F1" && pi < 3 {
                let closed = [16.0f64, 16.0, 12.0 + 24.0 * 2.0f64.sqrt()];
                assert!(
                    (a_fan - closed[pi]).abs() <= 1e-10 * closed[pi],
                    "F1 patch '{}': A_fan {a_fan} vs closed form {}",
                    p.name,
                    closed[pi]
                );
            }
        }
    }
}

/// Every boundary face's fan sums to `geom64()`'s area vector, and every
/// single triangle is wound with a positive dot product on it.
#[test]
fn true_loops_keep_the_outward_winding() {
    let cases = [("F1", prism()), ("F2", cube(3)), ("F3", refined())];
    for (name, (raw, hm)) in cases {
        let surf = boundary_surface_loops(&raw).expect("boundary_surface_loops");
        let g = hm.geom64();
        let pts: Vec<DVec3> =
            boundary_point_ids(&raw).iter().map(|&i| raw.points[i]).collect();
        let n_if = raw.neighbour.len();
        let mut tri = 0usize;
        for bf in 0..raw.faces.len() - n_if {
            let m = raw.faces[n_if + bf].len() - 2;
            let mut sum = DVec3::new(0.0, 0.0, 0.0);
            for _ in 0..m {
                let av = area_vec(&pts, [
                    surf.face_vertex_indices[3 * tri] as usize,
                    surf.face_vertex_indices[3 * tri + 1] as usize,
                    surf.face_vertex_indices[3 * tri + 2] as usize,
                ]);
                assert!(
                    av.dot(g.b_sf[bf]) > 0.0,
                    "{name} bf {bf} tri {tri}: dot {}",
                    av.dot(g.b_sf[bf])
                );
                sum += av;
                tri += 1;
            }
            let diff = sum - g.b_sf[bf];
            assert!(
                mag64(diff) <= 1e-10 * mag64(g.b_sf[bf]),
                "{name} bf {bf}: |sum - b_sf| = {}",
                mag64(diff)
            );
        }
    }
}

/// The compacted points carry the same bits as `to_vec3` of their raw
/// point, and the extent is, bit for bit, the bbox of ALL raw points.
#[test]
fn true_loops_points_and_extent() {
    let cases = [("F1", prism()), ("F2", cube(3)), ("F3", refined())];
    for (name, (raw, _hm)) in cases {
        let surf = boundary_surface_loops(&raw).expect("boundary_surface_loops");
        let ids = boundary_point_ids(&raw);
        assert_eq!(surf.points.len(), ids.len(), "{name}: compacted points");
        for (k, &id) in ids.iter().enumerate() {
            let want = to_vec3(raw.points[id]);
            assert_eq!(
                surf.points[k].x.to_bits(),
                want.x.to_bits(),
                "{name}: point {k} (raw {id}).x"
            );
            assert_eq!(
                surf.points[k].y.to_bits(),
                want.y.to_bits(),
                "{name}: point {k} (raw {id}).y"
            );
            assert_eq!(
                surf.points[k].z.to_bits(),
                want.z.to_bits(),
                "{name}: point {k} (raw {id}).z"
            );
        }
        let mut lo = to_vec3(raw.points[0]);
        let mut hi = to_vec3(raw.points[0]);
        for p in &raw.points[1..] {
            let q = to_vec3(*p);
            lo.x = lo.x.min(q.x);
            lo.y = lo.y.min(q.y);
            lo.z = lo.z.min(q.z);
            hi.x = hi.x.max(q.x);
            hi.y = hi.y.max(q.y);
            hi.z = hi.z.max(q.z);
        }
        assert_eq!(surf.extent.0.x.to_bits(), lo.x.to_bits(), "{name}: extent lo.x");
        assert_eq!(surf.extent.0.y.to_bits(), lo.y.to_bits(), "{name}: extent lo.y");
        assert_eq!(surf.extent.0.z.to_bits(), lo.z.to_bits(), "{name}: extent lo.z");
        assert_eq!(surf.extent.1.x.to_bits(), hi.x.to_bits(), "{name}: extent hi.x");
        assert_eq!(surf.extent.1.y.to_bits(), hi.y.to_bits(), "{name}: extent hi.y");
        assert_eq!(surf.extent.1.z.to_bits(), hi.z.to_bits(), "{name}: extent hi.z");
        if name == "F1" {
            assert_eq!(ids.len(), 12, "F1 points");
            assert_eq!(surf.extent.0.x, 0.0);
            assert_eq!(surf.extent.0.y, 0.0);
            assert_eq!(surf.extent.0.z, 0.0);
            assert_eq!(surf.extent.1.x, 6.0);
            assert_eq!(surf.extent.1.y, 4.0);
            assert_eq!(surf.extent.1.z, 3.0);
        }
    }
}

/// The refusals, each `Error::Mesh` starting with the function's name and
/// carrying the reason words of contract rule 5.
#[test]
fn true_loops_refusals() {
    let (raw0, _hm) = prism();
    let expect = |r: &PolyMeshRaw, words: &[&str]| match boundary_surface_loops(r) {
        Err(Error::Mesh(msg)) => {
            assert!(
                msg.starts_with("boundary_surface_loops: "),
                "'{msg}' does not start with the function's name"
            );
            for w in words {
                assert!(msg.contains(w), "'{msg}' lacks '{w}'");
            }
        }
        other => panic!("expected Error::Mesh for {words:?}, got {other:?}"),
    };
    // A face cut to 2 points.
    let mut r = raw0.clone();
    r.faces[0] = vec![0, 5];
    expect(&r, &["boundary face 0", "fewer than 3 points"]);
    // A point label past the end.
    let mut r = raw0.clone();
    r.faces[0] = vec![0, 99, 4, 3, 2, 1];
    expect(&r, &["boundary face 0", "point 99"]);
    // A negative point label.
    let mut r = raw0.clone();
    r.faces[0] = vec![-1, 5, 4, 3, 2, 1];
    expect(&r, &["boundary face 0", "point -1"]);
    // sides shrunk to 5: boundary face 7 lands in no patch.
    let mut r = raw0.clone();
    r.patches[2].size = 5;
    expect(&r, &["do not tile", "boundary face 7"]);
    // sides grown to 7: the patch range runs past the last boundary face.
    let mut r = raw0.clone();
    r.patches[2].size = 7;
    expect(&r, &["do not tile", "sides"]);
    // top moved to start 0: boundary face 0 lands in two patches.
    let mut r = raw0.clone();
    r.patches[1].start = 0;
    expect(&r, &["do not tile", "boundary face 0"]);
}

/// F1's surface through `write_usda_scene`: the Mesh prim, exactly four
/// GeomSubsets in patch order, the bottom patch's indices and the extent.
#[test]
fn true_loops_scene_text() {
    let (raw, _hm) = prism();
    let surf = boundary_surface_loops(&raw).expect("boundary_surface_loops");
    let spec = SceneSpec { volumes: vec![], boundary: Some(surf) };
    let path = std::env::temp_dir().join(format!(
        "usda_vio07_loops_scene_{}.usda",
        std::process::id()
    ));
    write_usda_scene(&path, &spec).expect("write_usda_scene");
    let text = std::fs::read_to_string(&path).expect("read back");
    assert!(text.contains("def Mesh \"boundary\""));
    assert_eq!(text.matches("def GeomSubset").count(), 4);
    for nm in ["bottom", "top", "sides", "unused"] {
        assert!(
            text.contains(&format!("def GeomSubset \"{nm}\"")),
            "missing subset {nm} in:\n{text}"
        );
    }
    assert!(
        text.contains("int[] indices = [0, 1, 2, 3]"),
        "bottom's indices in:\n{text}"
    );
    assert!(
        text.contains("float3[] extent = [(0.0, 0.0, 0.0), (6.0, 4.0, 3.0)]"),
        "extent in:\n{text}"
    );
    let _ = std::fs::remove_file(&path);
}




