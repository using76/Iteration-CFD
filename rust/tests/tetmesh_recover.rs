// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! TET-07a's integration test: the conforming edge recovery of SPEC-LIT
//! §116.11 through the public API only, in both builds. The fixtures are
//! the unit tests' cube with a patch per face, the 30 degree wedge and the
//! 10 degree spike, each remeshed at h = 0.1 (TET-06b), and the concave
//! L-shaped prism at h = 0.2; the bars are the same fixed ones: every
//! subsegment an edge of the final Delaunay triangulation, the cap (64)
//! not reached, the protected corners as the unit tests name them, and
//! the cap a refusal by name that is exact in the pass count. Every number
//! is evaluated in f64 (§116.2), so the printed lines match across builds.
//! TET-07b adds the face half: Schönhardt's twisted prism, the cube and
//! the remeshed icosphere with the boundary face set equal to the split
//! surface, the wedge, spike and prism with every face beyond it on a
//! tetrahedron flat to rounding, every Steiner point on its triangle within
//! 1e-12 h, and the face cap exact on the wedge.
//!
//! Provenance: ORIGINAL. Written from Shewchuk (1998), Murphy, Mount &
//! Gable (2001), Cohen-Steiner, Colin de Verdière & Yvinec (2004), Lawson
//! (1977), Ruppert (1995) and Schönhardt (1928); the choices are derived
//! in SPEC-LIT §116.11. No GPL-licensed source was
//! consulted.

use ofgpu::tetmesh::recover::{
    delaunay_edges, recover_edges, recover_faces, recover_remeshed, recover_remeshed_faces,
    EdgeRecovery, FaceRecovery, SteinerSite,
};
use ofgpu::tetmesh::sizefield::point_triangle_distance;
use ofgpu::tetmesh::remesh::{remesh, RemeshOptions, Remeshed};
use ofgpu::tetmesh::surfprep::{prepare, TetSurface};
use ofgpu::tetmesh::{Point, TetRecoverSpec};

fn norm(a: Point) -> f64 {
    (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt()
}

fn cube6() -> TetSurface {
    let points: Vec<Point> = (0..8u32)
        .map(|k| [(k & 1) as f64, ((k >> 1) & 1) as f64, ((k >> 2) & 1) as f64])
        .collect();
    let faces: [[u32; 4]; 6] = [
        [0, 4, 6, 2],
        [1, 3, 7, 5],
        [0, 1, 5, 4],
        [2, 6, 7, 3],
        [0, 2, 3, 1],
        [4, 5, 7, 6],
    ];
    let mut tris: Vec<[u32; 3]> = Vec::new();
    for f in &faces {
        tris.push([f[0], f[1], f[2]]);
        tris.push([f[0], f[2], f[3]]);
    }
    TetSurface {
        points,
        tris,
        tri_patch: (0..12u32).map(|t| t / 2).collect(),
        patch_names: ["xmin", "xmax", "ymin", "ymax", "zmin", "zmax"]
            .iter()
            .map(|s| s.to_string())
            .collect(),
    }
}

fn wedge() -> TetSurface {
    let c = 30f64.to_radians().cos();
    let s = 30f64.to_radians().sin();
    TetSurface {
        points: vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [c, s, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [c, s, 1.0],
        ],
        tris: vec![
            [0, 2, 1],
            [3, 4, 5],
            [0, 1, 4],
            [0, 4, 3],
            [1, 2, 5],
            [1, 5, 4],
            [2, 0, 3],
            [2, 3, 5],
        ],
        tri_patch: vec![0; 8],
        patch_names: vec!["wedge".to_string()],
    }
}

/// The triangular pyramid whose unit apex edges meet pairwise at 10 degrees.
fn spike() -> TetSurface {
    let rho = (2.0 * (1.0 - 10f64.to_radians().cos()) / 3.0).sqrt();
    let h = (1.0 - rho * rho).sqrt();
    let mut points: Vec<Point> = (0..3)
        .map(|k| {
            let a = 2.0 * std::f64::consts::PI * k as f64 / 3.0;
            [rho * a.cos(), rho * a.sin(), 0.0]
        })
        .collect();
    points.push([0.0, 0.0, h]);
    TetSurface {
        points,
        tris: vec![[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]],
        tri_patch: vec![0; 4],
        patch_names: vec!["spike".to_string()],
    }
}

/// The L-shaped prism with a 90 degree re-entrant edge over (1,1).
fn lprism() -> TetSurface {
    let xy = [[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [1.0, 1.0], [1.0, 2.0], [0.0, 2.0]];
    let mut points: Vec<Point> = Vec::new();
    for z in [0.0, 1.0] {
        for q in &xy {
            points.push([q[0], q[1], z]);
        }
    }
    let mut tris: Vec<[u32; 3]> = Vec::new();
    for k in 1..5u32 {
        tris.push([0, k + 1, k]);
        tris.push([6, 6 + k, 6 + k + 1]);
    }
    for i in 0..6u32 {
        let j = (i + 1) % 6;
        tris.push([i, j, j + 6]);
        tris.push([i, j + 6, i + 6]);
    }
    let n = tris.len();
    TetSurface {
        points,
        tris,
        tri_patch: vec![0; n],
        patch_names: vec!["lprism".to_string()],
    }
}

/// The rhombus pyramid whose top diagonal A C is in no Delaunay triangulation.
fn rhombus() -> TetSurface {
    TetSurface {
        points: vec![
            [-2.0, 0.0, 0.0],
            [0.3, -1.0, 0.0],
            [2.0, 0.0, 0.0],
            [-0.2, 1.0, 0.0],
            [0.0, 0.0, -1.0],
        ],
        tris: vec![[0, 1, 2], [0, 2, 3], [1, 0, 4], [2, 1, 4], [3, 2, 4], [0, 3, 4]],
        tri_patch: vec![0; 6],
        patch_names: vec!["rhombus".to_string()],
    }
}

fn remeshed(s: &TetSurface, h: f64) -> Remeshed {
    let prep = prepare(s, 60.0).unwrap();
    remesh(&prep, &|_: Point| h, &RemeshOptions::default()).unwrap()
}

/// 100 % of subsegments present, the cap not reached, the points the
/// triangulation's own, each Steiner point on its parent edge.
fn gate(name: &str, r: &EdgeRecovery, n_input: usize) {
    let (present, total) = r.recovered();
    assert_eq!(present, total, "{name}");
    assert!(r.passes < 64, "{name}: {} passes", r.passes);
    assert_eq!(r.delaunay.points(), &r.points[..]);
    assert_eq!(r.points.len(), n_input + r.steiner());
    for (k, (&e, &t)) in r.steiner_edge.iter().zip(&r.steiner_t).enumerate() {
        let [a, b] = r.edges[e as usize];
        let (pa, pb) = (r.points[a as usize], r.points[b as usize]);
        let x = r.points[n_input + k];
        let want = [
            pa[0] + t * (pb[0] - pa[0]),
            pa[1] + t * (pb[1] - pa[1]),
            pa[2] + t * (pb[2] - pa[2]),
        ];
        let off = norm([x[0] - want[0], x[1] - want[1], x[2] - want[2]]);
        let l = norm([pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2]]);
        assert!(off <= 1e-12 * l, "{name}: Steiner {k} off by {off}");
    }
    println!(
        "{name}: {total} subsegments, {} protected, Steiner {} (ball {}, split {}), passes {}, encroached {}",
        r.protected.len(),
        r.steiner(),
        r.ball_steiner,
        r.split_steiner,
        r.passes,
        r.encroached
    );
}

#[test]
fn recover_both_builds_scalar_width() {
    if cfg!(feature = "single") {
        assert_eq!(std::mem::size_of::<ofgpu::Scalar>(), 4);
    } else {
        assert_eq!(std::mem::size_of::<ofgpu::Scalar>(), 8);
    }
    println!("size_of<Scalar>={}", std::mem::size_of::<ofgpu::Scalar>());
}

#[test]
fn recover_both_builds_cube_wedge_spike() {
    let spec = TetRecoverSpec::default();
    let out = remeshed(&cube6(), 0.1);
    let r = recover_remeshed(&out, &spec).unwrap();
    assert!(r.protected.is_empty());
    gate("cube", &r, out.surface.points.len());

    let out = remeshed(&wedge(), 0.1);
    let r = recover_remeshed(&out, &spec).unwrap();
    assert_eq!(r.protected, vec![0, 1, 2, 3, 4, 5]);
    gate("wedge", &r, out.surface.points.len());

    let out = remeshed(&spike(), 0.1);
    let r = recover_remeshed(&out, &spec).unwrap();
    assert_eq!(r.protected, vec![0, 1, 2, 3]);
    let n_in = out.surface.points.len();
    for k in 0..r.ball_steiner {
        let [a, b] = r.edges[r.steiner_edge[k] as usize];
        let x = r.points[n_in + k];
        let on_ball = r.protected.iter().zip(&r.protect_radius).any(|(&v, &rv)| {
            let p = r.points[v as usize];
            (v == a || v == b)
                && (norm([x[0] - p[0], x[1] - p[1], x[2] - p[2]]) - rv).abs() <= 1e-12 * rv
        });
        assert!(on_ball, "spike ball point {k}");
    }
    gate("spike", &r, n_in);

    let out = remeshed(&lprism(), 0.2);
    let r = recover_remeshed(&out, &spec).unwrap();
    gate("lprism", &r, out.surface.points.len());
}

#[test]
fn recover_both_builds_cap() {
    let s = rhombus();
    let one = TetRecoverSpec { protect_angle_deg: 90.0, max_passes: 1 };
    let e = recover_edges(&s, &[], &one).unwrap_err().to_string();
    assert!(
        e.contains("recover/cap: 1 of 9 subsegments missing after 1 passes, first [0, 2]"),
        "{e}"
    );
    let two = TetRecoverSpec { protect_angle_deg: 90.0, max_passes: 2 };
    let r = recover_edges(&s, &[], &two).unwrap();
    assert_eq!((r.passes, r.steiner()), (2, 1));
    assert_eq!(r.points[5], [0.0, 0.0, 0.0]);
    assert_eq!(r.recovered(), (10, 10));

    let out = remeshed(&lprism(), 0.2);
    let spec = TetRecoverSpec::default();
    let r = recover_remeshed(&out, &spec).unwrap();
    assert!(r.passes >= 3);
    let at = TetRecoverSpec { max_passes: r.passes, ..spec.clone() };
    assert_eq!(recover_remeshed(&out, &at).unwrap().points, r.points);
    let below = TetRecoverSpec { max_passes: r.passes - 1, ..spec };
    let e = recover_remeshed(&out, &below).unwrap_err().to_string();
    assert!(e.contains("recover/cap: "), "{e}");
    println!("cap: {e}");
}

// ---------------------------------------------------------------------
// TET-07b: the face half.

/// Schönhardt's twisted prism (1928): the bottom `a_i` at 120 i degrees on
/// the unit circle at `z = 0`, the top `b_i` at 120 i + 30 degrees at
/// `z = 1`, each side quad split along its reflex diagonal `a_i b_(i+1)`,
/// wound outward.
fn schonhardt() -> TetSurface {
    let mut points: Vec<Point> = Vec::new();
    for (z, twist) in [(0.0, 0.0), (1.0, 30.0)] {
        for i in 0..3 {
            let a = (120.0 * i as f64 + twist).to_radians();
            points.push([a.cos(), a.sin(), z]);
        }
    }
    let mut tris: Vec<[u32; 3]> = vec![[0, 2, 1], [3, 4, 5]];
    for i in 0..3u32 {
        let j = (i + 1) % 3;
        tris.push([i, j, 3 + j]);
        tris.push([i, 3 + j, 3 + i]);
    }
    TetSurface {
        points,
        tris,
        tri_patch: vec![0; 8],
        patch_names: vec!["schonhardt".to_string()],
    }
}

/// The icosphere at `level` subdivisions on the unit sphere, outward.
fn ico(level: u32) -> TetSurface {
    let t = (1.0 + 5.0f64.sqrt()) / 2.0;
    let v: [Point; 12] = [
        [-1.0, t, 0.0],
        [1.0, t, 0.0],
        [-1.0, -t, 0.0],
        [1.0, -t, 0.0],
        [0.0, -1.0, t],
        [0.0, 1.0, t],
        [0.0, -1.0, -t],
        [0.0, 1.0, -t],
        [t, 0.0, -1.0],
        [t, 0.0, 1.0],
        [-t, 0.0, -1.0],
        [-t, 0.0, 1.0],
    ];
    let f: [[u32; 3]; 20] = [
        [0, 11, 5],
        [0, 5, 1],
        [0, 1, 7],
        [0, 7, 10],
        [0, 10, 11],
        [1, 5, 9],
        [5, 11, 4],
        [11, 10, 2],
        [10, 7, 6],
        [7, 1, 8],
        [3, 9, 4],
        [3, 4, 2],
        [3, 2, 6],
        [3, 6, 8],
        [3, 8, 9],
        [4, 9, 5],
        [2, 4, 11],
        [6, 2, 10],
        [8, 6, 7],
        [9, 8, 1],
    ];
    let unit = |p: Point| {
        let n = norm(p);
        [p[0] / n, p[1] / n, p[2] / n]
    };
    let mut pts: Vec<Point> = v.iter().map(|&p| unit(p)).collect();
    let mut tris: Vec<[u32; 3]> = f.to_vec();
    for _ in 0..level {
        let mut cache: std::collections::BTreeMap<(u32, u32), u32> = Default::default();
        let mut mid = |pts: &mut Vec<Point>, a: u32, b: u32| -> u32 {
            let k = (a.min(b), a.max(b));
            if let Some(&id) = cache.get(&k) {
                return id;
            }
            let (p, q) = (pts[a as usize], pts[b as usize]);
            pts.push(unit([(p[0] + q[0]) / 2.0, (p[1] + q[1]) / 2.0, (p[2] + q[2]) / 2.0]));
            let id = pts.len() as u32 - 1;
            cache.insert(k, id);
            id
        };
        let mut next = Vec::with_capacity(tris.len() * 4);
        for [a, b, c] in tris {
            let ab = mid(&mut pts, a, b);
            let bc = mid(&mut pts, b, c);
            let ca = mid(&mut pts, c, a);
            next.extend([[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]);
        }
        tris = next;
    }
    let n = tris.len();
    TetSurface {
        points: pts,
        tris,
        tri_patch: vec![0; n],
        patch_names: vec!["sphere".to_string()],
    }
}

fn area(a: Point, b: Point, c: Point) -> f64 {
    let u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let v = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
    0.5 * norm([
        u[1] * v[2] - u[2] * v[1],
        u[2] * v[0] - u[0] * v[2],
        u[0] * v[1] - u[1] * v[0],
    ])
}

/// The face gate (§116.18 TET-07b): every subsegment and subtriangle in
/// the final triangulation, the boundary face set equal to the split
/// surface, each input triangle tiled by its subtriangles (relative
/// 1e-12) and each Steiner point within `1e-12 h` of every triangle it is
/// on, `h` that triangle's longest edge.
fn face_gate(name: &str, r: &FaceRecovery, n_input: usize, exact: bool) {
    assert_eq!(r.points.len(), n_input + r.steiner());
    assert_eq!(r.delaunay.points(), &r.points[..]);
    assert!(r.passes < 64, "{name}: {} passes", r.passes);
    let (present, total) = r.recovered();
    assert_eq!(present, total, "{name}: subtriangles");
    let de = delaunay_edges(&r.delaunay);
    let subs = r.subsegments();
    assert!(subs.iter().all(|s| de.contains(&(s[0], s[1]))), "{name}: subsegments");
    let mut split: Vec<[u32; 3]> = r
        .subtris
        .iter()
        .map(|t| {
            let mut s = *t;
            s.sort_unstable();
            s
        })
        .collect();
    split.sort_unstable();
    let faces = r.surface_faces();
    let in_split: std::collections::HashSet<[u32; 3]> = split.iter().copied().collect();
    assert!(split.iter().all(|f| faces.binary_search(f).is_ok()), "{name}: split surface");
    let extra: Vec<[u32; 3]> = faces.iter().filter(|f| !in_split.contains(*f)).copied().collect();
    if exact {
        assert!(extra.is_empty(), "{name}: {} boundary faces beyond the split surface", extra.len());
    }
    let on = r.point_tris();
    let mut flat_tets = 0usize;
    let mut flat_vol = 0.0f64;
    let mut explained: std::collections::HashSet<[u32; 3]> = Default::default();
    for t in r.delaunay.finite_tets() {
        let p = t.map(|k| r.points[k as usize]);
        let d = |a: Point, b: Point| [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
        let (u, v, w) = (d(p[0], p[1]), d(p[0], p[2]), d(p[0], p[3]));
        let six = (u[0] * (v[1] * w[2] - v[2] * w[1]) - u[1] * (v[0] * w[2] - v[2] * w[0])
            + u[2] * (v[0] * w[1] - v[1] * w[0]))
            .abs();
        let mut l = 0.0f64;
        for i in 0..4 {
            for j in i + 1..4 {
                l = l.max(norm(d(p[i], p[j])));
            }
        }
        if six <= 1e-12 * l * l * l {
            flat_tets += 1;
            flat_vol = flat_vol.max(six / (l * l * l));
            for skip in 0..4 {
                let mut f = [t[(skip + 1) % 4], t[(skip + 2) % 4], t[(skip + 3) % 4]];
                f.sort_unstable();
                explained.insert(f);
            }
        }
    }
    assert!(extra.iter().all(|f| explained.contains(f)), "{name}: an extra face off a flat tet");
    let mut tiled = vec![0.0f64; r.tris.len()];
    for (t, &p) in r.subtris.iter().zip(&r.sub_parent) {
        let [a, b, c] = t.map(|k| r.points[k as usize]);
        tiled[p as usize] += area(a, b, c);
    }
    for (ti, t) in r.tris.iter().enumerate() {
        let [a, b, c] = t.map(|k| r.points[k as usize]);
        let want = area(a, b, c);
        assert!(
            (tiled[ti] - want).abs() <= 1e-12 * want,
            "{name}: triangle {ti} tiled {} of {want}",
            tiled[ti]
        );
    }
    let mut worst = 0.0f64;
    for k in 0..r.steiner() {
        let x = r.points[n_input + k];
        assert!(!on[n_input + k].is_empty());
        for &ti in &on[n_input + k] {
            let tri = r.tris[ti as usize].map(|v| r.points[v as usize]);
            let h = (0..3)
                .map(|m| {
                    let (p, q) = (tri[m], tri[(m + 1) % 3]);
                    norm([q[0] - p[0], q[1] - p[1], q[2] - p[2]])
                })
                .fold(0.0, f64::max);
            let dist = point_triangle_distance(x, &tri) / h;
            worst = worst.max(dist);
            assert!(dist <= 1e-12, "{name}: Steiner {k} off triangle {ti} by {dist} h");
        }
        if let SteinerSite::Facet { u, v, .. } = r.sites[k] {
            assert!(u > 0.0 && v > 0.0 && u + v < 1.0, "{name}: Steiner {k} at ({u}, {v})");
        }
    }
    println!(
        "{name}: {} triangles -> {} subtriangles, {} subsegments, Steiner {} (ball {}, split {}, encroach-split {}, facet {} of which centroid {}), passes {} (edge {}), encroached {} segments {} faces, {} extra faces on {flat_tets} flat tets (6V/L^3 <= {flat_vol:e}), worst off-surface {worst:e} h",
        r.tris.len(),
        r.subtris.len(),
        subs.len(),
        r.steiner(),
        r.ball_steiner,
        r.split_steiner,
        r.encroach_split_steiner,
        r.facet_steiner,
        r.centroid_steiner,
        r.passes,
        r.edge_passes,
        r.encroached,
        r.encroached_faces,
        extra.len()
    );
}

#[test]
fn recover_both_builds_faces() {
    let spec = TetRecoverSpec::default();
    let r = recover_faces(&schonhardt(), &[], &spec).unwrap();
    assert!(r.steiner() >= 1);
    face_gate("schonhardt", &r, 6, true);

    let out = remeshed(&cube6(), 0.1);
    let r = recover_remeshed_faces(&out, &spec).unwrap();
    face_gate("cube", &r, out.surface.points.len(), true);

    let out = remeshed(&ico(3), 0.15);
    let r = recover_remeshed_faces(&out, &spec).unwrap();
    face_gate("sphere", &r, out.surface.points.len(), true);

    for (name, s) in [("wedge", wedge()), ("spike", spike()), ("lprism", lprism())] {
        let out = remeshed(&s, 0.1);
        let r = recover_remeshed_faces(&out, &spec).unwrap();
        face_gate(name, &r, out.surface.points.len(), false);
    }
}

#[test]
fn recover_both_builds_faces_cap() {
    let out = remeshed(&wedge(), 0.1);
    let spec = TetRecoverSpec::default();
    let r = recover_remeshed_faces(&out, &spec).unwrap();
    assert!(r.passes > r.edge_passes);
    let at = TetRecoverSpec { max_passes: r.passes, ..spec.clone() };
    assert_eq!(recover_remeshed_faces(&out, &at).unwrap().points, r.points);
    let below = TetRecoverSpec { max_passes: r.passes - 1, ..spec };
    let e = recover_remeshed_faces(&out, &below).unwrap_err().to_string();
    let tail = format!("subtriangles missing after {} passes, first ", r.passes - 1);
    assert!(e.contains("recover/cap: ") && e.contains(&tail), "{e}");
    println!("faces cap: {e}");
}
