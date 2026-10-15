// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! TET-06a's integration test: the isotropic surface remesh of SPEC-LIT
//! §116.10 through the public API only, in both builds. The fixture is
//! the unit tests' SPHERE (level-5 icosphere of radius 1, one patch
//! "sphere") and the bars are the same fixed ones: 95 percent of the
//! edge length ratios inside [0.8, 4/3], 99 percent of the smallest
//! triangle angles at or above 25 degrees, the vertex Hausdorff
//! distance at or below 1 percent of h, a closed manifold of Euler
//! characteristic 2 that `prepare` accepts, every patch kept. TET-06b
//! adds the feature gate: on the unit cube with a patch per face and on
//! the 30 degree wedge, the corners, feature edges and seams survive
//! the remesh in both builds, and the no-flat-no-cut gate holds
//! (SPEC-LIT (116.14k)/(116.14l)) on the five named rows.
//!
//! Provenance: ORIGINAL. Written from Botsch & Kobbelt (2004), Proc. SGP
//! 185-192; the choices are derived in SPEC-LIT §116.10. No GPL-licensed
//! source was consulted.

use ofgpu::tetmesh::remesh::{
    check_manifold, edge_length_ratios, min_angles_deg, remesh, vertex_hausdorff,
    Projector, RemeshOptions, Remeshed,
};
use ofgpu::tetmesh::surfprep::{prepare, PreparedSurface, TetSurface};
use ofgpu::tetmesh::Point;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::OnceLock;

fn norm(a: Point) -> f64 {
    (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt()
}

/// One midpoint per unordered pair: the normalised sum, cached.
fn ico_mid(
    pts: &mut Vec<Point>,
    cache: &mut BTreeMap<(u32, u32), u32>,
    a: u32,
    b: u32,
) -> u32 {
    let key = if a < b { (a, b) } else { (b, a) };
    if let Some(&m) = cache.get(&key) {
        return m;
    }
    let m = [
        pts[a as usize][0] + pts[b as usize][0],
        pts[a as usize][1] + pts[b as usize][1],
        pts[a as usize][2] + pts[b as usize][2],
    ];
    let n = norm(m);
    let id = pts.len() as u32;
    pts.push([m[0] / n, m[1] / n, m[2] / n]);
    cache.insert(key, id);
    id
}

/// The level-5 icosphere on the unit sphere: 10242 points, 20480
/// triangles.
fn ico(level: u32) -> (Vec<Point>, Vec<[u32; 3]>) {
    let t = (1.0 + 5.0f64.sqrt()) / 2.0;
    let v: [Point; 12] = [
        [-1.0, t, 0.0], [1.0, t, 0.0], [-1.0, -t, 0.0], [1.0, -t, 0.0],
        [0.0, -1.0, t], [0.0, 1.0, t], [0.0, -1.0, -t], [0.0, 1.0, -t],
        [t, 0.0, -1.0], [t, 0.0, 1.0], [-t, 0.0, -1.0], [-t, 0.0, 1.0],
    ];
    let f: [[u32; 3]; 20] = [
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
    ];
    let mut pts: Vec<Point> = v.iter().map(|&p| {
        let n = norm(p);
        [p[0] / n, p[1] / n, p[2] / n]
    }).collect();
    let mut tris: Vec<[u32; 3]> = f.iter().map(|g| [g[0], g[1], g[2]]).collect();
    for _ in 0..level {
        let mut cache: BTreeMap<(u32, u32), u32> = BTreeMap::new();
        let mut next: Vec<[u32; 3]> = Vec::with_capacity(tris.len() * 4);
        for [a, b, c] in tris.clone() {
            let ab = ico_mid(&mut pts, &mut cache, a, b);
            let bc = ico_mid(&mut pts, &mut cache, b, c);
            let ca = ico_mid(&mut pts, &mut cache, c, a);
            next.push([a, ab, ca]);
            next.push([b, bc, ab]);
            next.push([c, ca, bc]);
            next.push([ab, bc, ca]);
        }
        tris = next;
    }
    (pts, tris)
}

fn sphere_surface() -> TetSurface {
    let (pts, tris) = ico(5);
    assert_eq!(pts.len(), 10242);
    assert_eq!(tris.len(), 20480);
    TetSurface {
        points: pts,
        tris,
        tri_patch: vec![0; 20480],
        patch_names: vec!["sphere".to_string()],
    }
}

fn sphere_prepared() -> &'static PreparedSurface {
    static P: OnceLock<PreparedSurface> = OnceLock::new();
    P.get_or_init(|| prepare(&sphere_surface(), 60.0).unwrap())
}

fn sphere_out() -> &'static Remeshed {
    static R: OnceLock<Remeshed> = OnceLock::new();
    R.get_or_init(|| {
        remesh(
            sphere_prepared(),
            &|_: Point| 0.05,
            &RemeshOptions::default(),
        )
        .unwrap()
    })
}

/// The distance from `x` to the segment `p q`.
fn seg_dist(x: Point, p: Point, q: Point) -> f64 {
    let pq = [q[0] - p[0], q[1] - p[1], q[2] - p[2]];
    let l2 = pq[0] * pq[0] + pq[1] * pq[1] + pq[2] * pq[2];
    let d = [x[0] - p[0], x[1] - p[1], x[2] - p[2]];
    let t = if l2 == 0.0 {
        0.0
    } else {
        ((d[0] * pq[0] + d[1] * pq[1] + d[2] * pq[2]) / l2).clamp(0.0, 1.0)
    };
    let c = [p[0] + pq[0] * t, p[1] + pq[1] * t, p[2] + pq[2] * t];
    norm([x[0] - c[0], x[1] - c[1], x[2] - c[2]])
}

/// The patch's area: the sum of the triangle areas, in triangle order.
fn patch_area(s: &TetSurface, patch: u32) -> f64 {
    let mut a = 0.0f64;
    for (t, &p) in s.tris.iter().zip(&s.tri_patch) {
        if p != patch {
            continue;
        }
        let x = [
            s.points[t[0] as usize],
            s.points[t[1] as usize],
            s.points[t[2] as usize],
        ];
        let u = [x[1][0] - x[0][0], x[1][1] - x[0][1], x[1][2] - x[0][2]];
        let v = [x[2][0] - x[0][0], x[2][1] - x[0][1], x[2][2] - x[0][2]];
        let cr = [
            u[1] * v[2] - u[2] * v[1],
            u[2] * v[0] - u[0] * v[2],
            u[0] * v[1] - u[1] * v[0],
        ];
        a += norm(cr) * 0.5;
    }
    a
}

/// The unit cube with a patch per face ("xmin" .. "zmax"): point k has
/// coordinate 1.0 on axis q when bit q of k is set, else 0.0; each face
/// f gives triangles [f0, f1, f2] and [f0, f2, f3].
fn cube6() -> TetSurface {
    let points: Vec<Point> = (0..8u32)
        .map(|k| {
            [
                (k & 1) as f64,
                ((k >> 1) & 1) as f64,
                ((k >> 2) & 1) as f64,
            ]
        })
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

/// The 30 degree wedge: the prism over the triangle (0,0), (1,0),
/// (cos 30, sin 30) from z = 0 to z = 1, one patch.
fn wedge() -> TetSurface {
    let c = 30f64.to_radians().cos();
    let s = 30f64.to_radians().sin();
    let points: Vec<Point> = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [c, s, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [c, s, 1.0],
    ];
    let tris: Vec<[u32; 3]> = vec![
        [0, 2, 1],
        [3, 4, 5],
        [0, 1, 4],
        [0, 4, 3],
        [1, 2, 5],
        [1, 5, 4],
        [2, 0, 3],
        [2, 3, 5],
    ];
    TetSurface {
        points,
        tris,
        tri_patch: vec![0; 8],
        patch_names: vec!["wedge".to_string()],
    }
}

#[test]
fn remesh_both_builds_scalar_width() {
    if cfg!(feature = "single") {
        assert_eq!(std::mem::size_of::<ofgpu::Scalar>(), 4);
    } else {
        assert_eq!(std::mem::size_of::<ofgpu::Scalar>(), 8);
    }
    println!("size_of<Scalar>={}", std::mem::size_of::<ofgpu::Scalar>());
}

#[test]
fn remesh_both_builds_sphere() {
    let out = sphere_out();
    let ratios = edge_length_ratios(&out.surface, &|_: Point| 0.05);
    let n = ratios.len();
    let k = ratios.iter().filter(|&&r| r >= 0.8 && r <= 4.0 / 3.0).count();
    let frac = k as f64 / n as f64;
    let min = ratios.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = ratios.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mean = ratios.iter().sum::<f64>() / n as f64;
    println!(
        "remesh_both_builds_sphere: fraction {frac:.6}, min {min:.4}, max {max:.4}, mean {mean:.4}, edges {n}"
    );
    assert!(frac >= 0.95, "in-band fraction {frac} below 0.95");
    let angs = min_angles_deg(&out.surface);
    let na = angs.len();
    let ka = angs.iter().filter(|&&a| a >= 25.0).count();
    let afrac = ka as f64 / na as f64;
    let smallest = angs.iter().cloned().fold(f64::INFINITY, f64::min);
    println!(
        "remesh_both_builds_sphere: angle fraction {afrac:.6}, smallest {smallest:.3} deg"
    );
    assert!(afrac >= 0.99, "angle fraction {afrac} below 0.99");
    let vh = vertex_hausdorff(&out.surface, &Projector::new(&sphere_surface()));
    println!("remesh_both_builds_sphere: vertex hausdorff {vh:.3e} bar {}", 0.01 * 0.05);
    assert!(vh <= 0.01 * 0.05, "vertex hausdorff {vh} above bar");
    let rep = check_manifold(&out.surface).unwrap();
    assert_eq!(rep.euler, 2);
    assert!(prepare(&out.surface, 60.0).is_ok());
    assert!(out.surface.tri_patch.iter().all(|&q| q == 0));
    assert_eq!(out.stats.n_points, out.surface.points.len());
    assert_eq!(out.stats.n_tris, out.surface.tris.len());
    println!(
        "remesh_both_builds_sphere: points {}, tris {}, splits {}, collapses {}, flips {}",
        out.stats.n_points,
        out.stats.n_tris,
        out.stats.splits,
        out.stats.collapses,
        out.stats.flips
    );
}

/// TET-06b's cube gate through the public API, in both builds: the
/// corners, feature edges and seams of the unit cube at h = 0.1 with a
/// patch per face survive the remesh, and each patch keeps its area.
#[test]
fn remesh_both_builds_cube() {
    let input = cube6();
    let prep = prepare(&input, 60.0).unwrap();
    let out = remesh(&prep, &|_: Point| 0.1, &RemeshOptions::default()).unwrap();
    let q = prepare(&out.surface, 60.0).unwrap();
    let rep = check_manifold(&out.surface).unwrap();
    assert_eq!(rep.euler, 2, "euler");
    assert_eq!(q.report.n_corners, 8, "corners");
    assert_eq!(q.report.n_polylines, 12, "polylines");
    assert_eq!(q.report.n_closed_polylines, 0, "closed polylines");
    for k in 0..8 {
        for d in 0..3 {
            assert_eq!(
                out.surface.points[k][d].to_bits(),
                input.points[k][d].to_bits(),
                "point {k} coordinate {d}"
            );
        }
    }
    assert_eq!(out.corners, vec![0, 1, 2, 3, 4, 5, 6, 7], "out corners");
    assert_eq!(out.corners, q.corners, "corners equal the re-extraction");
    assert_eq!(out.feature_edges, q.feature_edges, "feature edges");
    assert_eq!(out.edge_is_seam, q.edge_is_seam, "seam flags");
    assert!(out.edge_is_seam.iter().all(|&s| s), "all seams");
    assert_eq!(out.edge_polyline.len(), out.feature_edges.len());
    assert!(out.edge_polyline.iter().all(|&k| k < 12), "polyline k");
    let mut pairs: BTreeMap<(u32, u32), usize> = BTreeMap::new();
    for chain in &q.polylines {
        let (i, j) = (chain[0], *chain.last().unwrap());
        assert!(i < 8 && j < 8, "chain ends {i} {j}");
        let x = i ^ j;
        assert!(x == 1 || x == 2 || x == 4, "ends {i} {j} not one bit");
        let q0 = x.trailing_zeros() as usize;
        let p0 = out.surface.points[i as usize];
        assert_eq!(p0[q0], 0.0, "first corner on the axis");
        let mut prev = -1.0f64;
        for &v in chain {
            let p = out.surface.points[v as usize];
            for d in 0..3 {
                if d != q0 {
                    assert_eq!(p[d].to_bits(), p0[d].to_bits(), "off-axis");
                }
            }
            assert!(p[q0] > prev, "not increasing");
            assert!(p[q0] <= 1.0, "past the far corner");
            prev = p[q0];
        }
        assert_eq!(prev, 1.0, "far corner at 1.0");
        let edges = chain.len() - 1;
        assert!((8..=12).contains(&edges), "chain of {edges} edges");
        *pairs.entry((i, j)).or_default() += 1;
    }
    assert_eq!(pairs.len(), 12, "12 different cube edges");
    for [a, b] in &out.feature_edges {
        let pa = out.surface.points[*a as usize];
        let pb = out.surface.points[*b as usize];
        let d = [
            pa[0] - pb[0],
            pa[1] - pb[1],
            pa[2] - pb[2],
        ];
        let r = norm(d) / 0.1;
        assert!(r >= 0.8 && r <= 4.0 / 3.0, "edge ratio {r}");
    }
    for p in 0..6u32 {
        let ia = patch_area(&input, p);
        let oa = patch_area(&out.surface, p);
        assert!((oa - ia).abs() <= 1e-12 * ia, "patch {p}: {oa} vs {ia}");
        let (axis, val) = (p as usize / 2, if p % 2 == 0 { 0.0f64 } else { 1.0 });
        for (ti, (t, &tp)) in
            out.surface.tris.iter().zip(&out.surface.tri_patch).enumerate()
        {
            if tp != p {
                continue;
            }
            for k in 0..3 {
                assert_eq!(
                    out.surface.points[t[k] as usize][axis].to_bits(),
                    val.to_bits(),
                    "patch {p} triangle {ti}"
                );
            }
        }
        println!("remesh_both_builds_cube: patch {p} area diff {:.3e}", oa - ia);
    }
    println!(
        "remesh_both_builds_cube: points {}, tris {}, splits {}, collapses {}, flips {}, feature edges {}",
        out.stats.n_points,
        out.stats.n_tris,
        out.stats.splits,
        out.stats.collapses,
        out.stats.flips,
        out.feature_edges.len()
    );
}

/// TET-06b's wedge gate through the public API, in both builds: the 30
/// degree prism's 6 corners and 9 polylines survive, the polyline on
/// the 30 degree edge keeps x = y = 0 bit for bit with z increasing,
/// and every point of the other polylines lies on its segment.
#[test]
fn remesh_both_builds_wedge() {
    let input = wedge();
    let prep = prepare(&input, 60.0).unwrap();
    assert_eq!(prep.report.n_corners, 6, "input corners");
    assert_eq!(prep.report.n_polylines, 9, "input polylines");
    let out = remesh(&prep, &|_: Point| 0.1, &RemeshOptions::default()).unwrap();
    let rep = check_manifold(&out.surface).unwrap();
    assert_eq!(rep.euler, 2);
    let q = prepare(&out.surface, 60.0).unwrap();
    assert_eq!(q.report.n_corners, 6, "output corners");
    assert_eq!(q.report.n_polylines, 9, "output polylines");
    for k in 0..6 {
        for d in 0..3 {
            assert_eq!(
                out.surface.points[k][d].to_bits(),
                input.points[k][d].to_bits(),
                "corner {k} coordinate {d}"
            );
        }
    }
    assert_eq!(out.corners, vec![0, 1, 2, 3, 4, 5], "out corners");
    assert_eq!(out.corners, q.corners, "corners");
    assert_eq!(out.feature_edges, q.feature_edges, "feature edges");
    let pts = |v: u32| out.surface.points[v as usize];
    let zchain = q
        .polylines
        .iter()
        .find(|c| c[0] == 0 && *c.last().unwrap() == 3)
        .expect("the z-edge polyline");
    let mut prev = 0.0f64;
    for (i, &v) in zchain.iter().enumerate() {
        let p = pts(v);
        assert_eq!(p[0], 0.0, "x on the z edge");
        assert_eq!(p[1], 0.0, "y on the z edge");
        if i == 0 {
            assert_eq!(p[2], 0.0, "z starts at 0.0");
        } else {
            assert!(p[2] > prev, "z strictly increasing");
        }
        assert!(p[2] <= 1.0);
        prev = p[2];
    }
    assert_eq!(prev, 1.0, "top corner");
    let edges = zchain.len() - 1;
    assert!((8..=12).contains(&edges), "z chain of {edges} edges");
    let mut lens: Vec<usize> = Vec::new();
    let mut worst = 0.0f64;
    for chain in &q.polylines {
        lens.push(chain.len() - 1);
        if chain[0] == 0 && *chain.last().unwrap() == 3 {
            continue;
        }
        let (p0, p1) = (pts(chain[0]), pts(*chain.last().unwrap()));
        for &v in chain {
            worst = worst.max(seg_dist(pts(v), p0, p1));
        }
    }
    assert!(worst <= 1e-12, "off-segment by {worst}");
    println!(
        "remesh_both_builds_wedge: chain lengths {lens:?}, largest seg_dist {worst:.3e}, points {}, tris {}",
        out.stats.n_points,
        out.stats.n_tris
    );
}

// ==========================================================================
//  TET-06b: no flat and no cut triangle (116.14k)/(116.14l)
// ==========================================================================

/// The point turned exactly as the unit tests' RCUBE turns CUBE1:
/// Rx(0.7), then Ry(0.5), then Rz(0.3).
fn turn(p: Point) -> Point {
    let (ga, gb, gg) = (0.7f64, 0.5f64, 0.3f64);
    let (ca, sa) = (ga.cos(), ga.sin());
    let (cb, sb) = (gb.cos(), gb.sin());
    let (cg, sg) = (gg.cos(), gg.sin());
    let (x, y, z) = (p[0], p[1], p[2]);
    let (y, z) = (y * ca - z * sa, y * sa + z * ca);
    let (x, z) = (x * cb + z * sb, -x * sb + z * cb);
    let (x, y) = (x * cg - y * sg, x * sg + y * cg);
    [x, y, z]
}

/// The unit cube with one patch, every point turned: RCUBE.
fn rcube() -> TetSurface {
    let base = {
        let points: Vec<Point> = (0..8u32)
            .map(|k| {
                [
                    (k & 1) as f64,
                    ((k >> 1) & 1) as f64,
                    ((k >> 2) & 1) as f64,
                ]
            })
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
            tri_patch: vec![0; 12],
            patch_names: vec!["box".to_string()],
        }
    };
    TetSurface {
        points: base.points.iter().map(|&p| turn(p)).collect(),
        tris: base.tris,
        tri_patch: base.tri_patch,
        patch_names: base.patch_names,
    }
}

/// The 30 degree wedge with every point turned: RWEDGE.
fn rwedge() -> TetSurface {
    let base = wedge();
    TetSurface {
        points: base.points.iter().map(|&p| turn(p)).collect(),
        tris: base.tris,
        tri_patch: base.tri_patch,
        patch_names: base.patch_names,
    }
}

/// The capped 16-sided cylinder, two apex points, three rings, the
/// fans then the two side bands; patch 0 "lower", patch 1 "upper":
/// CYL2, as in the unit tests.
fn cyl2() -> TetSurface {
    let mut points = vec![[0.0, 0.0, -1.0], [0.0, 0.0, 1.0]];
    for r in 0..3u32 {
        let z = match r {
            0 => -1.0,
            1 => 0.0,
            _ => 1.0,
        };
        for k in 0..16u32 {
            let t = std::f64::consts::TAU * k as f64 / 16.0;
            points.push([t.cos(), t.sin(), z]);
        }
    }
    let ring = |r: u32, k: u32| 2 + 16 * r + (k % 16);
    let mut tris: Vec<[u32; 3]> = Vec::new();
    for k in 0..16u32 {
        tris.push([0, ring(0, k + 1), ring(0, k)]);
    }
    for k in 0..16u32 {
        tris.push([1, ring(2, k), ring(2, k + 1)]);
    }
    for r in 0..2u32 {
        for k in 0..16u32 {
            let (a, b, c, d) =
                (ring(r, k), ring(r, k + 1), ring(r + 1, k + 1), ring(r + 1, k));
            tris.push([a, b, c]);
            tris.push([a, c, d]);
        }
    }
    let tri_patch: Vec<u32> = (0..96u32)
        .map(|t| if (16..32).contains(&t) || t >= 64 { 1 } else { 0 })
        .collect();
    TetSurface {
        points,
        tris,
        tri_patch,
        patch_names: vec!["lower".to_string(), "upper".to_string()],
    }
}

/// CYL2 with every point turned.
fn rcyl2() -> TetSurface {
    let base = cyl2();
    TetSurface {
        points: base.points.iter().map(|&p| turn(p)).collect(),
        tris: base.tris,
        tri_patch: base.tri_patch,
        patch_names: base.patch_names,
    }
}

/// The undirected edge key `(min, max)`.
fn key(a: u32, b: u32) -> (u32, u32) {
    if a < b {
        (a, b)
    } else {
        (b, a)
    }
}

/// (116.14k): how many of the surface's triangles are flat, that is
/// `|(x1 - x0) x (x2 - x0)| <= 1e-6 max(l01^2, l12^2, l02^2)`.
fn n_flat(s: &TetSurface) -> usize {
    let d2 = |a: Point| a[0] * a[0] + a[1] * a[1] + a[2] * a[2];
    s.tris
        .iter()
        .filter(|t| {
            let x = [
                s.points[t[0] as usize],
                s.points[t[1] as usize],
                s.points[t[2] as usize],
            ];
            let u = [x[1][0] - x[0][0], x[1][1] - x[0][1], x[1][2] - x[0][2]];
            let v = [x[2][0] - x[0][0], x[2][1] - x[0][1], x[2][2] - x[0][2]];
            let w = [x[0][0] - x[2][0], x[0][1] - x[2][1], x[0][2] - x[2][2]];
            let cr = [
                u[1] * v[2] - u[2] * v[1],
                u[2] * v[0] - u[0] * v[2],
                u[0] * v[1] - u[1] * v[0],
            ];
            let l2 = d2(u).max(d2(v)).max(d2(w));
            norm(cr) <= 1e-6 * l2
        })
        .count()
}

/// (116.14l): how many of the output's triangles are cut triangles -
/// two edges of `out.feature_edges` meeting at a vertex that is not in
/// `out.corners`.
fn n_cut(out: &Remeshed) -> usize {
    let feat: BTreeSet<(u32, u32)> =
        out.feature_edges.iter().map(|&e| (e[0], e[1])).collect();
    let corners: BTreeSet<u32> = out.corners.iter().copied().collect();
    let mut n = 0usize;
    for t in &out.surface.tris {
        for k in 0..3 {
            let (u, a, b) = (t[k], t[(k + 1) % 3], t[(k + 2) % 3]);
            if !corners.contains(&u)
                && feat.contains(&key(u, a))
                && feat.contains(&key(u, b))
            {
                n += 1;
                break;
            }
        }
    }
    n
}

/// The five named rows of the no-flat-no-cut gate (SPEC-LIT §116.18),
/// through the public API in both builds: the rotated cube at
/// h = 0.07 after 3 iterations, the rotated wedge at h = 0.13 after 3,
/// the cylinder at h = 0.2 after 2 and at h = 0.07 after 7, and the
/// rotated cylinder at h = 0.07 after 11. Each row remeshes and
/// asserts a closed 2-manifold of Euler 2, `prepare` ok on the output,
/// 0 flat and 0 cut triangles, and the output's feature tables equal
/// to the re-extraction's.
#[test]
fn remesh_both_builds_no_flat_no_cut() {
    let rows: Vec<(&str, TetSurface, f64, u32)> = vec![
        ("rcube", rcube(), 0.07, 3),
        ("rwedge", rwedge(), 0.13, 3),
        ("cyl2", cyl2(), 0.2, 2),
        ("cyl2", cyl2(), 0.07, 7),
        ("rcyl2", rcyl2(), 0.07, 11),
    ];
    for (name, input, h, n) in &rows {
        let tag = format!("{name} h {h} n {n}");
        let prep = prepare(input, 60.0).unwrap();
        let out = remesh(&prep, &|_: Point| *h, &RemeshOptions { iterations: *n, threads: 0 })
            .unwrap();
        let rep = check_manifold(&out.surface).unwrap();
        assert_eq!(rep.euler, 2, "{tag}: euler");
        let q = prepare(&out.surface, 60.0).unwrap();
        assert_eq!(n_flat(&out.surface), 0, "{tag}: flat triangles");
        assert_eq!(n_cut(&out), 0, "{tag}: cut triangles");
        assert_eq!(out.feature_edges, q.feature_edges, "{tag}: feature edges");
        assert_eq!(out.edge_is_seam, q.edge_is_seam, "{tag}: seam flags");
        assert_eq!(out.corners, q.corners, "{tag}: corners");
        println!(
            "remesh_both_builds_no_flat_no_cut({tag}): points {}, tris {}, \
             feature edges {}",
            out.stats.n_points,
            out.stats.n_tris,
            out.feature_edges.len()
        );
    }
}
