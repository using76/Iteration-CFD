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
//! characteristic 2 that `prepare` accepts, every patch kept.
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
use std::collections::BTreeMap;
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


