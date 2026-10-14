// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! TET-10a's integration test: the flips of SPEC-LIT §116.14 through the
//! public API only, in both builds. The fixture is TET-09's: the level-3
//! icosphere remeshed to a size field with `h = 0.25` on the sphere,
//! `H = 0.5` and `g = 1.2`, in the domain box [-2, 2]^3, recovered and
//! refined from the hull. The bars are §116.18's for TET-10a: the worst
//! dihedral (116.21) never decreases in any pass, the boundary face set is
//! unchanged bit for bit, and every tetrahedron keeps `orient3d > 0`
//! (116.21b). Every number is evaluated in f64 (§116.2), so the printed
//! lines match across builds.
//!
//! Provenance: ORIGINAL. Shewchuk (2002)'s edge removal and 2-3 / 3-2
//! flips as SPEC-LIT §116.14 derives them. No GPL-licensed source was
//! consulted.

use ofgpu::tetmesh::carve::CarveStart;
use ofgpu::tetmesh::optimise::{boundary_faces, check_valid, dihedral_angles, flip, Flipped};
use ofgpu::tetmesh::predicates::orient3d;
use ofgpu::tetmesh::recover::recover_faces;
use ofgpu::tetmesh::refine::{refine, Refined};
use ofgpu::tetmesh::remesh::{remesh, RemeshOptions};
use ofgpu::tetmesh::sizefield::{OctreeOptions, SizeField, SizeParams, SizeSource};
use ofgpu::tetmesh::surfprep::{prepare, TetSurface};
use ofgpu::tetmesh::{Point, TetOptimiseSpec, TetRecoverSpec, TetRefineSpec};
use std::collections::HashMap;
use std::sync::OnceLock;

fn sub(a: Point, b: Point) -> Point {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn dot(a: Point, b: Point) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: Point, b: Point) -> Point {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// The level-3 icosphere of radius 1, one patch "sphere".
fn ico3() -> TetSurface {
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
        let n = dot(p, p).sqrt();
        [p[0] / n, p[1] / n, p[2] / n]
    };
    let mut pts: Vec<Point> = v.iter().map(|&p| unit(p)).collect();
    let mut tris: Vec<[u32; 3]> = f.to_vec();
    for _ in 0..3 {
        let mut cache: HashMap<(u32, u32), u32> = HashMap::new();
        let mut next = Vec::with_capacity(tris.len() * 4);
        for [a, b, c] in tris {
            let mut mid = |x: u32, y: u32| -> u32 {
                *cache.entry((x.min(y), x.max(y))).or_insert_with(|| {
                    let (p, q) = (pts[x as usize], pts[y as usize]);
                    pts.push(unit([
                        p[0] + 0.5 * (q[0] - p[0]),
                        p[1] + 0.5 * (q[1] - p[1]),
                        p[2] + 0.5 * (q[2] - p[2]),
                    ]));
                    pts.len() as u32 - 1
                })
            };
            let (ab, bc, ca) = (mid(a, b), mid(b, c), mid(c, a));
            next.extend([[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]);
        }
        tris = next;
    }
    let tri_patch = vec![0; tris.len()];
    TetSurface { points: pts, tris, tri_patch, patch_names: vec!["sphere".to_string()] }
}

/// TET-09's refined sphere in a box and its flips with the default passes.
struct Case {
    refined: Refined,
    flipped: Flipped,
}

fn case() -> &'static Case {
    static CASE: OnceLock<Case> = OnceLock::new();
    CASE.get_or_init(|| {
        let s = ico3();
        let tris: Vec<[Point; 3]> = s.tris.iter().map(|t| t.map(|v| s.points[v as usize])).collect();
        let field = SizeField::build(
            SizeParams::new(0.5, 1.2, None).unwrap(),
            &[SizeSource::Patch { name: "sphere".to_string(), tris, distance: 0.0, size: 0.25 }],
            [-2.0; 3],
            [2.0; 3],
            OctreeOptions::default(),
        )
        .unwrap();
        let prep = prepare(&s, 60.0).unwrap();
        let r = remesh(&prep, &|x: Point| field.h(x), &RemeshOptions::default()).unwrap();
        let mut surf = r.surface.clone();
        for k in 0..8u32 {
            let c = |b: u32| if (k >> b) & 1 == 1 { 2.0 } else { -2.0 };
            surf.points.push([c(0), c(1), c(2)]);
        }
        let fr = recover_faces(&surf, &r.feature_edges, &TetRecoverSpec::default()).unwrap();
        let refined = refine(
            &surf,
            fr,
            &CarveStart::Hull,
            &TetRefineSpec::default(),
            &|x: Point| field.h(x),
        )
        .unwrap();
        let flipped = flip(
            &refined.recovery.points,
            &refined.carved.tets,
            &refined.recovery.subtris,
            TetOptimiseSpec::default().passes,
        )
        .unwrap();
        Case { refined, flipped }
    })
}

fn volume(points: &[Point], tets: &[[u32; 4]]) -> f64 {
    let mut s = 0.0;
    for t in tets {
        let p = t.map(|v| points[v as usize]);
        s += dot(sub(p[0], p[3]), cross(sub(p[1], p[3]), sub(p[2], p[3])));
    }
    s / 6.0
}

#[test]
fn the_build_width_is_reported() {
    println!(
        "Scalar is {} bytes; the tetmesh geometry is f64 in both builds",
        std::mem::size_of::<ofgpu::Scalar>()
    );
}

#[test]
fn optimise_flips_monotone() {
    let c = case();
    let (r, f) = (&c.refined, &c.flipped);
    let pts = &r.recovery.points;
    let w = &f.stats.worst;
    let monotone = w.windows(2).all(|p| p[1] >= p[0]);
    let positive = f.tets.iter().all(|t| {
        let p = t.map(|v| pts[v as usize]);
        orient3d(&p[0], &p[1], &p[2], &p[3]) > 0.0
    });
    let max_dihedral = |tets: &[[u32; 4]]| {
        tets.iter()
            .map(|t| dihedral_angles(t.map(|v| pts[v as usize])).into_iter().fold(0.0, f64::max))
            .fold(0.0, f64::max)
            .to_degrees()
    };
    let (v0, v1) = (volume(pts, &r.carved.tets), volume(pts, &f.tets));
    let rv = (v1 - v0).abs() / v0.abs();
    println!(
        "optimise flips: {} -> {} tets, {:?}; worst min dihedral per pass {:?} deg, monotone {}; \
         largest dihedral {:.4} -> {:.4} deg (not gated); all orient3d > 0 {positive}; volume rel {rv:.2e}",
        r.carved.tets.len(),
        f.tets.len(),
        f.stats,
        w.iter().map(|x| format!("{x:.4}")).collect::<Vec<_>>(),
        if monotone { "PASS" } else { "MISSED" },
        max_dihedral(&r.carved.tets),
        max_dihedral(&f.tets)
    );
    assert!(monotone, "{w:?}");
    assert!(f.stats.moves() > 0 && f.stats.min_gain > 0.0);
    assert!(positive);
    check_valid(pts, &f.tets).unwrap();
    assert!(rv <= 1e-12, "volume rel {rv:e}");
}

#[test]
fn optimise_boundary_frozen() {
    let c = case();
    let (r, f) = (&c.refined, &c.flipped);
    let b0 = boundary_faces(&r.carved.tets);
    let b1 = boundary_faces(&f.tets);
    let mut carved: Vec<[u32; 3]> = r
        .carved
        .faces
        .iter()
        .map(|&x| {
            let k = (0..3).min_by_key(|&k| x[k]).unwrap();
            [x[k], x[(k + 1) % 3], x[(k + 2) % 3]]
        })
        .collect();
    carved.sort_unstable();
    let mut count: HashMap<[u32; 3], usize> = HashMap::new();
    for t in &f.tets {
        for f in [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]] {
            let mut k = f.map(|i| t[i]);
            k.sort_unstable();
            *count.entry(k).or_insert(0) += 1;
        }
    }
    let subtris_kept = r.recovery.subtris.iter().all(|s| {
        let mut k = *s;
        k.sort_unstable();
        count.contains_key(&k)
    });
    println!(
        "optimise boundary: {} faces before, {} after, carve's {}; identical {}; every subtriangle a face {}",
        b0.len(),
        b1.len(),
        carved.len(),
        b0 == b1 && b0 == carved,
        subtris_kept
    );
    assert_eq!(b0, carved);
    assert_eq!(b1, b0);
    assert!(subtris_kept);
}
