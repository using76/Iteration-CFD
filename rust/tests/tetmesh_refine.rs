// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! TET-09's integration test: the Delaunay refinement of SPEC-LIT §116.13
//! through the public API only, in both builds. The fixture is the unit
//! tests' sphere in a box: the level-3 icosphere remeshed to a size field
//! with `h = 0.25` on the sphere, `H = 0.5` and `g = 1.2`, in the domain box
//! [-2, 2]^3 (its 8 corner points), recovered by `recover_faces` and
//! refined from the hull. The bars are the same fixed ones: no kept
//! tetrahedron with `rho > 2` (116.17) or `R > c h(x_T)` (116.18) outside
//! the reported set; the split surface still recovered and the kept volume
//! still closing (116.16); the mean `edge / h(x)` in [0.85, 1.15]; the
//! count within 20 % of (116.20). Every number is evaluated in f64
//! (§116.2), so the printed lines match across builds.
//!
//! Provenance: ORIGINAL. Shewchuk (1998)'s radius-edge refinement and
//! Ruppert (1995)'s concentric shells as SPEC-LIT §116.13 derives them. No
//! GPL-licensed source was consulted.

use ofgpu::tetmesh::carve::CarveStart;
use ofgpu::tetmesh::recover::{delaunay_edges, recover_faces};
use ofgpu::tetmesh::refine::{mean_edge_ratio, predicted_count, quality, refine, Refined};
use ofgpu::tetmesh::remesh::{remesh, RemeshOptions};
use ofgpu::tetmesh::sizefield::{OctreeOptions, SizeField, SizeParams, SizeSource};
use ofgpu::tetmesh::surfprep::{prepare, TetSurface};
use ofgpu::tetmesh::{Point, TetRecoverSpec, TetRefineSpec};
use std::collections::{HashMap, HashSet};
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

fn rel(a: f64, b: f64) -> f64 {
    (a - b).abs() / b.abs()
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

struct Case {
    n_body: u32,
    field: SizeField,
    refined: Refined,
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
        let n_body = surf.tris.len() as u32;
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
        Case { n_body, field, refined }
    })
}

#[test]
fn the_build_width_is_reported() {
    println!(
        "Scalar is {} bytes; the tetmesh geometry is f64 in both builds",
        std::mem::size_of::<ofgpu::Scalar>()
    );
}

#[test]
fn refine_radius_edge_bound() {
    let c = case();
    let r = &c.refined;
    let fr = &r.recovery;
    let spec = TetRefineSpec::default();
    let h = |x: Point| c.field.h(x);
    let exempt: HashSet<usize> = r.exempt.iter().copied().collect();
    let mut violators = 0usize;
    for (k, t) in r.carved.tets.iter().enumerate() {
        let q = quality(t.map(|v| fr.points[v as usize]), &h);
        if q.violates(&spec) && !exempt.contains(&k) {
            violators += 1;
        }
    }
    let (present, total) = fr.recovered();
    let de = delaunay_edges(&fr.delaunay);
    let subs = fr.subsegments();
    let seg_present = subs.iter().filter(|s| de.contains(&(s[0], s[1]))).count();
    let body: Vec<[u32; 3]> = fr
        .subtris
        .iter()
        .zip(&fr.sub_parent)
        .filter(|(_, &p)| p < c.n_body)
        .map(|(s, _)| *s)
        .collect();
    let o = fr.points[body[0][0] as usize];
    let mut v_body = 0.0;
    for s in &body {
        let x = [0, 1, 2].map(|k| sub(fr.points[s[k] as usize], o));
        v_body += dot(x[0], cross(x[1], x[2]));
    }
    v_body /= 6.0;
    let e1 = rel(r.carved.volume, 64.0 - v_body);
    println!(
        "refine sphere: {} points, {} kept tets, {:?}; violators {violators} outside the reported \
         set of {}; subtriangles {present}/{total}, subsegments {seg_present}/{}; (116.16) rel {e1:.2e}",
        fr.points.len(),
        r.carved.tets.len(),
        r.stats,
        r.exempt.len(),
        subs.len()
    );
    assert_eq!(violators, 0);
    assert!(r.exempt.is_empty());
    assert_eq!(present, total);
    assert_eq!(seg_present, subs.len());
    assert!(e1 <= 1e-12, "(116.16) rel {e1:e}");
}

#[test]
fn refine_size_conformance() {
    let c = case();
    let r = &c.refined;
    let m = mean_edge_ratio(&r.recovery.points, &r.carved.tets, &|x: Point| c.field.h(x));
    let ok = (0.85..=1.15).contains(&m);
    println!(
        "refine size: mean edge/h(x) = {m:.4}, band [0.85, 1.15]: {}",
        if ok { "PASS" } else { "MISSED" }
    );
    assert!(ok, "mean edge/h {m}");
}

#[test]
fn refine_count_predicted() {
    let c = case();
    let r = &c.refined;
    let n = r.carved.tets.len() as f64;
    let np = predicted_count(&r.recovery.points, &r.carved.tets, &|x: Point| c.field.h(x));
    let dev = n / np - 1.0;
    let ok = dev.abs() <= 0.2;
    println!(
        "refine count: {n} kept tets against N_pred = {np:.1}, deviation {:+.2} %: {}",
        100.0 * dev,
        if ok { "PASS" } else { "MISSED" }
    );
    assert!(ok, "count deviation {dev}");
}
