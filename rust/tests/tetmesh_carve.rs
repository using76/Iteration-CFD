// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! TET-08's integration test: the carve of SPEC-LIT §116.12 through the
//! public API only, in both builds. The fixture is the unit tests' level-3
//! icosphere in the domain box [-2, 2]^3 (the box as its 8 corner points),
//! recovered by `recover_faces`; the bars are the same fixed ones: the kept
//! volume is the box minus the split sphere's enclosed volume (116.16),
//! relative 1e-12, the kept boundary closes on it by (116.16b), the patch
//! and hull face counts equal the surface's, a seed in the fluid keeps the
//! same tetrahedra, and a seed inside the sphere, a seed outside the box
//! and an inward-wound sphere at the hull are refusals by name. Every
//! number is evaluated in f64 (§116.2), so the printed lines match across
//! builds.
//!
//! Provenance: ORIGINAL. The divergence theorem of SPEC-LIT §116.9's
//! (116.13) and §92.4's R4 carried to tetrahedra; the choices are derived
//! in SPEC-LIT §116.12. No GPL-licensed source was consulted.

use ofgpu::tetmesh::carve::{carve, CarveStart, Side};
use ofgpu::tetmesh::recover::{recover_faces, FaceRecovery};
use ofgpu::tetmesh::surfprep::TetSurface;
use ofgpu::tetmesh::{Point, TetRecoverSpec};
use std::collections::HashMap;

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

/// The level-`level` icosphere of radius 1, one patch, wound inward when
/// `inward`, followed by the 8 corners of the box [-2, 2]^3.
fn sphere_in_box(level: u32, inward: bool) -> TetSurface {
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
    for _ in 0..level {
        let mut cache: HashMap<(u32, u32), u32> = HashMap::new();
        let mut next = Vec::with_capacity(tris.len() * 4);
        for [a, b, c] in tris {
            let mut mid = |x: u32, y: u32| -> u32 {
                *cache.entry((x.min(y), x.max(y))).or_insert_with(|| {
                    let (p, q) = (pts[x as usize], pts[y as usize]);
                    pts.push(unit([
                        0.5 * (p[0] + q[0]),
                        0.5 * (p[1] + q[1]),
                        0.5 * (p[2] + q[2]),
                    ]));
                    pts.len() as u32 - 1
                })
            };
            let (ab, bc, ca) = (mid(a, b), mid(b, c), mid(c, a));
            next.extend([[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]);
        }
        tris = next;
    }
    if inward {
        for t in &mut tris {
            t.swap(1, 2);
        }
    }
    for k in 0..8u32 {
        let c = |b: u32| if (k >> b) & 1 == 1 { 2.0 } else { -2.0 };
        pts.push([c(0), c(1), c(2)]);
    }
    let n = tris.len();
    TetSurface {
        points: pts,
        tris,
        tri_patch: vec![0; n],
        patch_names: vec!["sphere".to_string()],
    }
}

/// (116.13a) over every subtriangle, about the first one's first point.
fn enclosed(fr: &FaceRecovery) -> f64 {
    let o = fr.points[fr.subtris[0][0] as usize];
    let mut v = 0.0;
    for s in &fr.subtris {
        let x0 = sub(fr.points[s[0] as usize], o);
        let x1 = sub(fr.points[s[1] as usize], o);
        let x2 = sub(fr.points[s[2] as usize], o);
        v += dot(x0, cross(x1, x2));
    }
    v / 6.0
}

fn rel(a: f64, b: f64) -> f64 {
    (a - b).abs() / b.abs()
}

#[test]
fn carve_both_builds_scalar_width() {
    if cfg!(feature = "single") {
        assert_eq!(std::mem::size_of::<ofgpu::Scalar>(), 4);
    } else {
        assert_eq!(std::mem::size_of::<ofgpu::Scalar>(), 8);
    }
    println!("size_of<Scalar>={}", std::mem::size_of::<ofgpu::Scalar>());
}

#[test]
fn carve_both_builds_volume_closes() {
    let s = sphere_in_box(3, false);
    let fr = recover_faces(&s, &[], &TetRecoverSpec::default()).unwrap();
    let c = carve(&s, &fr, &CarveStart::Hull).unwrap();
    let target = 64.0 - enclosed(&fr);
    let e1 = rel(c.volume, target);
    let e2 = rel(c.boundary_volume(&fr.points), c.volume);
    println!(
        "carve sphere in box: kept {} tets, V {:.15}, target {:.15}, rel {e1:.2e}, (116.16b) {e2:.2e}",
        c.tets.len(),
        c.volume,
        target
    );
    assert!(e1 <= 1e-12, "(116.16) rel {e1:e}");
    assert!(e2 <= 1e-12, "(116.16b) rel {e2:e}");
    let (per, hull) = c.patch_face_counts(&s, &fr);
    assert_eq!(per, vec![fr.subtris.len()]);
    assert_eq!(hull, fr.delaunay.hull_faces().len());
    assert_eq!(c.regions.iter().filter(|r| r.side == Side::Solid).count(), 1);
    assert!(c.dropped().is_empty());
    let cs = carve(&s, &fr, &CarveStart::Seed([-1.9, -1.9, -1.9])).unwrap();
    assert_eq!(cs.kept_slots, c.kept_slots);
    assert_eq!(cs.volume.to_bits(), c.volume.to_bits());
}

#[test]
fn carve_both_builds_seed_refusal() {
    let s = sphere_in_box(3, false);
    let fr = recover_faces(&s, &[], &TetRecoverSpec::default()).unwrap();
    let err = format!("{}", carve(&s, &fr, &CarveStart::Seed([0.0, 0.0, 0.0])).unwrap_err());
    println!("{err}");
    assert!(err.contains("carve/seed") && err.contains("wrong side"), "{err}");
    assert!(err.contains("patch \"sphere\""), "{err}");
    let err = format!("{}", carve(&s, &fr, &CarveStart::Seed([5.0, 0.0, 0.0])).unwrap_err());
    println!("{err}");
    assert!(err.contains("carve/seed") && err.contains("outside the domain hull"), "{err}");

    let s = sphere_in_box(3, true);
    let fr = recover_faces(&s, &[], &TetRecoverSpec::default()).unwrap();
    let err = format!("{}", carve(&s, &fr, &CarveStart::Hull).unwrap_err());
    println!("{err}");
    assert!(err.contains("carve/hull"), "{err}");
    let c = carve(&s, &fr, &CarveStart::Seed([0.0, 0.0, 0.0])).unwrap();
    let e = rel(c.volume, -enclosed(&fr));
    println!("carve inward sphere from a seed: V {:.15}, rel {e:.2e}", c.volume);
    assert!(e <= 1e-12, "rel {e:e}");
}
