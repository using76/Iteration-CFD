// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! The TET-04b size-source gates of SPEC-LIT §116.8 re-run in BOTH builds
//! through the public API: the curvature of the level-3 icosphere against
//! the oracle window, the 1 mm box pair's gaps on all three sides, and the
//! gradation limit |grad h| <= g-1 on 10^5 probe pairs. The unit is f64
//! geometry by §116.2, so every number must be the same when `Scalar` is
//! f32 (precision classes D and A). No GPU needed.
//!
//! Provenance: ORIGINAL. Written from Persson (2006), Eng. Comput. 22:95-109; the
//! estimates are derived in SPEC-LIT §116.8. No GPL-licensed source was consulted.

use ofgpu::tetmesh::sizefield::*;
use ofgpu::tetmesh::surfprep::{prepare, TetSurface};
use ofgpu::tetmesh::{Point, TetSizeSpec};
use std::collections::HashMap;
use std::f64::consts::PI;

/// SplitMix64, Steele, Lea & Flood 2014: every operation wrapping.
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> SplitMix64 {
        SplitMix64 { state: seed }
    }

    fn next(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    /// Uniform in [0, 1).
    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / 9007199254740992.0
    }

    fn uni(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.unit()
    }

    /// A uniform point in the box: x, then y, then z.
    fn point(&mut self, lo: Point, hi: Point) -> Point {
        [
            self.uni(lo[0], hi[0]),
            self.uni(lo[1], hi[1]),
            self.uni(lo[2], hi[2]),
        ]
    }
}

fn rel_ok(got: f64, want: f64, e: f64) -> bool {
    (got - want).abs() <= e * want.abs()
}

fn sub(a: Point, b: Point) -> Point {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn norm(a: Point) -> f64 {
    (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt()
}

/// The icosahedron's 12 unit vertices and 20 outward faces.
fn ico_base() -> (Vec<Point>, Vec<[u32; 3]>) {
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
    let unit = |p: Point| {
        let n = norm(p);
        [p[0] / n, p[1] / n, p[2] / n]
    };
    let f: [[usize; 3]; 20] = [
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
    (
        v.iter().map(|&p| unit(p)).collect(),
        f.iter()
            .map(|g| [g[0] as u32, g[1] as u32, g[2] as u32])
            .collect(),
    )
}

/// One midpoint per unordered pair: the normalised sum, cached.
fn ico_mid(pts: &mut Vec<Point>, cache: &mut HashMap<(u32, u32), u32>, a: u32, b: u32) -> u32 {
    let key = if a < b { (a, b) } else { (b, a) };
    if let Some(&m) = cache.get(&key) {
        return m;
    }
    let m = [pts[a as usize][0] + pts[b as usize][0], pts[a as usize][1] + pts[b as usize][1], pts[a as usize][2] + pts[b as usize][2]];
    let n = norm(m);
    let id = pts.len() as u32;
    pts.push([m[0] / n, m[1] / n, m[2] / n]);
    cache.insert(key, id);
    id
}

/// The icosphere at `level` subdivisions, on the unit sphere.
fn ico_mesh(level: u32) -> (Vec<Point>, Vec<[u32; 3]>) {
    let (mut pts, mut tris) = ico_base();
    for _ in 0..level {
        let mut cache: HashMap<(u32, u32), u32> = HashMap::new();
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

/// `ico(3, 0.5, centre)`-style fixture as a one-patch TetSurface.
fn ico_surface(level: u32, r: f64, centre: Point) -> TetSurface {
    let (pts, tris) = ico_mesh(level);
    TetSurface {
        points: pts
            .iter()
            .map(|p| [centre[0] + r * p[0], centre[1] + r * p[1], centre[2] + r * p[2]])
            .collect(),
        tri_patch: vec![0; tris.len()],
        patch_names: vec!["sphere".to_string()],
        tris,
    }
}

/// Two icospheres: ico(3, 0.5, origin) then ico(3, 0.2, (0, 0, 0.75)),
/// the gap between them 0.05.
fn two_sphere_surface() -> TetSurface {
    let (pa, ta) = ico_mesh(3);
    let (pb, tb) = ico_mesh(3);
    let off = pa.len() as u32;
    let mut points: Vec<Point> = pa
        .iter()
        .map(|p| [0.5 * p[0], 0.5 * p[1], 0.5 * p[2]])
        .collect();
    points.extend(pb.iter().map(|p| [0.2 * p[0], 0.2 * p[1], 0.75 + 0.2 * p[2]]));
    let mut tris = ta;
    tris.extend(tb.iter().map(|t| [t[0] + off, t[1] + off, t[2] + off]));
    TetSurface {
        points,
        tri_patch: vec![0; tris.len()],
        patch_names: vec!["spheres".to_string()],
        tris,
    }
}

/// The unit cube's 12 outward triangles.
const CUBE_TRIS: [[u32; 3]; 12] = [
    [0, 2, 3],
    [0, 3, 1],
    [4, 5, 7],
    [4, 7, 6],
    [0, 1, 5],
    [0, 5, 4],
    [2, 6, 7],
    [2, 7, 3],
    [0, 4, 6],
    [0, 6, 2],
    [1, 3, 7],
    [1, 7, 5],
];

/// The 1 mm gap: box A on z in [-0.005, 0], box B on z in
/// [0.001, 0.006], the same 0.01 x 0.01 footprint, patches "a" and "b".
fn box_pair_surface() -> TetSurface {
    let map = |b: u32, z0: f64| {
        [
            0.01 * f64::from(b & 1),
            0.01 * f64::from((b & 2) >> 1),
            z0 + 0.005 * f64::from((b & 4) >> 2),
        ]
    };
    let mut points: Vec<Point> = (0..8u32).map(|b| map(b, -0.005)).collect();
    points.extend((0..8u32).map(|b| map(b, 0.001)));
    let mut tris: Vec<[u32; 3]> = CUBE_TRIS.to_vec();
    tris.extend(CUBE_TRIS.map(|t| [t[0] + 8, t[1] + 8, t[2] + 8]));
    let mut tri_patch: Vec<u32> = vec![0; 12];
    tri_patch.extend(vec![1; 12]);
    TetSurface {
        points,
        tris,
        tri_patch,
        patch_names: vec!["a".to_string(), "b".to_string()],
    }
}

/// Every fixture passes `prepare(.., 60.0)` before use.
fn prepared(surf: &TetSurface) {
    assert!(
        prepare(surf, 60.0).is_ok(),
        "prepare refused {:?}",
        surf.patch_names
    );
}

/// A facet point: triangle `next() % n`, barycentric u, v folded back
/// into the triangle.
fn facet_point(rng: &mut SplitMix64, pts: &[Point], tris: &[[u32; 3]]) -> Point {
    let n = tris.len() as u64;
    let t = &tris[(rng.next() % n) as usize];
    let x0 = pts[t[0] as usize];
    let x1 = pts[t[1] as usize];
    let x2 = pts[t[2] as usize];
    let mut u = rng.unit();
    let mut v = rng.unit();
    if u + v > 1.0 {
        u = 1.0 - u;
        v = 1.0 - v;
    }
    [
        x0[0] + u * (x1[0] - x0[0]) + v * (x2[0] - x0[0]),
        x0[1] + u * (x1[1] - x0[1]) + v * (x2[1] - x0[1]),
        x0[2] + u * (x1[2] - x0[2]) + v * (x2[2] - x0[2]),
    ]
}

/// A true sphere point by rejection: w in [-1, 1]^3 with
/// 1e-3 <= |w| <= 1, scaled to the radius about c.
fn sphere_point(rng: &mut SplitMix64, c: Point, r: f64) -> Point {
    loop {
        let w = [rng.uni(-1.0, 1.0), rng.uni(-1.0, 1.0), rng.uni(-1.0, 1.0)];
        let n = norm(w);
        if (1e-3..=1.0).contains(&n) {
            return [
                c[0] + r * w[0] / n,
                c[1] + r * w[1] / n,
                c[2] + r * w[2] / n,
            ];
        }
    }
}

#[test]
fn sizefield_both_builds_scalar_width() {
    if cfg!(feature = "single") {
        assert_eq!(std::mem::size_of::<ofgpu::Scalar>(), 4);
    } else {
        assert_eq!(std::mem::size_of::<ofgpu::Scalar>(), 8);
    }
    println!("size_of<Scalar>={}", std::mem::size_of::<ofgpu::Scalar>());
}

#[test]
fn sizefield_both_builds_curvature_sphere() {
    let surf = ico_surface(3, 0.5, [0.0; 3]);
    prepared(&surf);
    let kap = triangle_curvatures(&surf, 60.0).unwrap();
    assert_eq!(kap.len(), 1280);
    assert!(kap.iter().all(|k| *k > 0.0), "all kappa above 0");
    let mut kmin = f64::INFINITY;
    let mut kmax = 0.0f64;
    for k in &kap {
        kmin = kmin.min(k * 0.5);
        kmax = kmax.max(k * 0.5);
    }
    assert!(
        (0.9965..=0.9975).contains(&kmin),
        "min kappa R {kmin} outside [0.9965, 0.9975]"
    );
    assert!(
        (1.1455..=1.1470).contains(&kmax),
        "max kappa R {kmax} outside [1.1455, 1.1470]"
    );
    let p = SizeParams::new(1.0, 1.2, Some(1e-3)).unwrap();
    let sources = curvature_sources(&surf, &p, 16, 60.0).unwrap();
    assert_eq!(sources.len(), 1280);
    let field =
        SizeField::build(p, &sources, [-1.0; 3], [1.0; 3], OctreeOptions::default()).unwrap();
    let b = 2.0 * PI * 0.5 / 16.0;
    let mut rng = SplitMix64::new(41);
    let mut facet_max = 0.0f64;
    for _ in 0..10_000 {
        let x = facet_point(&mut rng, &surf.points, &surf.tris);
        let h = field.h(x);
        assert!(h / b <= 1.02, "facet {x:?}: h {h} is {} b", h / b);
        facet_max = facet_max.max(h / b);
        assert_eq!(field.h(x).to_bits(), field.h_brute(x).to_bits(), "at {x:?}");
    }
    assert!(facet_max >= 0.9, "facet_max {facet_max} below 0.9");
    let mut rng = SplitMix64::new(43);
    let mut sphere_max = 0.0f64;
    for _ in 0..10_000 {
        let x = sphere_point(&mut rng, [0.0; 3], 0.5);
        let h = field.h(x);
        assert!(h / b <= 1.02, "sphere {x:?}: h {h} is {} b", h / b);
        sphere_max = sphere_max.max(h / b);
        assert_eq!(field.h(x).to_bits(), field.h_brute(x).to_bits(), "at {x:?}");
    }
    println!("TET04B-CURV facet_max={facet_max} sphere_max={sphere_max}");
}

#[test]
fn sizefield_both_builds_proximity_gap() {
    let surf = box_pair_surface();
    prepared(&surf);
    let facing = [2usize, 3, 12, 13];
    let out = triangle_gaps(&surf, GapSide::Outward).unwrap();
    assert_eq!(out.len(), 24);
    for (i, g) in out.iter().enumerate() {
        if facing.contains(&i) {
            let v = g.unwrap_or_else(|| panic!("outward {i}: expected Some(1e-3)"));
            assert!(rel_ok(v, 1e-3, 1e-12), "outward {i}: {v}");
        } else {
            assert_eq!(*g, None, "outward {i}");
        }
    }
    let inw = triangle_gaps(&surf, GapSide::Inward).unwrap();
    for (i, g) in inw.iter().enumerate() {
        let want = if i % 12 < 4 { 0.005 } else { 0.01 };
        let v = g.unwrap_or_else(|| panic!("inward {i}: expected Some({want})"));
        assert!(rel_ok(v, want, 1e-12), "inward {i}: {v} vs {want}");
    }
    let both = triangle_gaps(&surf, GapSide::Both).unwrap();
    for (i, g) in both.iter().enumerate() {
        if facing.contains(&i) {
            let v = g.unwrap_or_else(|| panic!("both {i}: expected Some(1e-3)"));
            assert!(rel_ok(v, 1e-3, 1e-12), "both {i}: {v}");
        } else {
            assert_eq!(g.map(f64::to_bits), inw[i].map(f64::to_bits), "both {i}");
        }
    }
    let p = SizeParams::new(0.02, 1.2, Some(1e-5)).unwrap();
    let curv = curvature_sources(&surf, &p, 16, 60.0).unwrap();
    assert!(curv.is_empty());
    let mut sources = proximity_sources(&surf, &p, 3, GapSide::Both).unwrap();
    sources.extend(curv);
    let lo = [-0.01, -0.01, -0.01];
    let hi = [0.02, 0.02, 0.016];
    let field = SizeField::build(p, &sources, lo, hi, OctreeOptions::default()).unwrap();
    let mut rng = SplitMix64::new(47);
    let want = 1e-3 / 3.0;
    for _ in 0..10_000 {
        let x = rng.point([0.0; 3], [0.01, 0.01, 0.001]);
        let h = field.h(x);
        assert!(rel_ok(h, want, 1e-12), "gap h({x:?}) = {h} vs {want}");
        assert_eq!(field.h(x).to_bits(), field.h_brute(x).to_bits(), "at {x:?}");
    }
    let x = [0.005, 0.005, 0.0105];
    let want = 0.005 / 3.0 + 0.2 * 0.002;
    assert!(rel_ok(field.h(x), want, 1e-12), "above: {} vs {want}", field.h(x));

    // Inward alone does not see the fluid gap; nothing at all is the base.
    let inw_sources = proximity_sources(&surf, &p, 3, GapSide::Inward).unwrap();
    let field_in =
        SizeField::build(p, &inw_sources, lo, hi, OctreeOptions::default()).unwrap();
    let x = [0.005, 0.005, 0.0005];
    assert!(
        rel_ok(field_in.h(x), 0.005 / 3.0, 1e-12),
        "inward h({x:?}) = {}",
        field_in.h(x)
    );
    let field_no = SizeField::build(p, &[], lo, hi, OctreeOptions::default()).unwrap();
    assert_eq!(field_no.h(x), 0.02);
}

#[test]
fn sizefield_both_builds_gradation_bounded() {
    let surf = two_sphere_surface();
    prepared(&surf);
    let gaps = triangle_gaps(&surf, GapSide::Outward).unwrap();
    let n_some = gaps.iter().filter(|g| g.is_some()).count();
    assert!((70..=90).contains(&n_some), "{n_some} gaps, want [70, 90]");
    let gmin = gaps.iter().filter_map(|g| *g).fold(f64::INFINITY, f64::min);
    assert!(
        (0.0530..=0.0540).contains(&gmin),
        "min gap {gmin} outside [0.0530, 0.0540]"
    );

    // g = 1.2 only, the same seed 53 stream as the unit test's first leg.
    let g = 1.2f64;
    let p = SizeParams::new(1.0, g, Some(1e-3)).unwrap();
    let spec = TetSizeSpec {
        gradation: g,
        curvature_cells: 16,
        gap_cells: 3,
        min_size: Some(1e-3),
    };
    let mut sources = surface_sources(&surf, &p, &spec, 60.0, GapSide::Both).unwrap();
    sources.push(SizeSource::Point {
        at: [0.8, 0.8, -0.8],
        size: 0.02,
    });
    sources.push(SizeSource::Box {
        min: [-0.9; 3],
        max: [-0.6; 3],
        size: 0.05,
    });
    sources.push(SizeSource::Patch {
        name: "floor".to_string(),
        tris: vec![
            [[-1.0, -1.0, -0.95], [1.0, -1.0, -0.95], [1.0, 1.0, -0.95]],
            [[-1.0, -1.0, -0.95], [1.0, 1.0, -0.95], [-1.0, 1.0, -0.95]],
        ],
        distance: 0.05,
        size: 0.04,
    });
    let field =
        SizeField::build(p, &sources, [-1.0; 3], [1.0; 3], OctreeOptions::default()).unwrap();
    let s = g - 1.0;
    let mut rng = SplitMix64::new(53);
    let mut max_ratio = 0.0f64;
    for _ in 0..100_000 {
        let a = rng.point([-1.0; 3], [1.0; 3]);
        let v = loop {
            let w = [rng.uni(-1.0, 1.0), rng.uni(-1.0, 1.0), rng.uni(-1.0, 1.0)];
            if norm(w) >= 1e-3 {
                break w;
            }
        };
        let n = norm(v);
        let u = [v[0] / n, v[1] / n, v[2] / n];
        let t = 10f64.powf(rng.uni(-4.0, 0.5f64.log10()));
        let b = [a[0] + t * u[0], a[1] + t * u[1], a[2] + t * u[2]];
        let (ha, hb) = (field.h(a), field.h(b));
        let d = norm(sub(b, a));
        assert!(
            (ha - hb).abs() <= s * d * (1.0 + 1e-9),
            "g {g}: |{ha} - {hb}| above {s} d at {a:?} {b:?}"
        );
        assert_eq!(field.h(a).to_bits(), field.h_brute(a).to_bits(), "at {a:?}");
        max_ratio = max_ratio.max((ha - hb).abs() / (s * d));
    }
    assert!(max_ratio >= 0.99, "g {g}: max_ratio {max_ratio}");
    println!(
        "TET04B-GRAD g={g} max_ratio={max_ratio} leaves={}",
        field.stats().leaves
    );
}
