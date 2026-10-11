// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! The Delaunay kernel of SPEC-LIT §116.6 runs in BOTH builds with the same
//! assertions: the unit is f64 geometry by §116.2, so the tetrahedron sets,
//! digests, hull sizes and signed volumes must not move when `Scalar` is
//! f32. Precision class A of docs/17 §5.1 rule 4. No GPU needed.
//!
//! Provenance: ORIGINAL. Written from Bowyer (1981), Comput. J. 24(2):162-166, Watson (1981), Comput. J. 24(2):167-172, and Devillers, Pion & Teillaud (2002), Int. J. Found. Comput. Sci. 13(2):181-199. No GPL-licensed source was consulted.

use ofgpu::tetmesh::delaunay::*;
use ofgpu::tetmesh::order::*;
use ofgpu::Scalar;

/// SplitMix64 (Steele, Lea & Flood 2014), all operations wrapping - the
/// same stream the §116.5 tests use.
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        SplitMix64 { state: seed }
    }
    fn next(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
}

/// `u() = (next() >> 11) * 2^-53`, uniform on `[0, 1)`.
fn draw_u(r: &mut SplitMix64) -> f64 {
    (r.next() >> 11) as f64 * (1.0 / 9007199254740992.0)
}

/// `R(n, seed)`: n points `[u(), u(), u()]`, x then y then z draw.
fn r_points(n: usize, seed: u64) -> Vec<ofgpu::tetmesh::Point> {
    let mut r = SplitMix64::new(seed);
    (0..n).map(|_| [draw_u(&mut r), draw_u(&mut r), draw_u(&mut r)]).collect()
}

/// `L(m)`: the m x m x m lattice, index `i = x + m*(y + m*z)`.
fn lattice(m: usize) -> Vec<ofgpu::tetmesh::Point> {
    let mut out = Vec::with_capacity(m * m * m);
    for z in 0..m {
        for y in 0..m {
            for x in 0..m {
                out.push([x as f64, y as f64, z as f64]);
            }
        }
    }
    out
}

/// The insertion order `n-1 .. 0`.
fn reversed(n: usize) -> Vec<u32> {
    (0..n as u32).rev().collect()
}

#[test]
fn delaunay_both_builds_scalar_width() {
    if cfg!(feature = "single") {
        assert_eq!(std::mem::size_of::<Scalar>(), 4);
    } else {
        assert_eq!(std::mem::size_of::<Scalar>(), 8);
    }
}

/// Each tet's ids sorted ascending, then the list sorted.
fn sorted_tets(mut tets: Vec<[u32; 4]>) -> Vec<[u32; 4]> {
    for t in tets.iter_mut() {
        t.sort_unstable();
    }
    tets.sort_unstable();
    tets
}

#[test]
fn delaunay_both_builds_cube_and_lattice3() {
    let cube = lattice(2);
    let d = Delaunay::build(&cube).unwrap();
    d.check().unwrap();
    assert_eq!(tet_digest(&d.finite_tets()), 0x8ecc7ce0071e1d85);
    let r = Delaunay::build_ordered(&cube, &reversed(8)).unwrap();
    assert_eq!(tet_digest(&r.finite_tets()), 0x8ecc7ce0071e1d85);
    assert_eq!(d.finite_volume(), 1.0);

    let l3 = lattice(3);
    let d = Delaunay::build(&l3).unwrap();
    d.check().unwrap();
    assert_eq!(tet_digest(&d.finite_tets()), 0x1ab2235318ee986d);
    assert_eq!(d.finite_volume(), 8.0);
    let r = Delaunay::build_ordered(&l3, &reversed(27)).unwrap();
    assert_eq!(tet_digest(&r.finite_tets()), 0x1ab2235318ee986d);
}

#[test]
fn delaunay_both_builds_coplanar_hull() {
    let a: Vec<ofgpu::tetmesh::Point> = vec![
        [0.0, 0.0, 0.0],
        [5.0, 0.0, 0.0],
        [0.0, 5.0, 0.0],
        [0.0, 0.0, 5.0],
        [3.0, 3.0, 0.0],
    ];
    let b: Vec<ofgpu::tetmesh::Point> = vec![
        [0.0, 0.0, 0.0],
        [5.0, 0.0, 0.0],
        [0.0, 5.0, 0.0],
        [0.0, 0.0, 5.0],
        [5.0, 5.0, 0.0],
    ];
    let want_a: Vec<[u32; 4]> = vec![[0, 1, 3, 4], [0, 2, 3, 4]];
    let want_b: Vec<[u32; 4]> = vec![[0, 1, 2, 3], [1, 2, 3, 4]];
    for (pts, want) in [(&a, &want_a), (&b, &want_b)] {
        let d = Delaunay::build(pts).unwrap();
        d.check().unwrap();
        assert_eq!(sorted_tets(d.finite_tets()), *want);
        let r = Delaunay::build_ordered(pts, &reversed(5)).unwrap();
        r.check().unwrap();
        assert_eq!(sorted_tets(r.finite_tets()), *want);
    }
}

#[test]
fn delaunay_both_builds_random2000() {
    let pts = r_points(2000, 2026);
    let d = Delaunay::build(&pts).unwrap();
    let rep = d.check().unwrap();
    assert_eq!(rep.vertices, 2000);
    assert_eq!(rep.edges, 15044);
    assert_eq!(rep.faces, 26012);
    assert_eq!(rep.tets, 12967);
    assert_eq!(d.hull_faces().len(), 156);
    assert_eq!(tet_digest(&d.finite_tets()), 0xb88702ffa69aeb56);
    let vol = d.finite_volume();
    assert!(
        (vol - 0.9602216874050136).abs() <= 1e-12 * 0.9602216874050136,
        "vol {}",
        vol
    );
}

#[test]
fn delaunay_both_builds_lattice8() {
    let pts = lattice(8);
    let d = Delaunay::build(&pts).unwrap();
    let rep = d.check().unwrap();
    assert_eq!(d.hull_faces().len(), 588);
    let vol = d.finite_volume();
    assert!((vol - 343.0).abs() <= 1e-12 * 343.0, "vol {}", vol);
    let e = rep.vertices as i64 - rep.edges as i64 + rep.faces as i64 - rep.tets as i64;
    assert_eq!(e, 1);
    let r = Delaunay::build_ordered(&pts, &reversed(512)).unwrap();
    r.check().unwrap();
    assert_eq!(tet_digest(&d.finite_tets()), tet_digest(&r.finite_tets()));
}




/// `locate` from ANY live slot, a hull tet included (§116.6: a walk that
/// starts on a hull tet starts from its finite neighbour): the answer is a
/// finite tet whose closure holds `q`, or a hull tet whose face `q` lies
/// strictly beyond. Every hull slot is tried as the start.
#[test]
fn delaunay_both_builds_locate_from_hull_start() {
    use ofgpu::tetmesh::predicates::orient3d;
    let pts = r_points(2000, 2026);
    let d = Delaunay::build(&pts).unwrap();
    let qs = r_points(200, 2028);
    let mut hull_starts = 0;
    for h in 0..d.slots() as u32 {
        if !d.is_live(h) || !d.is_hull(h) {
            continue;
        }
        hull_starts += 1;
        for q in &qs {
            let loc = d.locate(q, h).unwrap();
            let v = d.tet(loc.tet);
            let p = |i: u32| &pts[i as usize];
            if d.is_hull(loc.tet) {
                assert!(orient3d(p(v[0]), p(v[1]), p(v[2]), q) > 0.0, "hull start {}", h);
            } else {
                let faces = [[v[1], v[3], v[2]], [v[0], v[2], v[3]], [v[0], v[3], v[1]], [v[0], v[1], v[2]]];
                for f in faces {
                    assert!(orient3d(p(f[0]), p(f[1]), p(f[2]), q) >= 0.0, "hull start {}", h);
                }
            }
        }
    }
    assert_eq!(hull_starts, 156);
}

/// FNV-1a 64 over the order, each index as 4 little-endian bytes.
fn order_digest(order: &[u32]) -> u64 {
    let mut h = 0xcbf29ce484222325_u64;
    for &i in order {
        for b in i.to_le_bytes() {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x100000001b3);
        }
    }
    h
}

/// The insertion order of §116.7, pinned in BOTH builds.
#[test]
fn order_both_builds_pins() {
    let pts = r_points(1000, 2026);
    assert_eq!(
        order_digest(&brio_order(&pts, BRIO_SEED)),
        0xed3e42d856fec369
    );
    let l3 = lattice(3);
    assert_eq!(
        order_digest(&brio_order(&l3, BRIO_SEED)),
        0x4a2afd7c88e3665e
    );
    assert_eq!(brio_rounds(1000), vec![0, 64, 128, 256, 512, 1000]);
}

/// The BRIO entry point on 2000 uniform points: the same triangulation
/// the input-order build gives, in BOTH builds.
#[test]
fn delaunay_both_builds_brio_random2000() {
    let pts = r_points(2000, 2026);
    let d = Delaunay::build_brio(&pts).unwrap();
    d.check().unwrap();
    assert_eq!(d.finite_tets().len(), 12967);
    assert_eq!(tet_digest(&d.finite_tets()), 0xb88702ffa69aeb56);
}
