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
//!
//! Provenance: ORIGINAL. Written from Shewchuk (1998), Murphy, Mount &
//! Gable (2001) and Cohen-Steiner, Colin de Verdière & Yvinec (2004); the
//! choices are derived in SPEC-LIT §116.11. No GPL-licensed source was
//! consulted.

use ofgpu::tetmesh::recover::{recover_edges, recover_remeshed, EdgeRecovery};
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
