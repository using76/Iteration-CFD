// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! TET-11's integration test: the emission of SPEC-LIT §116.16 through the
//! public API, in both builds. The f32 rounding check (116.26) is exercised
//! by a flat tetrahedron 1e4 m from the origin: its base lies in the plane
//! `z = 10000`, which f32 holds exactly, and its apex `3 * 2^-13` m above
//! it, three eighths of an f32 ulp at 1e4 (`2^-10`), so rounding to f32
//! puts the apex on the base and (116.2) becomes 0. Under `single` the
//! emitter refuses it, naming cell 0; under f64 it is accepted and passes
//! §92.3. Every coordinate is a multiple of `2^-13`, so the f64 geometry is
//! exact up to the products.
//!
//! Provenance: ORIGINAL. The fixture is this tree's own construction. No
//! GPL-licensed source was consulted.

use ofgpu::automesher::quality::QualityThresholds;
use ofgpu::tetmesh::emit::{emit, rounding_flips, EmitPatch};
use ofgpu::tetmesh::predicates::orient3d;
use ofgpu::tetmesh::Point;

/// The outward faces of a tetrahedron with `orient3d > 0` (§116.4).
const FACES: [[usize; 3]; 4] = [[0, 1, 2], [0, 3, 1], [0, 2, 3], [1, 3, 2]];

fn sliver() -> (Vec<Point>, Vec<[u32; 4]>) {
    let (o, l, d) = (1.0e4, 2f64.powi(-7), 3.0 * 2f64.powi(-13));
    let pts: Vec<Point> = vec![
        [o, o, o],
        [o + l, o, o],
        [o, o + l, o],
        [o + l / 4.0, o + l / 4.0, o + d],
    ];
    let mut t = [0u32, 1, 2, 3];
    let p = t.map(|v| pts[v as usize]);
    if orient3d(&p[0], &p[1], &p[2], &p[3]) < 0.0 {
        t.swap(0, 1);
    }
    (pts, vec![t])
}

fn faces(t: [u32; 4]) -> Vec<([u32; 3], u32)> {
    FACES.iter().map(|r| (r.map(|i| t[i]), 0)).collect()
}

#[test]
fn the_build_width_is_reported() {
    println!(
        "Scalar is {} bytes; the tetmesh geometry is f64 in both builds",
        std::mem::size_of::<ofgpu::Scalar>()
    );
}

#[test]
fn f32_flip_refused() {
    let (pts, tets) = sliver();
    let p = tets[0].map(|v| pts[v as usize]);
    assert!(orient3d(&p[0], &p[1], &p[2], &p[3]) > 0.0);
    let patches = vec![EmitPatch { name: "walls".to_string(), type_name: "wall".to_string() }];
    let flips = rounding_flips(&pts, &tets);
    let r = emit(&pts, &tets, &faces(tets[0]), &patches, &QualityThresholds::default());
    let single = std::mem::size_of::<ofgpu::Scalar>() == 4;
    match &r {
        Ok((_, q)) => println!(
            "emit 1e4 m sliver: accepted, rounding flips {flips:?}; 92.3 min tau {:.4}, max cond {:.1}, max closure {:.2e}",
            q.min_thickness_ratio, q.max_cond, q.max_closure
        ),
        Err(e) => println!("emit 1e4 m sliver: refused, rounding flips {flips:?}: {e}"),
    }
    if single {
        assert_eq!(flips, vec![0]);
        let e = r.err().expect("refused under single").to_string();
        assert!(e.contains("emit/f32: rounding the points to f32 makes 1 of 1 tetrahedra flat or inverted"), "{e}");
        assert!(e.contains("cells 0;") && e.contains("local origin"), "{e}");
    } else {
        assert!(flips.is_empty());
        let (e, q) = r.expect("accepted under f64");
        assert!(q.passed());
        assert_eq!((e.raw.points.len(), e.raw.faces.len(), e.raw.neighbour.len()), (4, 4, 0));
    }
}

