// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! AUT-03: the snap row's k = 16 worst boundary points, gated in BOTH
//! builds - the library's own test target does not compile under `single`,
//! an integration test links the normal (non-test) library, which does.
//! Precision class D of docs/17 §5.1 rule 4 - discrete / bitwise, the same
//! assertion in both builds. No GPU needed.
//!
//! Provenance: ORIGINAL - the hand feature graph and the cube are this
//! tree's own fixtures; no GPL-licensed source was consulted.

use ofgpu::automesher::driver::{self, Stage};
use ofgpu::automesher::features::{FeatureIndex, FeatureSet};
use ofgpu::automesher::read_config_str;
use ofgpu::automesher::snap::{feature_class, FeatureClass};
use ofgpu::surface::stl::parse_stl;
use ofgpu::{Scalar, Vec3};

/// (92.76) on a hand feature graph, in both builds: the free end of an open
/// feature curve names "open-tip", a three-edge corner "corner", a point
/// h_f from an edge "edge" and one further off "smooth" - and the corner
/// wins over the edges it lies on. Every value is binary-exact, so the two
/// builds agree bit for bit.
#[test]
fn the_feature_class_of_a_point_is_its_nearest_feature_within_r() {
    let fs = FeatureSet {
        points: vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
        ],
        edges: vec![[0, 1], [1, 2], [1, 3]],
        edge_patches: vec![vec![0], vec![0], vec![0]],
        polylines: vec![vec![0, 1], vec![1, 2], vec![1, 3]],
        corners: vec![0, 1, 2, 3],
        feature_angle_deg: 30.0,
    };
    let fidx = FeatureIndex::new(&fs, 1.0).expect("index");
    let r: Scalar = 0.125;
    let cases: &[([f64; 3], FeatureClass)] = &[
        ([0.0, 0.0, 0.0625], FeatureClass::OpenTip),
        ([2.0, 0.0, 0.0625], FeatureClass::OpenTip),
        ([1.0, 1.0, 0.0625], FeatureClass::OpenTip),
        ([1.0, 0.0, 0.0625], FeatureClass::Corner),
        ([0.03125, 0.0, 0.0], FeatureClass::OpenTip),
        ([0.5, 0.0, 0.125], FeatureClass::Edge),
        ([0.5, 0.0, 0.25], FeatureClass::Smooth),
    ];
    for &(x, want) in cases {
        let p = Vec3::new(x[0] as Scalar, x[1] as Scalar, x[2] as Scalar);
        assert_eq!(feature_class(&fs, Some(&fidx), p, r), want, "{x:?}");
    }
    assert_eq!(
        feature_class(&fs, None, Vec3::ZERO, r),
        FeatureClass::Smooth,
        "no index, no feature"
    );
    assert_eq!(FeatureClass::Edge.as_str(), "edge");
    assert_eq!(FeatureClass::Corner.as_str(), "corner");
    assert_eq!(FeatureClass::Smooth.as_str(), "smooth");
    assert_eq!(FeatureClass::OpenTip.as_str(), "open-tip");
}

/// The cube `[1.3, 2.3]^3` as an ASCII STL: 12 OUTWARD-wound triangles in
/// one `solid cube` block. `Surface::from_soup` is `pub(crate)`, so the
/// hand-written triangles reach it through the public [`parse_stl`], which
/// lands on it with the patch the solid names.
fn cube_stl() -> Vec<u8> {
    let (lo, hi) = ([1.3f64, 1.3, 1.3], [2.3f64, 2.3, 2.3]);
    let v: [[f64; 3]; 8] = [
        [lo[0], lo[1], lo[2]],
        [hi[0], lo[1], lo[2]],
        [hi[0], hi[1], lo[2]],
        [lo[0], hi[1], lo[2]],
        [lo[0], lo[1], hi[2]],
        [hi[0], lo[1], hi[2]],
        [hi[0], hi[1], hi[2]],
        [lo[0], hi[1], hi[2]],
    ];
    let t: [[usize; 3]; 12] = [
        [0, 2, 1], [0, 3, 2], // -z
        [4, 5, 6], [4, 6, 7], // +z
        [0, 1, 5], [0, 5, 4], // -y
        [2, 3, 7], [2, 7, 6], // +y
        [0, 4, 7], [0, 7, 3], // -x
        [1, 2, 6], [1, 6, 5], // +x
    ];
    let mut s = String::from("solid cube\n");
    for tr in &t {
        s.push_str("facet normal 0 0 0\nouter loop\n");
        for &i in tr {
            s.push_str(&format!("vertex {} {} {}\n", v[i][0], v[i][1], v[i][2]));
        }
        s.push_str("endloop\nendfacet\n");
    }
    s.push_str("endsolid cube\n");
    s.into_bytes()
}

/// The driver's own smallest config, as text: the 1 m cube at the centre of
/// a `[0, 4]^3` domain at base size 1, one level deep, and no layers.
fn cube_config_text() -> String {
    String::from(
        "{ \"input\": { \"surfaces\": [ { \"path\": \"cube.stl\" } ] },\n\
         \"domain\": { \"extent\": [0,4,0,4,0,4], \"base_size\": 1.0 },\n\
         \"refinement\": { \"levels\": [ { \"patch\": \"cube\", \"bands\": \
         [ { \"distance\": 0.0, \"level\": 1 } ], \"feature_level\": 0 } ], \
         \"feature_angle_deg\": 30.0, \"max_level\": 1 },\n\
         \"output\": { \"case_dir\": \"out\", \"name\": \"cube\" } }",
    )
}

/// AUT-03's gate on a real run, in both builds: the snap row of the cube
/// run stopped after snap carries min(16, n_boundary_points) rows whose
/// first `residual_m` is the row's own `max_residual` bit for bit.
#[test]
fn the_snap_row_of_a_cube_run_names_its_worst_points() {
    let surf = parse_stl(&cube_stl(), "cube", "aut-03 cube").expect("surface");
    let cfg = read_config_str(&cube_config_text(), "aut-03").expect("config");
    let mut lines: Vec<String> = Vec::new();
    let out = driver::run(
        &cfg,
        &surf,
        Some(Stage::parse("snap").expect("stage")),
        &mut |l: &str| lines.push(l.to_string()),
    )
    .expect("run to snap");
    let row = &out.stages[2];
    assert_eq!(row.stage, Stage::Snap);
    let n_b = row.counts["n_boundary_points"].as_u64().unwrap() as usize;
    let wps = row.counts["worst_points"].as_array().expect("worst_points");
    assert_eq!(wps.len(), n_b.min(16), "min(16, n_boundary_points)");
    let r0 = wps[0]["residual_m"].as_f64().unwrap();
    let max_r = row.counts["max_residual"].as_f64().unwrap();
    assert_eq!(
        r0.to_bits(),
        max_r.to_bits(),
        "row 0's residual is max_residual"
    );
}
