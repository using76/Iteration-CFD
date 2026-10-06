// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! SPEC-LIT §118.2 — the host mesh geometry is computed in f64: the points
//! are read as `f64`, the whole sweep runs in `f64`, and every value the
//! `Scalar` arrays store is rounded once. An integration test because the
//! library's own test target does not compile under `single`; an integration
//! test links the normal (non-test) library, which does. It runs in BOTH
//! builds with the same assertions — precision class A of docs/17 §5.1 rule
//! 4: a geometric identity evaluated in f64, SAME tolerance in both builds.
//! No GPU needed.
//!
//! Provenance: ORIGINAL — the meshes are this tree's own generators; no
//! GPL-licensed source was consulted.

use ofgpu::blockgen::{self, BlockSpec, GradedAxis};
use ofgpu::io::polymesh::{build_host_mesh, read_poly_mesh, write_poly_mesh_raw, PolyMeshRaw};
use ofgpu::mesh::refined::refined_core;
use ofgpu::types::{to_dvec3, to_vec3};
use ofgpu::{DVec3, HostMesh, Scalar, Vec3};

/// `ofgpu-validate`'s `make_mesh`, minus the shear: a `BlockSpec` whose axes
/// span exactly `[0, l_i]` in the build's own `Scalar`.
fn spec(n: [usize; 3], l: [f64; 3], e: [f64; 3], two_d: bool) -> BlockSpec {
    let axis = |i: usize| GradedAxis {
        lo: 0.0,
        hi: l[i] as Scalar,
        n: n[i],
        expansion: e[i] as Scalar,
        two_sided: e[i] != 1.0,
    };
    let w = if two_d { "empty" } else { "wall" };
    BlockSpec {
        x: axis(0),
        y: axis(1),
        z: axis(2),
        windows: Vec::new(),
        patch_name: BlockSpec::default().patch_name,
        patch_type: ["patch", "patch", "wall", "wall", w, w].map(String::from),
        cyclic: Vec::new(),
    }
}

/// The analytic volume of the box the generator actually built, whose extents
/// are the `Scalar` lengths.
fn volume(l: [f64; 3]) -> f64 {
    f64::from(l[0] as Scalar) * f64::from(l[1] as Scalar) * f64::from(l[2] as Scalar)
}

fn graded_spec() -> BlockSpec {
    spec([14, 11, 9], [1.0, 0.7, 0.4], [1.0, 8.0, 1.0], false)
}

fn graded_mesh() -> HostMesh {
    blockgen::build_mesh(&graded_spec()).expect("graded")
}

fn sheared_mesh() -> HostMesh {
    let mut raw = blockgen::raw_mesh(&spec(
        [9, 8, 7],
        [1.0, 0.7, 0.4],
        [1.0, 1.0, 1.0],
        false,
    ))
    .expect("raw");
    for p in raw.points.iter_mut() {
        p.x += 0.45 * p.z;
        p.y += 0.5 * 0.45 * p.z;
    }
    build_host_mesh(&raw).expect("sheared")
}

fn two_d_mesh() -> HostMesh {
    blockgen::build_mesh(&spec([20, 16, 1], [1.0, 0.7, 0.05], [1.0, 5.0, 1.0], true))
        .expect("two_d")
}

fn refined_mesh() -> HostMesh {
    refined_core(
        [8, 8, 8],
        Vec3::new(0.125, 0.125, 0.125),
        0.25,
        1,
    )
    .expect("refined")
    .mesh
}

/// The graded mesh, translated 2000 in x and y: big coordinates, so the
/// rounding of a coordinate difference is where the error comes from.
fn translated_mesh() -> HostMesh {
    let mut raw = blockgen::raw_mesh(&graded_spec()).expect("raw");
    for p in raw.points.iter_mut() {
        p.x += 2000.0;
        p.y += 2000.0;
    }
    build_host_mesh(&raw).expect("translated")
}

/// Every mesh of this file, by name, with the analytic volume its block spans.
fn meshes() -> Vec<(&'static str, HostMesh, f64)> {
    vec![
        ("graded", graded_mesh(), volume([1.0, 0.7, 0.4])),
        ("sheared", sheared_mesh(), volume([1.0, 0.7, 0.4])),
        ("two_d", two_d_mesh(), volume([1.0, 0.7, 0.05])),
        ("refined", refined_mesh(), 1.0),
        ("translated", translated_mesh(), volume([1.0, 0.7, 0.4])),
    ]
}

/// The closure identity `Σ_f s Sf = 0` holds to f64 round-off on the f64 view
/// of every mesh. Single at HEAD: graded 2.934e-7, sheared 2.756e-7, two_d
/// 1.963e-7, translated 1.515e-7 — this test is what fails, and the numbers
/// §118.2 records.
#[test]
fn host_geometry_closes_at_f64_tolerance() {
    for (name, m, _) in meshes() {
        let e = f64::from(m.check().max_closure_error);
        eprintln!("closure[{name}] = {e:.6e}");
        assert!(e <= 1e-12, "{name}: closure {e:e} > 1e-12");
    }
}

/// The sweep's total volume is the block's analytic volume to f64 round-off.
/// Single at HEAD: graded 2.442e-6, sheared 4.321e-7, two_d 2.135e-6,
/// translated 1.313e-4 — this test is what fails.
#[test]
fn host_volume_matches_the_block_at_f64_tolerance() {
    for (name, m, v) in meshes() {
        let total = f64::from(m.check().total_volume);
        let e = (total - v).abs() / v;
        eprintln!("volume[{name}] = {e:.6e} (total {total:.17} vs {v:.17})");
        assert!(e <= 1e-12, "{name}: volume error {e:e} > 1e-12");
    }
}

/// A point coordinate survives a write/read round trip exactly: the reader
/// stores the parsed f64 and the writer writes 17 significant digits.
/// Single at HEAD this fails — the coordinate is rounded to f32.
#[test]
fn points_are_read_and_written_in_f64() {
    let raw: PolyMeshRaw = blockgen::raw_mesh(&spec(
        [2, 2, 2],
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
        false,
    ))
    .expect("raw");
    let x0: Vec<f64> = raw.points.iter().map(|p| f64::from(p.x)).collect();

    let mut raw = raw;
    for p in raw.points.iter_mut() {
        p.x += 2000.000000123;
    }

    let dir = std::env::temp_dir().join("ofgpuF32GeometryPoints");
    let _ = std::fs::remove_dir_all(&dir);
    write_poly_mesh_raw(&dir, &raw).expect("write");
    let read = read_poly_mesh(&dir).expect("read");
    let _ = std::fs::remove_dir_all(&dir);

    for k in 0..raw.points.len() {
        assert_eq!(
            f64::from(read.points[k].x),
            x0[k] + 2000.000000123_f64,
            "point {k}: x did not round-trip in f64"
        );
        assert_eq!(
            f64::from(read.points[k].y).to_bits(),
            f64::from(raw.points[k].y).to_bits(),
            "point {k}: y changed in the round trip"
        );
        assert_eq!(
            f64::from(read.points[k].z).to_bits(),
            f64::from(raw.points[k].z).to_bits(),
            "point {k}: z changed in the round trip"
        );
    }
}

/// The graded mesh's report is the f64 build's: closure at round-off, one
/// region, every volume positive, LDU ordered.
#[test]
fn the_graded_report_is_the_f64_build_report() {
    let m = graded_mesh();
    let r = m.check();
    assert!(
        f64::from(r.max_closure_error) <= 1e-12,
        "closure {}",
        r.max_closure_error
    );
    assert_eq!(r.n_regions, 1);
    assert!(f64::from(r.min_volume) > 0.0, "min volume {}", r.min_volume);
    assert!(r.ldu_ordered);
}

/// The conversions of SPEC-LIT §118.2, at the public API: widening a point
/// into the host geometry is exact, and the round trip through the two is one
/// rounding, lossless on a value that IS a `Scalar`.
#[test]
fn dvec3_conversions_round_once() {
    let p = DVec3::new(0.1, -2000.000000123, 1.0e-30);
    let want = DVec3::new(
        f64::from(0.1 as Scalar),
        f64::from(-2000.000000123 as Scalar),
        f64::from(1.0e-30 as Scalar),
    );
    let got = to_dvec3(to_vec3(p));
    assert_eq!(got.x.to_bits(), want.x.to_bits());
    assert_eq!(got.y.to_bits(), want.y.to_bits());
    assert_eq!(got.z.to_bits(), want.z.to_bits());

    let v = Vec3::new(0.1, 0.7, -3.5);
    let back = to_vec3(to_dvec3(v));
    assert_eq!(back.x.to_bits(), v.x.to_bits());
    assert_eq!(back.y.to_bits(), v.y.to_bits());
    assert_eq!(back.z.to_bits(), v.z.to_bits());
}

/// Every one of the 16 arrays the sweep fills is the f64 geometry rounded
/// ONCE: the view's element rounds to the stored `Scalar` bit for bit, and
/// the closure identity recomputed from the VIEW in f64 holds at the f64
/// tolerance.
#[test]
fn the_scalar_arrays_are_the_f64_geometry_rounded_once() {
    for (name, m) in [("graded", graded_mesh()), ("translated", translated_mesh())] {
        assert!(m.geom64_is_shadow(), "{name}: the view does not borrow a shadow");
        let g = m.geom64();

        let cmp_s = |what: &str, view: &[f64], stored: &[Scalar]| {
            assert_eq!(view.len(), stored.len(), "{name}: {what} length");
            for (i, (&a, &b)) in view.iter().zip(stored.iter()).enumerate() {
                assert_eq!(
                    (a as Scalar).to_bits(),
                    b.to_bits(),
                    "{name}: {what}[{i}] is not the f64 value rounded once"
                );
            }
        };
        let cmp_v = |what: &str, view: &[DVec3], stored: &[Vec3]| {
            assert_eq!(view.len(), stored.len(), "{name}: {what} length");
            for (i, (a, b)) in view.iter().zip(stored.iter()).enumerate() {
                let rounded = to_vec3(*a);
                assert_eq!(
                    rounded.x.to_bits(),
                    b.x.to_bits(),
                    "{name}: {what}[{i}].x is not the f64 value rounded once"
                );
                assert_eq!(
                    rounded.y.to_bits(),
                    b.y.to_bits(),
                    "{name}: {what}[{i}].y is not the f64 value rounded once"
                );
                assert_eq!(
                    rounded.z.to_bits(),
                    b.z.to_bits(),
                    "{name}: {what}[{i}].z is not the f64 value rounded once"
                );
            }
        };

        cmp_s("v", &g.v, &m.v);
        cmp_v("c", &g.c, &m.c);
        cmp_v("sf", &g.sf, &m.sf);
        cmp_s("mag_sf", &g.mag_sf, &m.mag_sf);
        cmp_v("cf", &g.cf, &m.cf);
        cmp_s("weights", &g.weights, &m.weights);
        cmp_s("delta_coeffs", &g.delta_coeffs, &m.delta_coeffs);
        cmp_v("non_orth_corr", &g.non_orth_corr, &m.non_orth_corr);
        cmp_v("skew_corr", &g.skew_corr, &m.skew_corr);
        cmp_v("b_sf", &g.b_sf, &m.b_sf);
        cmp_s("b_mag_sf", &g.b_mag_sf, &m.b_mag_sf);
        cmp_v("b_cf", &g.b_cf, &m.b_cf);
        cmp_s("b_delta_coeffs", &g.b_delta_coeffs, &m.b_delta_coeffs);
        cmp_v("b_non_orth_corr", &g.b_non_orth_corr, &m.b_non_orth_corr);
        cmp_s("b_y", &g.b_y, &m.b_y);
        cmp_s("b_weights", &g.b_weights, &m.b_weights);

        // The closure identity, recomputed in this test from the VIEW alone.
        let mut closure = vec![DVec3::ZERO; m.n_cells];
        for f in 0..m.n_internal_faces {
            closure[m.owner[f] as usize] += g.sf[f];
            closure[m.neighbour[f] as usize] -= g.sf[f];
        }
        for bf in 0..m.n_boundary_faces {
            closure[m.b_face_cells[bf] as usize] += g.b_sf[bf];
        }
        let mut worst = 0.0f64;
        for cell in 0..m.n_cells {
            let scale = g.v[cell]
                .abs()
                .max(f64::MIN_POSITIVE)
                .powf(2.0 / 3.0);
            worst = worst.max(closure[cell].mag() / scale);
        }
        assert!(worst <= 1e-12, "{name}: view closure {worst:e} > 1e-12");
    }
}

/// The view never contradicts the stored arrays: touch one stored volume and
/// the whole view falls back to the widened arrays (all or nothing), and
/// after `release_geom64` the same assertions hold from the widening alone.
#[test]
fn the_view_never_contradicts_the_stored_arrays() {
    let mut m = graded_mesh();
    m.v[0] = m.v[0] * 2.0;

    assert_eq!(m.geom64().v[0], f64::from(m.v[0]));
    assert_eq!(m.geom64().v.len(), m.n_cells);

    m.release_geom64();
    assert_eq!(m.geom64().v[0], f64::from(m.v[0]));
    assert_eq!(m.geom64().v.len(), m.n_cells);
}
